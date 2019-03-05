import numpy as np 
import stargrit as sg
import quadpy.sphere as quadsph
import scipy.interpolate as spint
from stargrit.structure import potentials
from stargrit.radiative_transfer.cobain.general import RadiativeTransfer, DiffrotStarRadiativeTransfer, ContactBinaryRadiativeTransfer
from stargrit.geometry.spherical import SphericalMesh
from stargrit.geometry.cylindrical import CylindricalMesh
import astropy.units as u 
import astropy.constants as c
from stargrit.radiative_transfer.quadrature.lebedev import Lebedev 
from stargrit.radiative_transfer.quadrature.gauss_legendre import Gauss_Legendre


class GrayRadiativeTransfer(RadiativeTransfer):


    def _mesh_interpolation_functions(self, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points.

        If the opacity and atmosphere are mean/gray, the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        directory = self.star.directory
        mesh = self.star.mesh 

        RGI = spint.RegularGridInterpolator

        chi = np.load(directory + 'chi%s_%i.npy' % (component, iter_n-1))
        J = np.load(directory + 'S%s_%i.npy' % (component, iter_n-1))

        key1, key2, key3 = self.star.mesh.coords.keys()

        chi_interp = RGI(points=[mesh.coords[key1], mesh.coords[key2], mesh.coords[key3]], values=chi)
        J_interp = RGI(points=[mesh.coords[key1], mesh.coords[key2], mesh.coords[key3]], values=J)

        I_interp = []
        I = np.load(directory + 'I%s_%i.npy' % (component,iter_n-1))
        for i in range(self.quadrature.nI):
            I_interp.append(RGI(points=[mesh.coords[key1], mesh.coords[key2], mesh.coords[key3]], values=I[i]))
    
        return chi_interp, J_interp, I_interp


    def _adjust_stepsize(self, Mc, ndir, dirarg):

        #TODO: implement monochromatic vs mean

        exps = np.linspace(-10, -4, 1000)
        stepsizes = 10 ** exps * self.star.mesh.default_units['r']

        b = np.ones((len(stepsizes), 3)).T * stepsizes

        rs = Mc - ndir * b.T

        chis = self._compute_structure(points=(rs/self.star.structure.scale), dirarg=dirarg,  
                                        stepsize=True)
        # compute stepsize
        taus = chis * stepsizes / 2.

        diffs = np.abs(taus - 1.*taus.unit)
        stepsize_final = stepsizes[np.argmin(diffs)]

        # print 'stepsize: ', stepsize_final
        return stepsize_final / 2.


    def _intensity_step(self, Mc, ndir, dirarg, stepsize, N):
        
        rs = np.array([Mc.to(stepsize.unit).value - i * stepsize.value * ndir for i in range(N)])*stepsize.unit
        paths = np.array([i * stepsize.value for i in range(N)])*stepsize.unit

        chis, Ss, Is = self._compute_structure(points=(rs/self.star.structure.scale), 
        dirarg=dirarg, stepsize=False)
                          
        return self._intensity_integral(paths,chis,Ss,Is,N)


    def _intensity_integral(self, paths, chis, Ss, Is, N, spline_order=1, test=False):

        #TODO: implement gray vs monochromatic, handle limits in lambda? - in radiative equlibrium
        chis_sp = spint.UnivariateSpline(paths, chis, k=spline_order, s=0)
        taus = np.array([chis_sp.integral(paths[0].value, paths[i].value) for i in range(N)])
        Ss_exp = Ss * np.exp((-1.)*taus)

        if test:
            import matplotlib.pyplot as plt
            plt.plot(taus, Ss_exp)
            plt.ylabel('Ss_exp')
            plt.show()

            plt.plot(taus, Ss)
            plt.ylabel('Ss')
            plt.show()

            plt.plot(taus, Is)
            plt.ylabel('Is')
            plt.show()


        Iszero = np.argwhere(Is == 0.0).flatten()
        Ssezero = np.argwhere(Ss_exp == 0.0).flatten()

        if Iszero.size == 0 and Ssezero.size == 0:
            nbreak = N - 1
        else:
            nbreak = np.min(np.hstack((Iszero, Ssezero)))

        if nbreak > 1:
            taus = taus[:nbreak + 1]
            Ss_exp = Ss_exp[:nbreak + 1]
            Ss_exp[(Ss_exp == np.nan) | (Ss_exp == np.inf) | (Ss_exp == -np.nan) | (Ss_exp == -np.inf)] = 0.0

            taus_u, indices = np.unique(taus, return_index=True)

            if len(taus_u) > 1:
                Sexp_sp = spint.UnivariateSpline(taus[indices], Ss_exp[indices], k=spline_order, s=0)
                I = Is[nbreak] * np.exp(-taus[-1]) + Sexp_sp.integral(taus[0], taus[-1])
                # print 'I = %s, I0 = %s, Sint = %s' % (I, Is[nbreak], Sexp_sp.integral(taus[0], taus[-1]))
            else:
                I = 0.0

            if np.isnan(I) or I < 0.:
                I = 0.0

        else:
            taus = taus[:nbreak + 1]
            I = 0.0

        return nbreak, taus[-1], Is[nbreak], I


    def _compute_intensity(self, Mc, n):

        R = self._transformation_matrix(n)

        Is_j = np.zeros(self.quadrature.nI)
        taus_j = np.zeros(self.quadrature.nI)

        for dirarg in range(self.quadrature.nI):
            coords = self.quadrature.azimuthal_polar[dirarg]
            # print 'Computing direction %s, coords %s' % (dirarg,coords)
            ndir = self._rotate_direction_wrt_normal(Mc=Mc, coords=coords, R=R)

            stepsize = self._adjust_stepsize(Mc, ndir, dirarg)
            
            nbreak, tau, I0, I = self._intensity_step(Mc, ndir, dirarg, stepsize, self._N)
            
            subd = 'no'
            N = self._N
            if (nbreak < 1000 and I0 != 0.0) or nbreak == self._N-1:
                if (nbreak < 1000 and I0 != 0.0):
                    subd = 'yes'
                    div = 1000. / nbreak
                    stepsize = stepsize / div
                elif nbreak == self._N - 1:
                    subd = 'yes'
                    N = self._N * 2
                else:
                    subd = 'yes'
                    div = 1000. / nbreak
                    stepsize = stepsize / div
                    N = self._N * 2

                nbreak, tau, I0, I = self._intensity_step(Mc, ndir, dirarg, stepsize, N)

            taus_j[dirarg], Is_j[dirarg] = tau, I

        return Is_j, taus_j, self._compute_mean_intensity(Is_j), self._compute_flux(Is_j)


    def _compute_structure(self, points, dirarg, stepsize=False):
        raise NotImplementedError

    
    def _compute_mean_intensity(self, I):

        if isinstance(self.quadrature, Lebedev):
            return self.quadrature.integrate_over_4pi(I)
        elif isinstance(self.quadrature, Gauss_Legendre):
            return self.quadrature.integrate_over_4pi(I[2:].reshape((len(self.quadrature.thetas),len(self.quadrature.phis))))
        else:
            raise TypeError('Unrecognized quadrature type %s' % self.quadrature)


    def _compute_flux(self, I):

        #BUG: this wouldn't hold near the neck of contacts because of the concavity
        if isinstance(self.quadrature, Lebedev):
            return self.quadrature.integrate_outer_m_inner(I)
        elif isinstance(self.quadrature, Gauss_Legendre):
            return self.quadrature.integrate_outer_m_inner(I[2:].reshape((len(self.quadrature.thetas),len(self.quadrature.phis))))
        else:
            raise TypeError('Unrecognized quadrature type %s' % self.quadrature)


    def _compute_temperature(self, JF, ttype='J'):
        
        if ttype == 'J':
            return ((np.pi*JF*self.star.atmosphere.default_units['S']/c.sigma_sb)**(0.25)).to(u.K)

        elif ttype == 'F':
            return ((JF*self.star.atmosphere.default_units['S']/c.sigma_sb)**(0.25)).to(u.K)

        else:
            raise ValueError('Type for temperature computation can only be J or F.')


class DiffrotStarGrayRadiativeTransfer(GrayRadiativeTransfer, DiffrotStarRadiativeTransfer):


    def _compute_structure(self, points, dirarg, stepsize=False, test=False):

        pot_range_grid = [self.star.mesh.coords['pots'].min(), self.star.mesh.coords['pots'].max()]
        pots = self._compute_potentials(points, self._interp_funcs['bbT'], pot_range_grid)
        grid, le = self._compute_interp_regions(pots, points, pot_range_grid)
        thetas, phis = self._compute_coords_for_interpolation(points)
        rhos, Ts = self._interp_funcs['bbrho'](pots[le]), self._interp_funcs['bbT'](pots[le])
        rhos = rhos * self.star.structure.default_units['rho']
        Ts = Ts * self.star.structure.default_units['T']
        chis = np.zeros(len(pots))
        chis[grid] = self._interp_funcs['chi']((pots[grid], thetas[grid], phis[grid]))
        chis[le] = self.star.atmosphere._compute_absorption_coefficient(rhos, Ts)

        if test:
            import matplotlib.pyplot as plt

            pots_mesh = np.zeros(len(self.star.mesh.rs))
            pot_len = self.star.mesh.dims[1]*self.star.mesh.dims[2]
            for i in range(self.star.mesh.dims[0]):
                pots_mesh[i*pot_len:(i+1)*pot_len] = self.star.mesh.coords['pots'][i]
            
            chis_0 = np.load(self.star.directory+'chi_0.npy').flatten()

            plt.plot(pots_mesh, chis_0, label='chi_0')
            plt.plot(pots, chis, '--', label='chi_interp')
            plt.xlim(pots.min(),pots.max())
            plt.ylim(chis.min(), chis.max())
            plt.legend()
            plt.show()
    
        if stepsize:
            return chis

        else:
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

            Ss[grid] = self._interp_funcs['S']((pots[grid], thetas[grid], phis[grid]))
            Is[grid] = self._interp_funcs['I'][dirarg]((pots[grid], thetas[grid], phis[grid]))
            Ss[le] = Is[le] = self.star.atmosphere._compute_source_function(Ts)

            if test:
                import matplotlib.pyplot as plt

                pots_mesh = np.zeros(len(self.star.mesh.rs))
                pot_len = self.star.mesh.dims[1]*self.star.mesh.dims[2]
                for i in range(self.star.mesh.dims[0]):
                    pots_mesh[i*pot_len:(i+1)*pot_len] = self.star.mesh.coords['pots'][i]
                
                Ss_0 = np.load(self.star.directory+'S_0.npy').flatten()

                plt.plot(pots_mesh, Ss_0, label='S_0')
                plt.plot(pots, Ss, '--', label='S_interp')
                plt.xlim(pots.min(),pots.max())
                plt.ylim(Ss.min(), Ss.max())
                plt.legend()
                plt.show()

            return chis, Ss, Is


class ContactBinaryGrayRadiativeTransfer(GrayRadiativeTransfer, ContactBinaryRadiativeTransfer):


    def _compute_structure(self, points, dirarg, stepsize=False):
        """
        Returns the radiative structure in all points along a ray.
        """
        pot_range_grid = [self.star.mesh.coords['pots'].min(), self.star.mesh.coords['pots'].max()]
        pots = self._compute_potentials(points, self.star.q, self._interp_funcs['bbT1'], self._interp_funcs['bbT2'], pot_range_grid)
        pots2 = pots / self.star.q + 0.5 * (self.star.q - 1) / self.star.q

        if stepsize:
            chis = np.zeros(len(pots))
        else:
            chis = np.zeros(len(pots))
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

        if isinstance(self.star.mesh, SphericalMesh):
            
            grid_prim, grid_sec, le_prim, le_sec = self._compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid)
            # thetas and phis are returned for all points (zeros out of grid)... maybe fix this?
            thetas, phis = self._compute_coords_for_interpolation(points, grid_prim=grid_prim, grid_sec=grid_sec)

            if stepsize:
                chis[grid_prim] = self._interp_funcs['chi1']((pots[grid_prim], thetas[grid_prim], phis[grid_prim]))
                chis[grid_sec] = self._interp_funcs['chi2']((pots[grid_sec], thetas[grid_sec], phis[grid_sec]))

            else:
                chis[grid_prim] = self._interp_funcs['chi1']((pots[grid_prim], thetas[grid_prim], phis[grid_prim]))
                chis[grid_sec] = self._interp_funcs['chi2']((pots[grid_sec], thetas[grid_sec], phis[grid_sec]))
                Ss[grid_prim] = self._interp_funcs['S1']((pots[grid_prim], thetas[grid_prim], phis[grid_prim]))
                Ss[grid_sec] = self._interp_funcs['S2']((pots[grid_sec], thetas[grid_sec], phis[grid_sec]))
                Is[grid_prim] = self._interp_funcs['I1'][dirarg]((pots[grid_prim], thetas[grid_prim], phis[grid_prim]))
                Is[grid_sec] = self._interp_funcs['I2'][dirarg]((pots[grid_sec], thetas[grid_sec], phis[grid_sec]))

        elif isinstance(self.star.mesh, CylindricalMesh):

            grid, le_prim, le_sec = self._compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid)
            # here xnorms and thetas are only those pertaining to grid points
            xnorms, thetas = self._compute_coords_for_interpolation(points, grid=grid, pots=pots)
            
            if stepsize:
                chis[grid] = self._interp_funcs['chi']((pots[grid], xnorms, thetas))
            
            else:
                chis[grid] = self._interp_funcs['chi']((pots, xnorms, thetas))
                Ss[grid] = self._interp_funcs['S']((pots, xnorms, thetas))
                Is[grid] = self._interp_funcs['I'][dirarg]((pots, xnorms, thetas))

        else:
            raise ValueError('Geometry not supported with rt_method cobain')

        rhos1 = self._interp_funcs['bbrho1'](pots[le_prim])*self.star.structure.default_units['rho']
        rhos2 = self._interp_funcs['bbrho2'](pots2[le_sec])*self.star.structure.default_units['rho']
        Ts1 = self._interp_funcs['bbT1'](pots[le_prim])*self.star.structure.default_units['T']
        Ts2 = self._interp_funcs['bbT2'](pots2[le_sec])*self.star.structure.default_units['T']

        if stepsize:
            
            chis[le_prim] = self.star.atmosphere._compute_absorption_coefficient(rhos1, Ts1)
            chis[le_sec] = self.star.atmosphere._compute_absorption_coefficient(rhos2, Ts2)

            return chis

        else:

            chis[le_prim] = self.star.atmosphere._compute_absorption_coefficient(rhos1, Ts1)
            chis[le_sec] = self.star.atmosphere._compute_absorption_coefficient(rhos2, Ts2)
            Ss[le_prim] = Is[le_prim] = self.star.atmosphere._compute_source_function(Ts1)
            Ss[le_sec] = Is[le_sec] = self.star.atmosphere._compute_source_function(Ts2)
        
            return chis, Ss, Is
            