import numpy as np 
import stargrit as sg
import quadpy.sphere as quadsph
import scipy.interpolate as spint
from stargrit import potentials
from stargrit.radiative_transfer.cobain.general import RadiativeTransfer, DiffrotStarRadiativeTransfer, ContactBinaryRadiativeTransfer

class GrayRadiativeTransfer(RadiativeTransfer):


    def mesh_interpolation_functions(self, directory, mesh, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points.

        If the opacity and atmosphere are mean/gray the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        RGI = spint.RegularGridInterpolator

        chi = np.load(directory + 'chi_' + str(iter_n - 1) + '%s.npy' % component)
        J = np.load(directory + 'J_' + str(iter_n - 1) + '%s.npy' % component)

        chi_interp = RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=chi)
        J_interp = RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=J)

        I_interp = []

        for i in range(self._quadrature.nI):
            I = np.load(directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '%s.npy' % component)
            I_interp.append(RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=I))
    
        return chi_interp, J_interp, I_interp


    def adjust_stepsize(self, Mc, ndir, dirarg):

        #TODO: implement monochromatic vs mean

        exps = np.linspace(-10, -4, 1000)
        stepsizes = 10 ** exps

        b = np.ones((len(stepsizes), 3)).T * stepsizes
        rs = Mc - ndir * b.T

        chis = self.compute_structure(points=rs, dirarg=dirarg,  stepsize=True)
        # compute stepsize
        taus = chis * stepsizes / 2.

        diffs = np.abs(taus - 1.)
        stepsize_final = stepsizes[np.argmin(diffs)]

        # print 'stepsize: ', stepsize_final
        return stepsize_final / 2.


    def intensity_step(self, Mc, ndir, dirarg, stepsize, N):
        
        rs = np.array([Mc - i * stepsize * ndir for i in range(N)])
        paths = np.array([i * stepsize for i in range(N)])

        chis, Ss, Is = self.compute_structure(points=rs, dirarg=dirarg, stepsize=False)
                                        
        return self.intensity_integral(paths,chis,Ss,Is,N)


    def intensity_integral(self, paths, chis, Ss, Is, N, spline_order=1):

        #TODO: implement gray vs monochromatic, handle limits in lambda? - in radiative equlibrium
        
        chis_sp = spint.UnivariateSpline(paths, chis, k=spline_order, s=0)
        taus = np.array([chis_sp.integral(paths[0], paths[i]) for i in range(self._N)])
        Ss_exp = Ss * np.exp((-1.)*taus)

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
            else:
                I = 0.0

            if np.isnan(I) or I < 0.:
                I = 0.0

        else:
            taus = taus[:nbreak + 1]
            I = 0.0

        return nbreak, taus[-1], Is[nbreak], I


    def compute_intensity(self, Mc, n):

        R = self.transformation_matrix(n)

        Is_j = np.zeros(self._quadrature.nI)
        taus_j = np.zeros(self._quadrature.nI)

        for dirarg in range(self._quadrature.nI):

            coords = self._quadrature.azimuthal_polar[dirarg]
            ndir = self.rotate_direction_wrt_normal(Mc=Mc, coords=coords, R=R)
            stepsize = self.adjust_stepsize(Mc, ndir, dirarg)
            
            nbreak, tau, I0, I = self.intensity_step(Mc, ndir, dirarg, stepsize, self._N)
            
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
                
                nbreak, tau, I0, I = self.intensity_step(Mc, ndir, dirarg, stepsize, N)

            taus_j[dirarg], Is_j[dirarg] = tau, I
            
        return taus_j, Is_j


    def compute_structure(self, points, dirarg, stepsize=False):
        raise NotImplementedError


class DiffrotStarGrayRadiativeTransfer(GrayRadiativeTransfer, DiffrotStarRadiativeTransfer):


    def compute_structure(self, points, dirarg, stepsize=False):

        pot_range_grid = [self.__atmosphere.__mesh.coords['pots'].min(), self.__atmosphere.__mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, self._interp_funcs['bbT'], pot_range_grid)
        grid, le = self.compute_interp_regions(pots, points, pot_range_grid)
        thetas, phis = self.compute_coords_for_interpolation(points)
        rhos, Ts = self._interp_funcs['bbrho'](pots[le]), self._interp_funcs['bbT'](pots[le])
        chis = np.zeros(len(pots))
        chis[grid] = self._interp_funcs['chi'](pots[grid], thetas[grid], phis[grid])
        chis[le] = self.__atmosphere.compute_chis(rhos, Ts, opactype=self.__atmosphere._opactype)
    
        if stepsize:
            return chis

        else:
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

            Ss[grid] = self._interp_funcs['S'](pots[grid], thetas[grid], phis[grid])
            Is[grid] = self._interp_funcs['I'][dirarg](pots[grid], thetas[grid], phis[grid])
            Ss[le] = Is[le] = self.__atmosphere.compute_source_function(Ts)

            return chis, Ss, Is


class ContactBinaryGrayRadiativeTransfer(GrayRadiativeTransfer, ContactBinaryRadiativeTransfer):


    def compute_structure(self, points, dirarg, stepsize=False):
        """
        Returns the radiative structure in all points along a ray.
        """
        pot_range_grid = [self.__atmosphere.__mesh.coords['pots'].min(), self.__atmosphere.__mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, self.__atmosphere.__mesh._q, self._interp_funcs['bbT1'], self._interp_funcs['bbT2'], pot_range_grid)
        q = self.__atmosphere.__mesh._q
        pots2 = pots / q + 0.5 * (q - 1) / q

        if stepsize:
            chis = np.zeros(len(pots))
        else:
            chis = np.zeros(len(pots))
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

        if self.__atmosphere.__mesh._geometry == 'spherical':
            
            grid_prim, grid_sec, le_prim, le_sec = self.compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid, geometry='spherical')
            # thetas and phis are returned for all points (zeros out of grid)... maybe fix this?
            thetas, phis = self.compute_coords_for_interpolation(points, geometry='spherical', grid_prim=grid_prim, grid_sec=grid_sec)

            if stepsize:
                chis[grid_prim] = self._interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = self._interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

            else:
                chis[grid_prim] = self._interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = self._interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Ss[grid_prim] = self._interp_funcs['S1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Ss[grid_sec] = self._interp_funcs['S2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Is[grid_prim] = self._interp_funcs['I1'][dirarg](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Is[grid_sec] = self._interp_funcs['I2'][dirarg](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':

            grid, le_prim, le_sec = self.compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid,geometry='cylindrical')
            # here xnorms and thetas are only those pertaining to grid points
            xnorms, thetas = self.compute_coords_for_interpolation(points, geometry='cylindrical', grid=grid, pots=pots)
            
            if stepsize:
                chis[grid] = self._interp_funcs['chi'](pots[grid], xnorms, thetas)
            
            else:
                chis[grid] = self._interp_funcs['chi'](pots, xnorms, thetas)
                Ss[grid] = self._interp_funcs['S'](pots, xnorms, thetas)
                Is[grid] = self._interp_funcs['I'][dirarg](pots, xnorms, thetas)

        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)

        rhos1, rhos2 = self._interp_funcs['bbrho1'](pots[le_prim]), self._interp_funcs['bbrho2'](pots2[le_sec])
        Ts1, Ts2 = self._interp_funcs['bbT1'](pots[le_prim]), self._interp_funcs['bbT2'](pots2[le_sec])

        if stepsize:
            
            chis[le_prim] = self.__atmosphere.compute_chis(rhos1, Ts1, opactype=self.__atmosphere._opactype)
            chis[le_sec] = self.__atmosphere.compute_chis(rhos2, Ts2, opactype=self.__atmosphere._opactype)

            return chis

        else:

            chis[le_prim] = self.__atmosphere.compute_chis(rhos1, Ts1, opactype=self.__atmosphere._opactype)
            chis[le_sec] = self.__atmosphere.compute_chis(rhos2, Ts2, opactype=self.__atmosphere._opactype)
            Ss[le_prim] = Is[le_prim] = self.__atmosphere.compute_source_function(Ts1)
            Ss[le_sec] = Is[le_sec] = self.__atmosphere.compute_source_function(Ts2)
        
            return chis, Ss, Is