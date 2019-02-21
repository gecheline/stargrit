from stargrit.radiative_transfer.cobain.general import RadiativeTransfer, DiffrotStarRadiativeTransfer, ContactBinaryRadiativeTransfer
import numpy as np
import scipy.interpolate as spint
from stargrit.geometry.spherical import SphericalMesh
from stargrit.geometry.cylindrical import CylindricalMesh


class MonochromaticRadiativeTransfer(RadiativeTransfer):

    
    def _initialize_I_tau_arrays(self):

        meshsize = self.star.mesh.dims[0]*self.star.mesh.dims[1]*self.star.mesh.dims[2]
        arr = np.zeros((meshsize, len(self.star.atmosphere.wavelenghts), self.quadrature.nI))

        return arr, arr


    def _save_array(self, arr, arrname, iter_n):

        # based on object type, geometry and atmosphere: for spherical contact splitting required
        # arr in shape (mesh, nI), needs to be (nI, npot, ntheta, nphi)

        np.save(self.star.directory + '%s_%s.npy' % (arrname,iter_n), 
                arr.T.reshape((self.quadrature.nI,)+tuple(self.star.mesh.dims)+tuple(len(self.star.atmosphere.wavelenghts))))


    def _load_array(self, arrname, iter_n):
        # based on object type, geometry and atmosphere: for spherical contact merging required
        meshsize = self.star.mesh.dims[0]*self.star.mesh.dims[1]*self.star.mesh.dims[2]
        return np.load(self.star.directory + '%s_%s.npy' ).reshape((self.quadrature.nI,)+(meshsize,)).T


    def _mesh_interpolation_functions(self, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points.

        If the opacity and atmosphere are mean/gray the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        directory = self.star.directory
        mesh = self.star.mesh 

        RGI = spint.RegularGridInterpolator

        chi = np.load(directory + 'chi%s_%i.npy' % (component, iter_n-1))
        J = np.load(directory + 'S%s_%i.npy' % (component, iter_n-1))

        key1, key2, key3 = self.star.mesh.coords.keys()

        if self.star.atmosphere.opactype == 'mean':
            # if the atmosphere is gray
            chi_interp = RGI(points=[self.star.mesh.coords[key1], self.star.mesh.coords[key2], 
                            self.star.mesh.coords[key3]], values=chi)
        elif self.star.atmosphere.opactype == 'monochromatic':
            chi_interp = chi
        else:
            raise NotImplementedError
        
        J_interp = J
        I_interp = []
        for i in range(self.quadrature.nI):
            I_interp.append(np.load(directory + 'I%s_%i_%i.npy' % (component,iter_n-1,i)))

        return chi_interp, J_interp, I_interp


    def _adjust_stepsize(self, Mc, ndir, dirarg):

        #TODO: implement monochromatic vs mean

        exps = np.linspace(-10, -4, 1000)
        stepsizes = 10 ** exps

        b = np.ones((len(stepsizes), 3)).T * stepsizes
        rs = Mc - ndir * b.T

        chis = self._compute_structure(points=rs, dirarg=dirarg, stepsize=True)
        # compute stepsize
        taus = chis * stepsizes / 2.

        diffs = np.abs(taus - 1.)
        stepsize_final = stepsizes[np.argmin(diffs)]

        # 'stepsize: ', stepsize_final
        return stepsize_final / 2.


    def _trilinear_interp_monochromatic(self, points, grid, f):

        # the integer values of the input arrays are the (0,0,0) points of the box
        # once the values and distance factors are computed in each box for the chosen point
        # trilinear interpolation (https://en.wikipedia.org/w/index.php?title=Trilinear_interpolation) 
        # is performed on all wavelength datapoints
        # points = x,y,z are the coordinates of the points along the ray
        # grid = grid_x, grid_y, grid_z are coordinates of the corresponding grid variables
        # inds = indx_in, indy_in, indz_in are the indices used for interpolation of points in grid

        [x, y, z] = points
        [grid_x, grid_y, grid_z] = grid
        [indx_in, indy_in, indz_in] = self._grid_interpolation_indices(grid, points)

        indx_in = indx_in.astype(int)
        indy_in = indy_in.astype(int)
        indz_in = indz_in.astype(int)

        x0s, x1s = grid_x[indx_in], grid_x[indx_in+1]
        y0s, y1s = grid_y[indy_in], grid_y[indy_in+1]
        z0s, z1s = grid_z[indz_in], grid_z[indz_in+1]

        c000s = f[indx_in, indy_in, indz_in]
        c100s = f[indx_in+1, indy_in, indz_in]
        c010s = f[indx_in, indy_in+1, indz_in]
        c110s = f[indx_in+1, indy_in+1, indz_in]
        c001s = f[indx_in, indy_in, indz_in+1]
        c101s = f[indx_in+1, indy_in, indz_in+1]
        c011s = f[indx_in, indy_in+1, indz_in+1]
        c111s = f[indx_in+1, indy_in+1, indz_in+1]

        xds = (x-x0s)/(x1s-x0s)
        yds = (y-y0s)/(y1s-y0s)
        zds = (z-z0s)/(z1s-z0s)

        c00s = c000s.T*(1.-xds) + c100s.T*xds
        c01s = c001s.T*(1.-xds) + c101s.T*xds
        c10s = c010s.T*(1.-xds) + c110s.T*xds
        c11s = c011s.T*(1.-xds) + c111s.T*xds

        c0s = c00s*(1.-yds) + c10s*yds
        c1s = c01s*(1.-yds) + c11s*yds

        cs = (c0s*(1.-zds) + c1s*zds).T 

        return cs


    @staticmethod
    def _grid_interpolation_indices(mesh, points):

        xind = np.arange(0, len(mesh['x']),1)
        yind = np.arange(0, len(mesh['y']),1)
        zind = np.arange(0, len(mesh['z']),1)

        xi, yi, zi = np.meshgrid(xind, yind, zind, indexing='ij')

        RGI = spint.RegularGridInterpolator

        xindinterp = RGI(points=(mesh['x'],mesh['y'],mesh['z']),values=xi)
        yindinterp = RGI(points=(mesh['x'],mesh['y'],mesh['z']),values=yi)
        zindinterp = RGI(points=(mesh['x'],mesh['y'],mesh['z']),values=zi)

        xinds = xindinterp((points['x'], points['y'], points['z'])).astype(int)
        yinds = yindinterp((points['x'], points['y'], points['z'])).astype(int)
        zinds = zindinterp((points['x'], points['y'], points['z'])).astype(int)

        return [xinds, yinds, zinds]


    def _intensity_step(self, Mc, ndir, dirarg, stepsize, **kwargs):
        
        rs = np.array([Mc - i * stepsize * ndir for i in range(self._N)])
        paths = np.array([i * stepsize for i in range(self._N)])

        chis, Ss, Is = self._compute_structure(points=rs, dirarg=dirarg, stepsize=False, **kwargs)
                                        
        return self._intensity_integral(paths,chis,Ss,Is, **kwargs)


    def _intensity_integral(self, paths, chis, Ss, Is, spline_order=1):

        #TODO: implement gray vs monochromatic, handle limits in lambda? - in radiative equlibrium
        
        chis_sp = spint.UnivariateSpline(paths, chis, k=spline_order, s=0)
        taus = np.array([chis_sp.integral(paths[0], paths[i]) for i in range(self._N)])
        Ss_exp = Ss * np.exp((-1.)*taus)

        Iszero = np.argwhere(Is == 0.0).flatten()
        Ssezero = np.argwhere(Ss_exp == 0.0).flatten()

        if Iszero.size == 0 and Ssezero.size == 0:
            nbreak = self._N - 1
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

        return nbreak, taus, I


    def _compute_intensity(self, Mc, n, dirarg, coords, **kwargs):
        
        R = self._transformation_matrix(n)
        ndir = self._rotate_direction_wrt_normal(Mc=Mc, coords=coords, R=R)
        stepsize = self._adjust_stepsize(Mc, ndir, dirarg)
        
        nbreak, taus, Is = self._intensity_integral(Mc, ndir, dirarg, stepsize, **kwargs)
        
        Iinit = Is[nbreak]
        subd = 'no'
        if (nbreak < 1000 and Iinit != 0.0) or nbreak == self._N-1:
            if (nbreak < 1000 and Iinit != 0.0):
                subd = 'yes'
                div = 1000. / nbreak
                stepsize = stepsize / div
            elif nbreak == self._N - 1:
                subd = 'yes'
                self._N = self._N * 2
            else:
                subd = 'yes'
                div = 1000. / nbreak
                stepsize = stepsize / div
                self._N = self._N * 2
            
            nbreak, taus, I = self._intensity_integral(Mc, dirarg, stepsize, **kwargs)

        return I, taus


    def _compute_structure(self, points, dirarg, stepsize=False):
        raise NotImplementedError


class DiffrotStarMonochromaticRadiativeTransfer(MonochromaticRadiativeTransfer, DiffrotStarRadiativeTransfer):

    def compute_structure(self, points, dirarg, stepsize=False):

        pot_range_grid = [self.star.mesh.coords['pots'].min(), self.star.mesh.coords['pots'].max()]
        pots = self._compute_potentials(points, self._interp_funcs['bbT'], pot_range_grid)
        grid, le = self._compute_interp_regions(pots, points, pot_range_grid)
        thetas, phis = self._compute_coords_for_interpolation(points)
        rhos, Ts = self._interp_funcs['bbrho'](pots[le]), self._interp_funcs['bbT'](pots[le])
        chis = np.zeros(len(pots))
        chis[grid] = self._interp_funcs['chi'](pots[grid], thetas[grid], phis[grid])
        chis[le] = self.star.atmosphere._compute_absorption_coefficient(rhos, Ts)
    
        if stepsize:
            return chis

        else:
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

            Ss[grid] = self._interp_funcs['S'](pots[grid], thetas[grid], phis[grid])
            Is[grid] = self._interp_funcs['I'][dirarg](pots[grid], thetas[grid], phis[grid])
            Ss[le] = Is[le] = self.star.atmosphere._compute_source_function(Ts)

            return chis, Ss, Is


class ContactBinaryMonochromaticRadiativeTransfer(MonochromaticRadiativeTransfer, ContactBinaryRadiativeTransfer):

    def compute_structure(self, points, dirarg, stepsize=False):
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
                chis[grid_prim] = self._interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = self._interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

            else:
                chis[grid_prim] = self._interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = self._interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Ss[grid_prim] = self._interp_funcs['S1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Ss[grid_sec] = self._interp_funcs['S2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Is[grid_prim] = self._interp_funcs['I1'][dirarg](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Is[grid_sec] = self._interp_funcs['I2'][dirarg](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

        elif isinstance(self.star.mesh, CylindricalMesh):
            grid, le_prim, le_sec = self._compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid)
            # here xnorms and thetas are only those pertaining to grid points
            xnorms, thetas = self._compute_coords_for_interpolation(points, grid=grid, pots=pots)
            
            if stepsize:
                chis[grid] = self._interp_funcs['chi'](pots[grid], xnorms, thetas)
            
            else:
                chis[grid] = self._interp_funcs['chi'](pots, xnorms, thetas)
                Ss[grid] = self._interp_funcs['S'](pots, xnorms, thetas)
                Is[grid] = self._interp_funcs['I'][dirarg](pots, xnorms, thetas)

        else:
            raise ValueError('Geometry not supported with rt_method cobain')

        rhos1, rhos2 = self._interp_funcs['bbrho1'](pots[le_prim]), self._interp_funcs['bbrho2'](pots2[le_sec])
        Ts1, Ts2 = self._interp_funcs['bbT1'](pots[le_prim]), self._interp_funcs['bbT2'](pots2[le_sec])

        chis[le_prim] = self.star.atmosphere._compute_absorption_coefficient(rhos1, Ts1)
        chis[le_sec] = self.star.atmosphere._compute_absorption_coefficient(rhos2, Ts2)
        
        if stepsize:
            return chis

        else:

            Ss[le_prim] = Is[le_prim] = self.star.atmosphere._compute_source_function(Ts1)
            Ss[le_sec] = Is[le_sec] = self.star.atmosphere._compute_source_function(Ts2)
        
            return chis, Ss, Is