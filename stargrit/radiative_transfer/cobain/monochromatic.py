from stargrit.radiative_transfer.cobain.general import RadiativeTransfer, DiffrotStarRadiativeTransfer, ContactBinaryRadiativeTransfer

class MonochromaticRadiativeTransfer(RadiativeTransfer):

    def mesh_interpolation_functions(self, directory, mesh, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points.

        If the opacity and atmosphere are mean/gray the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        RGI = spint.RegularGridInterpolator

        chi = np.load(directory + 'chi_' + str(iter_n - 1) + '%s.npy' % component)
        J = np.load(directory + 'J_' + str(iter_n - 1) + '%s.npy' % component)

        if self._atmosphere._opactype == 'mean':
            # if the atmosphere is gray
            chi_interp = RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=chi)
        elif self._atmosphere._opactype == 'monochromatic':
            chi_interp = chi
        else:
            raise NotImplementedError
        
        J_interp = J
        I_interp = []
        for i in range(self._quadrature.nI):
            I_interp.append(np.load(directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '%s.npy' % component))

        return chi_interp, J_interp, I_interp

    def trilinear_interp_monochromatic(self, points, grid, f):

        # the integer values of the input arrays are the (0,0,0) points of the box
        # once the values and distance factors are computed in each box for the chosen point
        # trilinear interpolation (https://en.wikipedia.org/w/index.php?title=Trilinear_interpolation) 
        # is performed on all wavelength datapoints
        # points = x,y,z are the coordinates of the points along the ray
        # grid = grid_x, grid_y, grid_z are coordinates of the corresponding grid variables
        # inds = indx_in, indy_in, indz_in are the indices used for interpolation of points in grid

        [x, y, z] = points
        [grid_x, grid_y, grid_z] = grid
        [indx_in, indy_in, indz_in] = self.grid_interpolation_indices(grid, points)

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
    def grid_interpolation_indices(mesh, points):

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

    def compute_solution(self, Mc, ndir, dirarg, stepsize, **kwargs):
        
        rs = np.array([Mc - i * stepsize * ndir for i in range(self._N)])
        paths = np.array([i * stepsize for i in range(self._N)])

        chis, Ss, Is = self.compute_structure(points=rs, dirarg=dirarg, stepsize=False, **kwargs)
                                        
        return self.compute_integral(paths,chis,Ss,Is)


    def intensity_step(self, Mc, ndir, dirarg, stepsize, **kwargs):
        
        rs = np.array([Mc - i * stepsize * ndir for i in range(self._N)])
        paths = np.array([i * stepsize for i in range(self._N)])

        chis, Ss, Is = self.compute_structure(points=rs, dirarg=dirarg, stepsize=False, **kwargs)
                                        
        return self.intensity_integral(paths,chis,Ss,Is)


    def compute_integral(self, paths, chis, Ss, Is, spline_order=1):

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


    def compute_intensity(self, Mc, n, dirarg, coords, **kwargs):

        R = self.compute_transformation_matrix(n)
        ndir = self.rotate_direction_wrt_normal(Mc=Mc, coords=coords, R=R)
        stepsize = self.adjust_stepsize(Mc, ndir, dirarg, **kwargs)
        
        nbreak, taus, Is = self.compute_solution(Mc, ndir, dirarg, stepsize, **kwargs)
        
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
            
            nbreak, taus, I = self.compute_solution(Mc, dirarg, stepsize, **kwargs)

        return taus, I


class DiffrotStarMonochromaticRadiativeTransfer(MonochromaticRadiativeTransfer, DiffrotStarRadiativeTransfer):

    def compute_structure(self, points, dirarg, stepsize=False):

        pot_range_grid = [self._atmosphere.mesh.coords['pots'].min(), self._atmosphere.mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, self._interp_funcs['bbT'], pot_range_grid)
        grid, le = self.compute_interp_regions(pots, points, pot_range_grid)
        thetas, phis = self.compute_coords_for_interpolation(points)
        rhos, Ts = self._interp_funcs['bbrho'](pots[le]), self._interp_funcs['bbT'](pots[le])
        chis = np.zeros(len(pots))
        chis[grid] = self._interp_funcs['chi'](pots[grid], thetas[grid], phis[grid])
        chis[le] = self._atmosphere.compute_chis(rhos, Ts, opactype=self._atmosphere._opactype)
    
        if stepsize:
            return chis

        else:
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

            Ss[grid] = self._interp_funcs['S'](pots[grid], thetas[grid], phis[grid])
            Is[grid] = self._interp_funcs['I'][dirarg](pots[grid], thetas[grid], phis[grid])
            Ss[le] = Is[le] = self._atmosphere.compute_source_function(Ts)

            return chis, Ss, Is


class ContactBinaryMonochromaticRadiativeTransfer(MonochromaticRadiativeTransfer, ContactBinaryRadiativeTransfer):

    def compute_structure(self, points, dirarg, stepsize=False):
        """
        Returns the radiative structure in all points along a ray.
        """
        pot_range_grid = [self._atmosphere.mesh.coords['pots'].min(), self._atmosphere.mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, self._atmosphere.mesh._q, self._interp_funcs['bbT1'], self._interp_funcs['bbT2'], pot_range_grid)
        q = self._atmosphere.mesh._q
        pots2 = pots / q + 0.5 * (q - 1) / q

        if stepsize:
            chis = np.zeros(len(pots))
        else:
            chis = np.zeros(len(pots))
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

        if self._atmosphere.mesh._geometry == 'spherical':
            
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

        elif self._atmosphere.mesh._geometry == 'cylindrical':
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
            raise ValueError('Geometry %s not supported with rt_method cobain' % self._atmosphere.mesh._geometry)

        rhos1, rhos2 = self._interp_funcs['bbrho1'](pots[le_prim]), self._interp_funcs['bbrho2'](pots2[le_sec])
        Ts1, Ts2 = self._interp_funcs['bbT1'](pots[le_prim]), self._interp_funcs['bbT2'](pots2[le_sec])

        if stepsize:
            
            chis[le_prim] = self._atmosphere.compute_chis(rhos1, Ts1, opactype=self._atmosphere._opactype)
            chis[le_sec] = self._atmosphere.compute_chis(rhos2, Ts2, opactype=self._atmosphere._opactype)

            return chis

        else:

            chis[le_prim] = self._atmosphere.compute_chis(rhos1, Ts1, opactype=self._atmosphere._opactype)
            chis[le_sec] = self._atmosphere.compute_chis(rhos2, Ts2, opactype=self._atmosphere._opactype)
            Ss[le_prim] = Is[le_prim] = self._atmosphere.compute_source_function(Ts1)
            Ss[le_sec] = Is[le_sec] = self._atmosphere.compute_source_function(Ts2)
        
            return chis, Ss, Is