import numpy as np 
import stargrit as sg
import quadpy.sphere as quadsph
import scipy.interpolate as spint
from stargrit import potentials

class RadiativeTransfer(object):

    def __init__(self, atmosphere, quadrature='Lebedev', ndir=15):

        self.__atmosphere = atmosphere

        if hasattr(quadsph, quadrature.title()):
            quadfunc = getattr(quadsph, quadrature.title())
            self._quadrature = quadfunc(ndir)
            self._quadrature.nI = len(self._quadrature.weights)
            # thetas and phis are in quadrature.azimuthal_polar ([:,0] is phi, [:,1] is theta) 

        else:
            raise ValueError('Quadrature %s not supported by quadpy' % quadrature.title())

    def compute_initial_Is(self, component=''):

        Ss = np.load(self.__atmosphere.__directory+'S%s_0.npy' % component)

        for i in range(self._quadrature.nI):
            np.save(self.__atmosphere.__directory+'I%s_%s_0.npy' % (component, i), Ss)

       
    @staticmethod
    def rot_theta(theta):

        if isinstance(theta, (list, tuple, np.ndarray)):
            theta[theta <= np.pi] = np.pi-theta[theta <= np.pi]
            theta[(theta > np.pi) & (theta <= 1.5*np.pi)] = theta[(theta > np.pi) & (theta <= 1.5*np.pi)] - np.pi
            theta[theta > 1.5*np.pi] = 2*np.pi - theta[theta > 1.5*np.pi]
            return theta

        else:
            if theta <= np.pi:
                return np.pi - theta
            elif theta <= 1.5 * np.pi:
                return theta - np.pi
            else:
                return 2 * np.pi - theta

    # ------------------------MAIN RADIATIVE TRANSFER FUNCTIONS ---------------------------- #

    @staticmethod
    def compute_transformation_matrix(normal):

        # Cartesian orthonormal unit vectors
        c1 = np.array([1., 0., 0.])
        c2 = np.array([0., 1., 0.])
        c3 = np.array([0., 0., 1.])

        # Roche normal orthonormal unit vectors

        tan_st_1 = np.array([normal[1], -normal[0], 0.])
        tan_st_2 = np.cross(normal, tan_st_1)

        n1 = tan_st_1 / np.sqrt(np.sum(tan_st_1 ** 2))
        n2 = tan_st_2 / np.sqrt(np.sum(tan_st_2 ** 2))
        n3 = normal / np.sqrt(np.sum(normal ** 2))

        # Transformation matrix direction cosines

        Q11 = np.dot(c1, n1)
        Q12 = np.dot(c1, n2)
        Q13 = np.dot(c1, n3)
        Q21 = np.dot(c2, n1)
        Q22 = np.dot(c2, n2)
        Q23 = np.dot(c2, n3)
        Q31 = np.dot(c3, n1)
        Q32 = np.dot(c3, n2)
        Q33 = np.dot(c3, n3)

        R = np.array([[Q11, Q12, Q13], [Q21, Q22, Q23], [Q31, Q32, Q33]])

        return R

    @staticmethod
    def rotate_direction_wrt_normal(Mc, coords, R):

        phi = coords[0]
        theta = coords[1]

        vorig = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

        vnew = np.dot(R, vorig)
        theta_new = np.arccos(vnew[2])
        phi_new = np.arctan2(vnew[1], vnew[0])
        ndir = np.array(
            [np.sin(theta_new) * np.cos(phi_new), np.sin(theta_new) * np.sin(phi_new), np.cos(theta_new)])
        
        return ndir


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
        [indx_in, indy_in, indz_in] = self.compute_grid_interpolation_indices(grid, points)

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
    def compute_grid_interpolation_indices(mesh, points):

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

    def adjust_stepsize(self, Mc, ndir, dirarg, **kwargs):

        #TODO: implement monochromatic vs mean

        exps = np.linspace(-10, -4, 1000)
        stepsizes = 10 ** exps

        b = np.ones((len(stepsizes), 3)).T * stepsizes
        rs = Mc - ndir * b.T

        # in case there are points in the centers of the stars (probably never for cylindrical)
        # make sure they get the maximum pot from LE value there

        chis = self.compute_structure(points=rs, dirarg=dirarg,  stepsize=True, **kwargs)
        # compute stepsize
        taus = chis * stepsizes / 2.

        diffs = np.abs(taus - 1.)
        stepsize_final = stepsizes[np.argmin(diffs)]

        # print 'stepsize: ', stepsize_final
        return stepsize_final / 2.

    def compute_solution(self, Mc, ndir, dirarg, stepsize, **kwargs):
        
        rs = np.array([Mc - i * stepsize * ndir for i in range(self._N)])
        paths = np.array([i * stepsize for i in range(self._N)])

        chis, Ss, Is = self.compute_structure(points=rs, dirarg=dirarg, stepsize=False, **kwargs)
                                        
        return self.compute_integral(paths,chis,Ss,Is)

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

    # ------------------- TOP-LEVEL MESH STUFF ---------------------------- #

    def compute_blackbody_interpolation(self, directory, component=''):
        """
        Computes the interpolation in potential for temperature and density from 
        stored blackbody structure files.
        """

        potTrho_bb = np.load(directory+'potTrho_bb%s.npy' % component)
        potT_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,1])
        potrho_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,2])

        return potT_bb, potrho_bb

    def compute_mesh_interpolation(self, directory, mesh, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points.

        If the opacity and atmosphere are mean/gray the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        RGI = spint.RegularGridInterpolator

        chi = np.load(directory + 'chi_' + str(iter_n - 1) + '%s.npy' % component)
        J = np.load(directory + 'J_' + str(iter_n - 1) + '%s.npy' % component)

        if self.__atmosphere._opactype == 'mean':
            # if the atmosphere is gray
            chi_interp = RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=chi)
        elif self.__atmosphere._opactype == 'monochromatic':
            chi_interp = chi
        else:
            raise NotImplementedError
        
        if self.__atmosphere._atm_type == 'gray':
            J_interp = RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=J)

            I_interp = []

            for i in range(self._quadrature.nI):
                I = np.load(directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '%s.npy' % component)
                I_interp.append(RGI(points=[mesh.coords['pots'], mesh.coords['thetas'], mesh.coords['phis']], values=I))
        
        elif self.__atmosphere._atm_type == 'monochromatic':
            J_interp = J
            I_interp = []
            for i in range(self._quadrature.nI):
                I_interp.append(np.load(directory + 'I_' + str(iter_n - 1) + '_' + str(int(i)) + '%s.npy' % component))

        else:
            raise NotImplementedError

        
        return chi_interp, J_interp, I_interp


    def compute_interpolation_functions(self, directory, mesh, iter_n=1):

        """
        Computes the blackbody and grid interpolation functions from a chosen iteration.
        """

        potT_bb, potrho_bb = self.compute_blackbody_interpolation(directory)
        chi_interp, J_interp, I_interp = self.compute_mesh_interpolation(directory, mesh, iter_n=iter_n)

        return {'bbT': potT_bb, 'bbrho': potrho_bb, 'chi': chi_interp, 'J':J_interp, 'I': I_interp}


    def compute_radiative_transfer(self, points, directory, mesh, iter_n=1, ray_discretization=5000):
        
        """
        Computes radiative transfer in a given set of mesh points.
        """

        self._N = ray_discretization
        self._interp_funcs = self.compute_interpolation_functions(iter_n=iter_n, directory=directory, mesh=mesh)
        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self._quadrature.nI))
        taus_new = np.zeros((len(points), self._quadrature.nI))

    # --------------- METHODS CALLED EXCLUSIVELY BY SUBCLASSES ----------------- #

    def compute_potentials(self, points, bbT, pot_range_grid, **kwargs):
        raise NotImplementedError
    
    def compute_interp_regions(self, pots, points, pot_range_grid, **kwargs):
        raise NotImplementedError

    def compute_structure(self, points, dirarg, stepsize=False, **kwargs):
        raise NotImplementedError


class DiffrotStarRadiativeTransfer(RadiativeTransfer):

    def __init__(self, atmosphereinstance, **kwargs):

        quadrature = kwargs.pop('quadrature', 'Lebedev')
        ndir = kwargs.pop('ndir', 15)
        super(DiffrotStarRadiativeTransfer,self).__init__(atmosphereinstance, quadrature=quadrature, ndir=ndir)
    
    def compute_potentials(self, points, bbT, pot_range_grid):
        
        pots = np.zeros(len(points))
        center = np.all(points == 0., axis=1)
        pots[center] = bbT[:,0].max()
        pots[~center] = potentials.diffrot.DiffRotRoche(points, self.__atmosphere.__mesh._bs)
        pots[(np.round(pots, 8) >= np.round(pot_range_grid[0], 8)) & (pots < pot_range_grid[0])] = pot_range_grid[0]
        
        return pots

    def compute_interp_regions(self, pots, points, pot_range_grid):
        
        grid = np.argwhere((pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
        le = np.argwhere(pots > pot_range_grid[1]).flatten()

        return grid, le

    def compute_coords_for_interpolation(self, rs):

        thetas = np.arccos(rs[:,2] / np.sqrt(np.sum(rs ** 2, axis=1)))
        phis = np.abs(np.arctan2(rs[:,1] / np.sqrt(np.sum(rs ** 2, axis=1)), rs[:,0] / np.sqrt(np.sum(rs ** 2, axis=1))))
        
        return thetas, phis

    def compute_structure(self, points, dirarg, interp_funcs, stepsize=False):

        pot_range_grid = [self.__atmosphere.__mesh.coords['pots'].min(), self.__atmosphere.__mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, interp_funcs['bbT'], pot_range_grid)
        grid, le = self.compute_interp_regions(pots, points, pot_range_grid)
        thetas, phis = self.compute_coords_for_interpolation(points)
        rhos, Ts = interp_funcs['bbrho'](pots[le]), interp_funcs['bbT'](pots[le])
        chis = np.zeros(len(pots))
        chis[grid] = interp_funcs['chi'](pots[grid], thetas[grid], phis[grid])
        chis[le] = self.__atmosphere.compute_chis(rhos, Ts, opactype=self.__atmosphere._opactype)
    
        if stepsize:
            return chis

        else:
            Ss = np.zeros(len(pots))
            Is = np.zeros(len(pots))

            Ss[grid] = interp_funcs['S'](pots[grid], thetas[grid], phis[grid])
            Is[grid] = interp_funcs['I'][dirarg](pots[grid], thetas[grid], phis[grid])
            Ss[le] = Is[le] = self.__atmosphere.compute_source_function(Ts)

            return chis, Ss, Is

    def adjust_stepsize(self, r, ndir, dirarg, interp_funcs):

        exps = np.linspace(-10, -4, 1000)
        stepsizes = 10 ** exps

        b = np.ones((len(stepsizes), 3)).T * stepsizes
        newpoints = r - ndir * b.T

        # in case there are points in the centers of the stars (probably never for cylindrical)
        # make sure they get the maximum pot from LE value there

        chis = self.compute_structure(newpoints, dirarg, interp_funcs, stepsize=True)
        # compute stepsize
        taus = chis * stepsizes / 2.

        diffs = np.abs(taus - 1.)
        stepsize_final = stepsizes[np.argmin(diffs)]

        # print 'stepsize: ', stepsize_final
        return stepsize_final / 2.


class ContactBinaryRadiativeTransfer(RadiativeTransfer):

    def __init__(self, atmosphere, **kwargs):

        quadrature = kwargs.pop('quadrature', 'Lebedev')
        ndir = kwargs.pop('ndir', 15)
        super(ContactBinaryRadiativeTransfer,self).__init__(atmosphere, quadrature=quadrature, ndir=ndir)

    def compute_initial_Is(self):

        if self.__atmosphere.__mesh._geometry == 'spherical':
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is(component='1')
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is(component='2')

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is()
        
        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)


    def compute_interpolation_functions(self, iter_n=1):
        
        potT_bb1, potrho_bb1 = self.compute_blackbody_interpolation(self.__atmosphere.__directory, component='1')
        potT_bb2, potrho_bb2 = self.compute_blackbody_interpolation(self.__atmosphere.__directory, component='2')

        if self.__atmosphere.__mesh._geometry == 'spherical':

            chi1_interp, J1_interp, I1_interp = self.compute_mesh_interpolation(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='1', iter_n=iter_n)
            chi2_interp, J2_interp, I2_interp = self.compute_mesh_interpolation(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='2', iter_n=iter_n)

            return {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                    'chi1': chi1_interp, 'chi2': chi2_interp, 'J1': J1_interp, 'J2': J2_interp,
                    'I1': I1_interp, 'I2': I2_interp}

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':

            chi_interp, J_interp, I_interp = self.compute_mesh_interpolation(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='', iter_n=iter_n)
            
            return {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                      'chi': chi_interp, 'J':J_interp, 'I': I_interp}

        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)
    

    def compute_potentials(self, points, q, bbT1, bbT2, pot_range_grid):

        pots = np.zeros(len(points))
        center1 = np.all(points == 0., axis=1)
        center2 = (points[:, 0] == 1.) & (points[:, 1] == 0.) & (points[:, 2] == 0.)
        pots[~center1 & ~center2] = potentials.roche.BinaryRoche_cartesian(points[~center1 & ~center2], q)
        pots[center1] = bbT1[:,0].max()
        pots[center2] = bbT2[:,0].max()
        pots[(np.round(pots, 8) >= np.round(pot_range_grid[0], 8)) & (pots < pot_range_grid[0])] = pot_range_grid[0]
        
        return pots


    @staticmethod
    def normalize_xs(xs, pots, q):

        pots2 = pots / q + 0.5 * (q - 1) / q

        xrb1s = - sg.geometry.cylindrical.ContactBinaryMesh.get_rbacks(pots, q, 1)
        xrb2s = (np.ones(len(pots)) + sg.geometry.cylindrical.ContactBinaryMesh.get_rbacks(pots, q, 2))

        return sg.geometry.cylindrical.ContactBinaryMesh.xphys_to_xnorm_cb(xs, xrb1s, xrb2s)


    def compute_interp_regions(self, pots, points, pot_range_grid, geometry='cylindrical'):

        """
        Computes the different regions along a ray for interpolation: grid/blackbody, primary/secondary.
        """
        
        prim = (points[:,0] <= self.__atmosphere.__mesh._nekmin)
        sec = (points[:,0] > self.__atmosphere.__mesh._nekmin)
        le_prim = np.argwhere(prim & (pots > pot_range_grid[1])).flatten()
        le_sec = np.argwhere(sec & (pots > pot_range_grid[1])).flatten()

        if geometry == 'cylindrical':
            
            # in cylindrical geometry, there is no separation between primary and secondary in the grid
            grid = np.argwhere((pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            
            return grid, le_prim, le_sec

        elif geometry == 'spherical':

            # in spherical geometry, the two halves are built and interpolated separately
            grid_prim = np.argwhere(prim & (pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            grid_sec = np.argwhere(sec & (pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            
            return grid_prim, grid_sec, le_prim, le_sec
        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % geometry)
    

    def compute_coords_for_interpolation(self, points, geometry='cylindrical', **kwargs):

        """
        Based on geometry, returns the grid coordinates of the ray points for interpolation.
        """

        if geometry == 'cylindrical':

            grid = kwargs.pop('grid')
            pots = kwargs.pop('pots')

            xnorms_grid = self.normalize_xs(points[:, 0][grid], pots[grid], self.__atmosphere.__mesh._q)
            thetas_grid = np.abs(np.arcsin(points[:,1][grid]/np.sqrt(np.sum((points[:,1][grid]**2,points[:,2][grid]**2),axis=0))))
            thetas_grid[np.isnan(thetas_grid)] = 0.0

            return xnorms_grid, thetas_grid

        elif geometry == 'spherical':
            
            grid_prim = kwargs.pop('grid_prim')
            grid_sec = kwargs.pop('grid_sec')

            rs = points
            rs_sec = rs[grid_sec].copy()
            rs_sec[:,0] = 1.0 - rs_sec[:,0]
            thetas = np.zeros(len(rs)) 
            phis = np.zeros(len(rs))    

            thetas[grid_prim] = np.arccos(rs[grid_prim][:,2] / np.sqrt(np.sum(rs[grid_prim] ** 2, axis=1)))
            phis[grid_prim] = np.abs(np.arctan2(rs[grid_prim][:,1] / np.sqrt(np.sum(rs[grid_prim] ** 2, axis=1)), rs[grid_prim][:,0] / np.sqrt(np.sum(rs[grid_prim] ** 2, axis=1))))
            thetas[grid_sec] = np.arccos(rs_sec[:, 2] / np.sqrt(np.sum(rs_sec ** 2, axis=1)))
            phis[grid_sec] = np.abs(np.arctan2(rs_sec[:, 1] / np.sqrt(np.sum(rs_sec ** 2, axis=1)), rs_sec[:, 0] / np.sqrt(np.sum(rs_sec ** 2, axis=1))))

            thetas[thetas > np.pi / 2] = self.rot_theta(thetas[thetas > np.pi / 2])

            return thetas, phis

        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % geometry)


    def compute_structure(self, points, dirarg, interp_funcs, stepsize=False):
        """
        Returns the radiative structure in all points along a ray.
        """
        pot_range_grid = [self.__atmosphere.__mesh.coords['pots'].min(), self.__atmosphere.__mesh.coords['pots'].max()]
        pots = self.compute_potentials(points, self.__atmosphere.__mesh._q, interp_funcs['bbT1'], interp_funcs['bbT2'], pot_range_grid)
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
                chis[grid_prim] = interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

            else:
                chis[grid_prim] = interp_funcs['chi1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                chis[grid_sec] = interp_funcs['chi2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Ss[grid_prim] = interp_funcs['S1'](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Ss[grid_sec] = interp_funcs['S2'](pots[grid_sec], thetas[grid_sec], phis[grid_sec])
                Is[grid_prim] = interp_funcs['I1'][dirarg](pots[grid_prim], thetas[grid_prim], phis[grid_prim])
                Is[grid_sec] = interp_funcs['I2'][dirarg](pots[grid_sec], thetas[grid_sec], phis[grid_sec])

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':
            grid, le_prim, le_sec = self.compute_interp_regions(pots=pots,points=points,pot_range_grid=pot_range_grid,geometry='cylindrical')
            # here xnorms and thetas are only those pertaining to grid points
            xnorms, thetas = self.compute_coords_for_interpolation(points, geometry='cylindrical', grid=grid, pots=pots)
            
            if stepsize:
                chis[grid] = interp_funcs['chi'](pots[grid], xnorms, thetas)
            
            else:
                chis[grid] = interp_funcs['chi'](pots, xnorms, thetas)
                Ss[grid] = interp_funcs['S'](pots, xnorms, thetas)
                Is[grid] = interp_funcs['I'][dirarg](pots, xnorms, thetas)

        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)

        rhos1, rhos2 = interp_funcs['bbrho1'](pots[le_prim]), interp_funcs['bbrho2'](pots2[le_sec])
        Ts1, Ts2 = interp_funcs['bbT1'](pots[le_prim]), interp_funcs['bbT2'](pots2[le_sec])

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
            












