import numpy as np 
import stargrit as sg
import quadpy.sphere as quadsph
import scipy.interpolate as spint
from stargrit import potentials
import logging

class RadiativeTransfer(object):

    def __init__(self, atmosphere, quadrature='Lebedev', ndir=15):

        self.__atmosphere = atmosphere

        if hasattr(quadsph, quadrature.title()):
            quadfunc = getattr(quadsph, quadrature.title())
            self._quadrature = quadfunc(str(ndir))
            self._quadrature.nI = len(self._quadrature.weights)
            # thetas and phis are in quadrature.azimuthal_polar ([:,0] is phi, [:,1] is theta) 

        else:
            raise ValueError('Quadrature %s not supported by quadpy' % quadrature.title())

       
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


    @staticmethod
    def transformation_matrix(normal):

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


    def blackbody_interpolation_functions(self, directory, component=''):
        """
        Computes the interpolation in potential for temperature and density from 
        stored blackbody structure files.
        """

        potTrho_bb = np.load(directory+'potTrho_bb%s.npy' % component)
        potT_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,1])
        potrho_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,2])

        return potT_bb, potrho_bb


    def mesh_interpolation_functions(self, directory, mesh, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points, object and atmosphere specific.

        If the opacity and atmosphere are mean/gray the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        raise NotImplementedError


    def compute_interpolation_functions(self, directory, mesh, iter_n=1):

        """
        Computes the blackbody and grid interpolation functions from a chosen iteration.
        """

        potT_bb, potrho_bb = self.blackbody_interpolation_functions(directory)
        chi_interp, J_interp, I_interp = self.mesh_interpolation_functions(directory, mesh, iter_n=iter_n)

        self._interp_funcs = {'bbT': potT_bb, 'bbrho': potrho_bb, 'chi': chi_interp, 'J':J_interp, 'I': I_interp}


    def compute_initial_Is(self, component=''):

        Ss = np.load(self.__atmosphere.__directory+'S%s_0.npy' % component)

        for i in range(self._quadrature.nI):
            np.save(self.__atmosphere.__directory+'I%s_%s_0.npy' % (component, i), Ss)


    def compute_intensity(self, Mc, n):
        # this technically could be implemented here as well, with checking for gray/mono in array creation
        return NotImplementedError


    def compute_radiative_transfer(self, points, directory, mesh, iter_n=1, ray_discretization=5000):
        
        """
        Computes radiative transfer in a given set of mesh points.
        """

        self._N = ray_discretization
        self._interp_funcs = self.compute_interpolation_functions(iter_n=iter_n, directory=directory, mesh=mesh)
        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self._quadrature.nI))
        taus_new = np.zeros((len(points), self._quadrature.nI))

        for j, indx in enumerate(points):
            # print 'Entering rt computation'
            logging.info('Computing intensities for point %i of %i' % (j+1, len(points)))
            r = self.__atmosphere.__mesh.rs[indx]
            if np.all(r==0.) or (r[0]==1. and r[1]==0. and r[2]==0.):
                pass
            else:
                I_new[j], taus_new[j] = self.compute_intensity(self.__atmosphere.__mesh.rs[indx], self.__atmosphere.__mesh.ns[indx])
  
        return I_new, taus_new


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



class ContactBinaryRadiativeTransfer(RadiativeTransfer):

    def __init__(self, atmosphere, **kwargs):

        quadrature = kwargs.pop('quadrature', 'Lebedev')
        ndir = kwargs.pop('ndir', 15)
        super(ContactBinaryRadiativeTransfer,self).__init__(atmosphere, quadrature=quadrature, ndir=ndir)

    def compute_potentials(self, points, q, bbT1, bbT2, pot_range_grid):

        pots = np.zeros(len(points))
        center1 = np.all(points == 0., axis=1)
        center2 = (points[:, 0] == 1.) & (points[:, 1] == 0.) & (points[:, 2] == 0.)
        pots[~center1 & ~center2] = potentials.roche.BinaryRoche_cartesian(points[~center1 & ~center2], q)
        pots[center1] = bbT1[:,0].max()
        pots[center2] = bbT2[:,0].max()
        pots[(np.round(pots, 8) >= np.round(pot_range_grid[0], 8)) & (pots < pot_range_grid[0])] = pot_range_grid[0]
        
        return pots

    def compute_initial_Is(self):

        if self.__atmosphere.__mesh._geometry == 'spherical':
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is(component='1')
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is(component='2')

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':
            super(ContactBinaryRadiativeTransfer,self).compute_initial_Is()
        
        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)


    def compute_interpolation_functions(self, iter_n=1):
        
        potT_bb1, potrho_bb1 = self.blackbody_interpolation_functions(self.__atmosphere.__directory, component='1')
        potT_bb2, potrho_bb2 = self.blackbody_interpolation_functions(self.__atmosphere.__directory, component='2')

        if self.__atmosphere.__mesh._geometry == 'spherical':

            chi1_interp, J1_interp, I1_interp = self.mesh_interpolation_functions(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='1', iter_n=iter_n)
            chi2_interp, J2_interp, I2_interp = self.mesh_interpolation_functions(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='2', iter_n=iter_n)

            self._interp_funcs = {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                    'chi1': chi1_interp, 'chi2': chi2_interp, 'J1': J1_interp, 'J2': J2_interp,
                    'I1': I1_interp, 'I2': I2_interp}

        elif self.__atmosphere.__mesh._geometry == 'cylindrical':

            chi_interp, J_interp, I_interp = self.mesh_interpolation_functions(directory=self.__atmosphere.__directory, mesh=self.__atmosphere.__mesh, component='', iter_n=iter_n)
            
            self._interp_funcs = {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                      'chi': chi_interp, 'J':J_interp, 'I': I_interp}

        else:
            raise ValueError('Geometry %s not supported with rt_method cobain' % self.__atmosphere.__mesh._geometry)
    

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
