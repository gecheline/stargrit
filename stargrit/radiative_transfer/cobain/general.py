import numpy as np 
import stargrit as sg
import quadpy.sphere as quadsph
import scipy.interpolate as spint
from stargrit.structure import potentials
from stargrit.geometry.spherical import ContactBinarySphericalMesh
import logging
import random
import astropy.units as u

class RadiativeTransfer(object):

    def __init__(self, starinstance, quadrature='Lebedev:15'):

        self.__star = starinstance
        self.quadrature = quadrature


    @property
    def star(self):
        return self.__star


    @property
    def quadrature(self):
        return self.__quadrature


    @quadrature.setter 
    def quadrature(self, value):

        try:
            quadrature, degree = value.split(':')
        except:
            logging.info('Assuming value %s is quadrature type to be used with default degree=15.' % value)
            logging.info('To adjust both quadrature type and degree, provide in format quadrature:degree.')

            quadrature = value 
            degree = '15'

        if hasattr(quadsph, quadrature.title()):
            quadfunc = getattr(quadsph, quadrature.title())
            self.__quadrature = quadfunc(str(degree))
            self.__quadrature.nI = len(self.__quadrature.weights)
            # thetas and phis are in quadrature.azimuthal_polar ([:,0] is phi, [:,1] is theta) 

        else:
            raise ValueError('Quadrature %s not supported by quadpy' % value.title())


    @staticmethod
    def _rot_theta(theta):

        if isinstance(theta, (list, tuple, np.ndarray)):
            theta[theta <= np.pi*u.rad] = np.pi*u.rad-theta[theta <= np.pi*u.rad]
            theta[(theta > np.pi*u.rad) & (theta <= 1.5*np.pi*u.rad)] = theta[(theta > np.pi*u.rad) & (theta <= 1.5*np.pi*u.rad)] - np.pi*u.rad
            theta[theta > 1.5*np.pi*u.rad] = 2*np.pi*u.rad - theta[theta > 1.5*np.pi*u.rad]
            return theta

        else:
            if theta <= np.pi*u.rad:
                return np.pi*u.rad - theta
            elif theta <= 1.5 * np.pi*u.rad:
                return theta - np.pi*u.rad
            else:
                return 2 * np.pi*u.rad - theta


    @staticmethod
    def _transformation_matrix(normal):

        # Cartesian orthonormal unit vectors
        c1 = np.array([1., 0., 0.])
        c2 = np.array([0., 1., 0.])
        c3 = np.array([0., 0., 1.])

        # Roche normal orthonormal unit vectors
        if np.all(normal == np.array([0.,0.,1.])):
            tan_st_1 = np.array([1.,0.,0.])
        else:
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
    def _rotate_direction_wrt_normal(Mc, coords, R):

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


    def _blackbody_interpolation_functions(self, component=''):
        """
        Computes the interpolation in potential for temperature and density from 
        stored blackbody structure files.
        """
        potTrho_bb = np.load(self.star.directory+'potTrho_bb%s.npy' % component)
        potT_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,1])
        potrho_bb = spint.interp1d(potTrho_bb[:,0], potTrho_bb[:,2])

        return potT_bb, potrho_bb


    def _mesh_interpolation_functions(self, component='', iter_n=1):

        """
        Computes interpolation in the chosen iteration of the grid points, object and atmosphere specific.

        If the opacity and atmosphere are mean/gray, the interpolation objects are returned.
        If monochromatic, the arrays of the latest interpolation are returned, to be used in trilinear_interp method.
        """

        raise NotImplementedError


    def _compute_interpolation_functions(self, iter_n=1):

        """
        Computes the blackbody and grid interpolation functions from a chosen iteration.
        """

        potT_bb, potrho_bb = self._blackbody_interpolation_functions()
        chi_interp, S_interp, I_interp = self._mesh_interpolation_functions(iter_n=iter_n)

        self._interp_funcs = {'bbT': potT_bb, 'bbrho': potrho_bb, 'chi': chi_interp, 'S':S_interp, 'I': I_interp}


    def _compute_initial_Is(self, component=''):
        
        """
        Computes the initial intensity distribution based on the source function.

        Assumes uniform initial distribution.
        """
        Ss = np.load(self.star.directory+'S%s_0.npy' % component)

        #TODO: extend with analytical distributions
        Is = np.zeros((self.quadrature.nI,)+Ss.shape)
        for i in range(self.quadrature.nI):
            Is[i] = Ss
        
        np.save(self.star.directory+'I%s_0.npy' % component, Is)


    def _compute_intensity(self, Mc, n):
        # this technically could be implemented here as well, with checking for gray/mono in array creation
        return NotImplementedError


    def _compute_radiative_transfer(self, points, iter_n=1, ray_discretization=5000):
        
        """
        Computes radiative transfer in a given set of mesh points.
        """

        self._N = ray_discretization
        self._compute_interpolation_functions(iter_n=iter_n)
        # setup the arrays that the computation will output
        I_new = np.zeros((len(points), self.quadrature.nI))
        taus_new = np.zeros((len(points), self.quadrature.nI))

        for j, indx in enumerate(points):
            # print 'Entering rt computation'
            logging.info('Computing intensities for point %i of %i, index %i' % (j+1, len(points), indx))
            r = self.star.mesh.rs[indx]
            if np.all(r==0.) or (r[0]==1. and r[1]==0. and r[2]==0.):
                pass
            else:
                I_new[j], taus_new[j] = self._compute_intensity(self.star.mesh.rs[indx], self.star.mesh.ns[indx])
  
        return I_new, taus_new


    def _compute_rescale_factors(self, **kwargs):
        
        dims = self.star.mesh.dims
        points = random.sample(range(int(0.7*dims[0]*dims[1]*dims[2]), int(0.8*dims[0]*dims[1]*dims[2])), 5)

        rescale_factors = np.zeros(self.quadrature.nI)
        taus, Is = self._compute_radiative_transfer(points, iter_n=1, **kwargs)

        for l in range(self.quadrature.nI):
            Il = np.load(self.star.directory+'I_0_'+str(int(l))+'.npy').flatten()
            # print 'rescale:', Il[points]/Is[:,l]
            rescale_factors[l] = np.average(Il[points]/Is[:,l])

        return rescale_factors

    def _compute_iter(self, iter_n=1, **kwargs):
        # needs to import mpi4py, detect the number of available processors
        # open a file for storing, get the values returned by the processors,
        # fill them up in the file and save it.
        # each child process can open the file, assign the values to the correct
        # positions and close it.
        return None

    def _compute_mean_intensity_iter(self, iter_n=1, **kwargs):
        return None

    def _compute_flux_iter(self, iter_n=1, **kwargs):
        return None 

    def _compute_chi_iter(self, iter_n=1, **kwargs):
        return None 

    def _compute_tau_iter(self, iter_n=1, **kwargs):
        return None

    def _compute_temperatures_iter(self, iter_n=1, **kwargs):
        return None 

    

    
class DiffrotStarRadiativeTransfer(RadiativeTransfer):

    def __init__(self, starinstance, **kwargs):

        quadrature = kwargs.pop('quadrature', 'Lebedev:15')
        super(DiffrotStarRadiativeTransfer,self).__init__(starinstance, quadrature=quadrature)
    
    def _compute_potentials(self, points, bbT, pot_range_grid):
        
        pots = np.zeros(len(points))
        # center = np.all(points == 0., axis=1)
        # pots[center] = bbT[:,0].max()
        pots = potentials.diffrot.DiffRotRoche(points.value, self.star.structure.bs)
        pots[(np.round(pots, 8) >= np.round(pot_range_grid[0], 8)) & (pots < pot_range_grid[0])] = pot_range_grid[0]
        
        return pots

    def _compute_interp_regions(self, pots, points, pot_range_grid):
        
        grid = np.argwhere((pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
        le = np.argwhere(pots > pot_range_grid[1]).flatten()

        return grid, le

    def _compute_coords_for_interpolation(self, rs, **kwargs):

        thetas = np.arccos(rs[:,2] / np.sqrt(np.sum(rs ** 2, axis=1)))
        thetas[thetas > (np.pi/2)*u.rad] = self._rot_theta(thetas[thetas > (np.pi/2)*u.rad])
        phis = np.abs(np.arctan2(rs[:,1] / np.sqrt(np.sum(rs ** 2, axis=1)), rs[:,0] / np.sqrt(np.sum(rs ** 2, axis=1))))*u.rad
        
        return thetas, phis


class ContactBinaryRadiativeTransfer(RadiativeTransfer):


    def __init__(self, starinstance, **kwargs):

        quadrature = kwargs.pop('quadrature', 'Lebedev:15')
        super(ContactBinaryRadiativeTransfer,self).__init__(starinstance, quadrature=quadrature)


    def _compute_potentials(self, points, q, bbT1, bbT2, pot_range_grid):

        pots = np.zeros(len(points))
        # center1 = np.all(points == 0., axis=1)
        # center2 = (points[:, 0] == 1.) & (points[:, 1] == 0.) & (points[:, 2] == 0.)
        pots = potentials.roche.BinaryRoche_cartesian(points/self.star.structure.scale, q)
        # pots[center1] = bbT1[:,0].max()
        # pots[center2] = bbT2[:,0].max()
        pots[(np.round(pots, 8) >= np.round(pot_range_grid[0], 8)) & (pots < pot_range_grid[0])] = pot_range_grid[0]
        
        return pots

    def _compute_initial_Is(self):

        if isinstance(self.star.mesh, ContactBinarySphericalMesh):
            super(ContactBinaryRadiativeTransfer,self)._compute_initial_Is(component='1')
            super(ContactBinaryRadiativeTransfer,self)._compute_initial_Is(component='2')

        else:
            super(ContactBinaryRadiativeTransfer,self)._compute_initial_Is()
        

    def _compute_interpolation_functions(self, iter_n=1):
        
        potT_bb1, potrho_bb1 = self._blackbody_interpolation_functions(component='1')
        potT_bb2, potrho_bb2 = self._blackbody_interpolation_functions(component='2')

        if isinstance(self.star.mesh, ContactBinarySphericalMesh):

            chi1_interp, J1_interp, I1_interp = self._mesh_interpolation_functions(component='1', iter_n=iter_n)
            chi2_interp, J2_interp, I2_interp = self._mesh_interpolation_functions(component='2', iter_n=iter_n)

            self._interp_funcs = {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                    'chi1': chi1_interp, 'chi2': chi2_interp, 'S1': J1_interp, 'S2': J2_interp,
                    'I1': I1_interp, 'I2': I2_interp}

        else:

            chi_interp, J_interp, I_interp = self._mesh_interpolation_functions(component='', iter_n=iter_n)
            
            self._interp_funcs = {'bbT1': potT_bb1, 'bbT2': potT_bb2, 'bbrho1': potrho_bb1, 'bbrho2': potrho_bb2, 
                      'chi': chi_interp, 'S':J_interp, 'I': I_interp}


    @staticmethod
    def _normalize_xs(xs, pots, q):

        pots2 = pots / q + 0.5 * (q - 1) / q

        xrb1s = - sg.geometry.cylindrical.ContactBinaryMesh.get_rbacks(pots, q, 1)
        xrb2s = (np.ones(len(pots)) + sg.geometry.cylindrical.ContactBinaryMesh.get_rbacks(pots, q, 2))

        return sg.geometry.cylindrical.ContactBinaryMesh.xphys_to_xnorm_cb(xs, xrb1s, xrb2s)


    def _compute_interp_regions(self, pots, points, pot_range_grid):

        """
        Computes the different regions along a ray for interpolation: grid/blackbody, primary/secondary.
        """
        
        prim = (points[:,0] <= self.star.mesh.nekmin)
        sec = (points[:,0] > self.star.mesh.nekmin)
        le_prim = np.argwhere(prim & (pots > pot_range_grid[1])).flatten()
        le_sec = np.argwhere(sec & (pots > pot_range_grid[1])).flatten()


        if isinstance(self.star.mesh, ContactBinarySphericalMesh):

            # in spherical geometry, the two halves are built and interpolated separately
            grid_prim = np.argwhere(prim & (pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            grid_sec = np.argwhere(sec & (pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            
            return grid_prim, grid_sec, le_prim, le_sec

        else:
            # in cylindrical geometry, there is no separation between primary and secondary in the grid
            grid = np.argwhere((pots >= pot_range_grid[0]) & (pots <= pot_range_grid[1])).flatten()
            
            return grid, le_prim, le_sec
    

    def _compute_coords_for_interpolation(self, points, **kwargs):

        """
        Based on geometry, returns the grid coordinates of the ray points for interpolation.
        """

        if isinstance(self.star.mesh, ContactBinarySphericalMesh):
            
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

            thetas[thetas > np.pi / 2] = self._rot_theta(thetas[thetas > np.pi / 2])

            return thetas, phis

        else:

            grid = kwargs.pop('grid')
            pots = kwargs.pop('pots')

            xnorms_grid = self._normalize_xs(points[:, 0][grid], pots[grid], self.star.q)
            thetas_grid = np.abs(np.arcsin(points[:,1][grid]/np.sqrt(np.sum((points[:,1][grid]**2,points[:,2][grid]**2),axis=0))))
            thetas_grid[np.isnan(thetas_grid)] = 0.0
            thetas_grid[thetas_grid > np.pi/2] = self._rot_theta(thetas_grid[thetas_grid > np.pi/2])

            return xnorms_grid, thetas_grid
            