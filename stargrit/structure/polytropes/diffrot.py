from stargrit.structure.polytropes.spherical import Polytrope
from stargrit.structure import potentials
import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import astropy.constants as const 
import astropy.units as u


class DiffrotPolytrope(object):

    def __init__(self, starinstance, **kwargs):
        """
        Solves the Lane-Emden equation of a differentially rotating polytrope.

        Model adapted from Mohan et. al (1992).

        Parameters
        ----------
        n: float
            Polytropic index of the model
        bs: array of shape (3,0)
            Three parameters describing the differentially rotating model.
            From Mohan et al. (1992):
            .. math::
                \omega^2 = bs[0] + bs[1] s^2 + bs[2] s^4
            where s is the distance of a point from the rotation axis:
            .. math::
                s^2 = x^2 + y^2
        starinstance: instance of stargrit.objects.star
            Used to reference and update shared parameters, like 
            mass, radius, teff and mass_fractions.

        Methods
        -------
        _le_dimless
            Computes the dimensionless solution of the Lane-Emden equation
            of a differentially rotating polytrope.
        _lane_emden
            The full Lane-Emden solution with central and surface values.
        _area_volume
            Approximate dimensionless area and volume of the star.
        _bb_files
            Saves the blackbody files for interpolation outside of mesh.
        _structure_files
            Saves the mesh structure files.
        _compute
            Calls all of the above the compute and store the polytropic 
            model structure.
        _params
            Returns updateable parameters.

        Attributes
        ----------
        model
        n
        bs
        starinstance
            mass
            radius
            teff
            mass_fractions
            mu
        """

        self.n = kwargs.get('n', 3.)
        self.bs = kwargs.get('bs', [0.,0.,0.])
        self.__star = starinstance
        self._lane_emden()


    @property
    def model(self):
        return 'diffrot'


    @property    
    def bs(self):
        """
        Parameter array defining the differentially rotating model.

        From Mohan et al. (1992):
                .. math::
                    \omega^2 = bs[0] + bs[1] s^2 + bs[2] s^4
                where s is the distance of a point from the rotation axis:
                .. math::
                    s^2 = x^2 + y^2
        """
        return self.__bs

    @bs.setter
    def bs(self,value):
        # check if right type and shape and set or raise error
        if isinstance(value, (list,np.ndarray)):
            value = np.array(value)
            if value.shape == (3,):
                self.__bs = value
            else:
                raise TypeError('Wrong array shape: {}. \
                bs parameter array needs to have shape (3,)'.format(value.shape))
        else:
            raise TypeError('bs parameter needs to be an array of shape (3,).')

    
    @property
    def n(self):
        """
        Polytropic index of the model.
        """
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value

    
    @property
    def mass(self):
        """
        Mass of the star (default unit 1 Solar mass).
        """
        return self.__star.mass
    

    @mass.setter
    def mass(self, value):
        self.__star.mass = value

    
    @property
    def radius(self):
        """
        Radius of the star (default unit 1 Solar radius).
        """
        return self.__star.radius
    

    @radius.setter
    def radius(self, value):
        self.__star.radius = value


    @property
    def teff(self):
        """
        Effective temperature (defined at optical depth = 2/3) of the star (default unit K).
        """
        return self.__star.teff
    

    @teff.setter
    def teff(self, value):
        self.__star.teff = value

    
    @property
    def mass_fractions(self):

        return self.__star.mass_fractions
    

    @mass_fractions.setter 
    def mass_fractions(self, value):
        value = np.array(value)
        if value.shape == (3,):
            (X,Y,Z) = value
            self.__star.mass_fractions = (X,Y,Z)
        else:
            raise TypeError('Mass fractions (X,Y,Z) need to be input as a tuple\
            or numpy array of shape (3,)')


    @property
    def mu(self):
        return self.__star.mu

    
    @property
    def scale(self):
        """Scaling constant to convert dimensionless radial distance
        to physical."""
        return self.__scale

    
    @property 
    def pot(self):
        """Surface potential."""
        return self.__pot


    def _le_dimless(self, xiu):

        """
        Computes the differential polytrope solution based on Mohan et al. (1992)

        Parameters
        ----------

        xiu: float
            The surface value of the dimensionless radial variable
            of the equivalent spherical polytrope.

        Returns
        -------
        t_surface: float
            Dimensionless radial value of the surface.
        dtheta_surface: float
            Derivative of the theta function at the surface.
        ts_theta: numpy.ndarray
            Solution of the differentially rotating Lane-Emden eq.
        """
        
        bs = self.bs
        n = self.n
        
        def f(y, t):
            def A(t):
                return t ** 2 * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 
                - 8. / 105. * bs[0] * bs[1] * t ** 8 
                - (16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 
                - 196. / 525. * bs[0] ** 2 * bs[1] * t ** 11 
                - 128. / 3405. * bs[1] * bs[2] * t ** 12)

            def dA(t):
                return 2 * t * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 
                - 8. / 105. * bs[0] * bs[1] * t ** 8 - 
                (16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 
                - 196. / 525. * bs[0] ** 2 * bs[1] * t ** 11 
                - 128. / 3405. * bs[1] * bs[2] * t ** 12) 
                + t ** 2 * (-6. / 15. * bs[0] ** 2 * t ** 5 
                - 8. * 8. / 105. * bs[0] * bs[1] * t ** 7 
                - 10 * (16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 9 
                - 196. * 11. / 525. * bs[0] ** 2 * bs[1] * t ** 10 
                - 128. * 12. / 3405. * bs[1] * bs[2] * t ** 11)

            def B(t):
                return 1 + 2 * bs[0] * t ** 3 + 16. / 15. * bs[1] * t ** 5 
                + 24. / 5. * bs[0] ** 2 * t ** 6 + 16. / 21. * bs[2] * t ** 7 
                + 44. / 7. * bs[0] * bs[1] * t ** 8 
                + (1664./315. * bs[0] * bs[2] + 208./105. * bs[1]**2) * t ** 10 
                + 2912./105. * bs[0]**2 * bs[1] * t ** 11 
                + 2240./693. * bs[1] * bs[2] * t ** 12

            return [y[1],(-xiu ** 2 * t ** 2 * B(t) * np.abs(y[0]) ** n - dA(t) * y[1])/A(t)]

        # NOTE: odeint not good for bvp, results differ from paper
        # at 3rd decimal. For now this is ignored.

        y0 = [1., 0.]
        ts = np.arange(1e-120, 1.1, 1e-4)
        soln = odeint(f, y0, ts)

        # func has the form (xi, theta, dtheta/dxi)
        theta_interp = interp1d(ts, soln[:,0], fill_value='extrapolate')
        dtheta_interp = interp1d(ts, soln[:,1], fill_value='extrapolate')

        negtheta = int(np.argwhere(theta_interp(ts) < 0)[1])
        # compute the value of t and dthetadt where theta falls to zero
        ts_theta_interp = interp1d(soln[:,0][:negtheta], ts[:negtheta])
        t_surface = float(ts_theta_interp(0.))
        dtheta_surface = float(dtheta_interp(t_surface))

        return t_surface, dtheta_surface, np.array([ts, soln[:, 0]])


    def _lane_emden(self):

        """
        Compute the all functions and variables related to the polytrope.

        Assigns a ._le attribute to the class that contains a dictionary
        of all the relevant parameters and their computed values.
        """

        # compute equivalent spherical polytrope
        pt = Polytrope(self.n)
        xi_s, dthetadxi_s, theta_interp_xi = pt()

        # self.radius is the undistorted radius of the outermost potential surface
        rn = self.radius / xi_s
        rhoc = (-1) * self.mass / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)
        Tc = (1. / ((self.n + 1) * xi_s * ((-1) * dthetadxi_s)) * const.G * self.mass * self.mu * const.m_p / const.k_B / self.radius).to(u.K)
        # compute the polytrope of the distorted star
        r0_s, dthetadr0_s, theta_soln = self._le_dimless(xi_s)
        area_volume = self._area_volume(r0_s, xi_s, alpha=1.*u.R_sun/xi_s)

        self.__le = {'r0_s': r0_s, 'area': area_volume['area'], 
        'volume': area_volume['volume'], 'theta_soln': theta_soln, 
        'rhoc': rhoc, 'Tc': Tc}

        self._compute_surface_pot()


    @property
    def le(self):
        return self.__le

        
    def _area_volume(self, r0, xiu, alpha=1.):
        """
        Approximate area and volume of a differentially rotating polytrope.
        """

        bs = self.bs

        V = 4. * np.pi / 3. * (xiu*alpha) ** 3 * r0 ** 3 * (
            1 + bs[0] * r0 ** 3 + 0.4 * bs[1] * r0 ** 5 + 1.6 * bs[0] ** 2 * r0 ** 6 + 8. / 35. * bs[
                2] * r0 ** 7 + 12. / 7. * bs[0] * bs[1] * r0 ** 8 + (
                128. / 105. * bs[0] * bs[2] + 16. / 35. * bs[1] ** 2) * r0 ** 10 + 208. / 35. * bs[0] ** 2 * bs[
                1] * r0 ** 11 + 64. / 99. * bs[1] * bs[2] * r0 ** 12)

        S = 4. * np.pi * (xiu*alpha) ** 2 * r0 ** 2 * (
            1 + 2. / 3. * bs[0] * r0 ** 3 + 4. / 15. * bs[1] * r0 ** 5 + 14. / 15. * bs[0] ** 2 * r0 ** 6 + 16. / 105. *
            bs[
                2] * r0 ** 7 + 36. / 35. * bs[0] * bs[1] * r0 ** 8 + (
                704. / 945. * bs[0] * bs[2] + 88. / 315. * bs[1] ** 2) * r0 ** 10 + 1024. / 315. * bs[0] ** 2 * bs[
                1] * r0 ** 11 + 832. / 2079. * bs[1] * bs[2] * r0 ** 12)

        return {'volume': V, 'area': S}

    
    def _compute_surface_pot(self):

        """
        Computes the surface potential value for the polytropic model.
        """

        pot = 1./self.le['r0_s']
        r0s = np.linspace(1./pot - 0.1, 1./pot, 10000)
        op_depths = np.zeros(len(r0s))

        theta_interp = interp1d(self.le['theta_soln'][0],self.le['theta_soln'][1])
        for i, r0 in enumerate(r0s):
            T = self.le['Tc'] * theta_interp(r0)
            if T > 0:
                op_depths[i] = ((T / self.teff) ** 4 - 0.5) * 4.0 / 3.0
            else:
                op_depths[i] = 0.0
            # print 'diffrot pot = %.2f T = %.2f, op_depth = %.2f' % (1. / r0, T, op_depths[i])

        r0_intp = interp1d(op_depths, r0s)

        r0_tau1 = r0_intp([2.0 / 3.0])[0]

        self.__pot = 1./r0_tau1

        # the potentials in the mesh will be given arbitrarily
        # need to readjust so that lowest potential corresponds to surface
        if self.__star.mesh.coords['pots'][0] == 1.:
            self.__star.mesh.coords['pots'] = self.__pot*self.__star.mesh.coords['pots']
        else:
            potdiff = self.__pot/self.__star.mesh.coords['pots'][0]
            self.__star.mesh.coords['pots'] = potdiff*self.__star.mesh.coords['pots']

        # compute the scaling so that the surface potential at tau=2/3 corresponds to the user-provided radius
        requiv = (3*self.le['volume']/(4*np.pi))**(1./3.)
        
        self.__scale = self.radius/requiv
        

    def _bb_files(self):
        """
        Computes and saves the blackbody file (pot, T, rho).

        Used in RT to compute structure when ray goes outside of grid.
        """

        # make pots that go beyond the mesh for black body interpolation outside of mesh
        pots = self.__star.mesh.coords['pots']
        theta_interp = interp1d(self.le['theta_soln'][0], self.le['theta_soln'][1])

        pots_bb = np.linspace(pots[0], pots[1]+10, 10000)
        theta_pots_bb = theta_interp(1./pots_bb)
        Ts_bb = self.le['Tc']*pots_bb
        rhos_bb = self.le['rhoc'] * theta_pots_bb ** self.n

        bb_file = np.array([pots_bb, Ts_bb, rhos_bb]).T
        np.save(self.__star.directory+'potTrho_bb', bb_file)


    def _structure_files(self):
        """
        Computes and saves the structure files (density, temperature).
        """

        theta_interp = interp1d(self.le['theta_soln'][0],self.le['theta_soln'][1], fill_value='extrapolate')
        
        # this logic needs to be replaced based on the point indices
        # for the first iteration when populating with polytropes, all we need is the
        # potential corresponding to the given index for interpolation

        mesh = self.__star.mesh

        points = np.arange(0,len(mesh.rs),1)
        theta_pots = theta_interp(1./mesh.coords['pots'])

        Ts_pots = self.le['Tc'] * theta_pots
        rhos_pots = self.le['rhoc'] * theta_pots ** self.n
        pots_inds = points//(mesh.dims[1]*mesh.dims[2])
        
        np.save(self.__star.directory+'T_0', Ts_pots[pots_inds].value.reshape(mesh.dims))
        np.save(self.__star.directory+'rho_0', rhos_pots[pots_inds].value.reshape(mesh.dims))
            

    def _params(self):
        """
        Returns
        -------
        List of updateable parameters.
        """
        return [key for key in dir(self) if not key.startswith('_')]


    def _compute(self, **kwargs):
        """
        Computes the differentially rotating Lane-Emden eq. and structure files.

        Can accept all relevant parameters on call and update them before compute.
        A mesh object with mesh.coords needs to exist before calling structure.compute

        Returns
        -------
        Temperature and density saved in files 'T_0.npy' and 'rho_0.npy'
        as structured arrays of shape mesh.dims
        """

        # if some parameters provided in kwargs, update values here

        if kwargs:
            newparams = set(kwargs.keys()) & set(self._params())
            if newparams:
                for param in newparams:
                    setattr(self,param,kwargs[param])
            
            self._lane_emden()

        if self.__star.mesh == None:
            raise ValueError('Mesh object needs to be created first.')
        else:
            self._bb_files()
            self._structure_files()


    def _compute_radius(self, pot, direction):

        """
        Computes the radius at a given direction for an equpotential.

        Parameters
        ----------
        pot: float
            Value of the equipotential
        directions: list, np.ndarray (3,)
            Direction cosines array
        """

        r = potentials.diffrot.radius_newton(pot, self.bs, np.arccos(direction[2]))
        return r*direction
                    

    def _compute_normal(self, r_cs):

        """
        Computes the normal to the equipotential at a given point on surface.

        Parameters
        ----------
        r_cs: list, np.ndarray (3,)
            Grid point at which to compute the normal.
        """
        

        nx = potentials.diffrot.dDiffRotRochedx(r_cs, self.bs)
        ny = potentials.diffrot.dDiffRotRochedy(r_cs, self.bs)
        nz = potentials.diffrot.dDiffRotRochedz(r_cs, self.bs)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)

        return np.array([nx / nn, ny / nn, nz / nn])
