import numpy as np
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class SphericalMesh(object):

    def __init__(self, **kwargs):
        """
        A spherical mesh is initialized with a coordinate grid in (pot, theta, phi).

        The initial potential array is dimensionless, to be scaled based on object
        type and surface potential computed from structure or provided by user.

        Parameters
        ----------
        **kwargs
            Can be:
            dims: list, np.ndarray
                Dimensions of the grid (Npot, Ntheta, Nphi)
            atm_range: float
                Relative size of the atmosphere with respect to surface potential.
        """

        self.__coords = None # required when setting dims and atm_range for the first time
        self.dims = kwargs.get('dims', [50,50,50])
        self.atm_range = kwargs.get('atm_range', 0.01)

        pots = np.linspace(1.,1.+self.atm_range,self.dims[0])
        thetas = np.linspace(0., np.pi/2, self.dims[1])
        phis = np.linspace(0., np.pi, self.dims[2])

        self.__coords = {'pots': pots, 'thetas': thetas, 'phis': phis}


    @property
    def dims(self):
        return self.__dims

    @dims.setter 
    def dims(self, value):
        if isinstance(value, (list,np.ndarray)):
            value = np.array(value)
            if value.shape == (3,):
                self.__dims = value
                if self.__coords != None:
                    # recompute coords
                    pots = self.__coords['pots'][0]*np.linspace(1.,1.+self.atm_range,self.__dims[0])
                    thetas = np.linspace(0., np.pi/2, self.__dims[1])
                    phis = np.linspace(0., np.pi, self.__dims[2])
                    self.__coords = {'pots': pots, 'thetas': thetas, 'phis': phis}

            else:
                raise TypeError('Wrong array shape: {}. \
                dims parameter array needs to have shape (3,)'.format(value.shape))
        else:
            raise TypeError('dims parameter needs to be an array of shape (3,).')

    @property
    def atm_range(self):
        return self.__atm_range

    
    @atm_range.setter 
    def atm_range(self, value):
        self.__atm_range = value 
        if self.__coords != None:
            self.__coords['pots'] = self.__coords['pots'][0]*np.linspace(1.,1.+self.__atm_range,self.__dims[0])


    @property
    def coords(self):
        return self.__coords


class StarSphericalMesh(SphericalMesh):


    def __init__(self, starinstance, **kwargs):
        super(StarSphericalMesh,self).__init__(**kwargs)
        self.__star = starinstance


    def _params(self):
        """
        Returns
        -------
        List of updateable parameters.
        """
        return [key for key in dir(self) if not key.startswith('_')]

    
    def _compute(self, **kwargs):

        if kwargs:
            newparams = set(kwargs.keys()) & set(self._params())
            if newparams:
                for param in newparams:
                    setattr(self,param,kwargs[param])


        meshsize = self.dims[0]*self.dims[1]*self.dims[2]
        rs = np.zeros((meshsize, 3))
        normals = np.zeros((meshsize, 3))

        n = 0
        for i, pot in enumerate(self.coords['pots']):
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(self.coords['thetas']):
                for k, phi in enumerate(self.coords['phis']):
                    
                    direction = np.array([np.sin(theta) * np.cos(phi), 
                                        np.sin(theta) * np.sin(phi), 
                                        np.cos(theta)])

                    rs[n] = self.__star.structure._compute_radius(pot=pot, direction=direction)
                    normals[n] = self.__star.structure._compute_normal(rs[n])

                    n+=1

        self.rs = rs * self.__star.structure.scale
        self.ns = normals


class ContactBinarySphericalMesh(SphericalMesh):


    def __init__(self, starinstance, **kwargs):
        super(ContactBinarySphericalMesh,self).__init__(**kwargs)
        self.__star = starinstance

