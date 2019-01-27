from stargrit.objects.object import Object
import astropy.units as u 
import numpy as np


class Star(Object):

    def __init__(self, geometry = 'spherical', structure='polytropes:diffrot', 
                atmosphere='blackbody:gray', **kwargs):

        """
        Initializes a Star class instance.

        The Star class initializes an Object with object_type 'star',
        a spherical mesh and choice of hydrodynamical structure and atmosphere model.

        Parameters
        ----------
        geometry: str
            Geometry of the mesh. Default and currently only supported value is 'spherical'.
        structure: str
            Hydrodynamical model of the structure (default 'polytropes:diffrot').
        atmosphere: str
            Atmosphere model (default 'blackbody:gray').
        **kwargs
            Any choice of parameter values associated with the overall model.


        Attributes
        ----------
        mass
        radius
        teff


        Methods
        -------
        set_object_attributes
            Generates the attributes associated with the Star class.
        implemented_geometries
            Returns a list the geometries implemented for Star class.
        implemented_structures
            Returns a list of the structures compatible with Star class.
        default_units
            Returns a dictionary of the attributes and their corresponding default units.

        """

        super(Star,self).__init__(object_type='star', geometry=geometry, 
                                    structure=structure, atmosphere=atmosphere, **kwargs)


    def set_object_attributes(self, **kwargs):
        """
        Adds parameters associated with object type.

        The values of parameters can be provided as floats or astropy quantities.
        If floats, the default unit is assumed.

        Parameters
        ----------
        **kwargs - if provided when initializing class, the values will be set 
                    to input values, otherwise default
            mass - default 1. Solar mass
            radius - default 1. Solar radius
            teff - default 5777 K

        """

        # collect values if provided in kwargs or set to default
        massv = kwargs.get('mass', 1.*u.M_sun)
        radiusv = kwargs.get('radius', 1.*u.R_sun)
        teffv = kwargs.get('teff', 5777.*u.K)
        (X, Y, Z) = kwargs.get('XYZ',(0.7381,0.2485,0.0134))

        # add attributes and set collected values
        self.mass = massv 
        self.radius = radiusv 
        self.teff = teffv
        self.mass_fractions = (X,Y,Z)


    @property
    def mass(self):
        """
        Mass of the star (default unit 1 Solar mass).
        """
        return self.__mass
    

    @mass.setter
    def mass(self, value):
        if hasattr(value,'unit'):
            self.__mass = value
        else:
            self.__mass = value*self.default_units()['mass']

    
    @property
    def radius(self):
        """
        Radius of the star (default unit 1 Solar radius).
        """
        return self.__radius
    

    @radius.setter
    def radius(self, value):
        if hasattr(value,'unit'):
            self.__radius = value
        else:
            self.__radius = value*self.default_units()['radius']


    @property
    def teff(self):
        """
        Effective temperature (defined at optical depth = 2/3) of the star (default unit K).
        """
        return self.__teff
    

    @teff.setter
    def teff(self, value):
        if hasattr(value,'unit'):
            self.__teff = value
        else:
            self.__teff = value*self.default_units()['teff']

    @property
    def mass_fractions(self):
        """
        Mass fractions of H (X), He (Y) and metals (Z).
        """
        return self.__mass_fractions
    

    @mass_fractions.setter 
    def mass_fractions(self, value):
        value = np.array(value)
        if value.shape == (3,):
            (X,Y,Z) = value
            self.__mass_fractions = (X,Y,Z) 
        else:
            raise TypeError('Mass fractions (X,Y,Z) need to be input as a tuple\
            or numpy array of shape (3,)')


    @property
    def mu(self):
        """
        Mean molecular mass based on mass fractions.

        Assuming fully ionized plasma, computed through:
        .. math::
            \mu = (2X+0.75Y+0.5Z)^{-1}
        """
        (X,Y,Z) = self.__mass_fractions
        return 1./(2.*X + 0.75*Y + 0.5*Z)


    def implemented_geometries(self):
        """Returns implemented geometries associated with the object."""
        return ['spherical']


    def implemented_structures(self):
        """Returns implemented structure models associated with the object."""
        return ['polytropes:diffrot']

    
    def default_units(self):
        """
        Default units used by the object attributes.

        Default units throughout the code should not be edited because they strongly
        affect the radiative transfer computations.

        Returns
        -------
        dict
            A dictionary of attributes with assigned default (astropy) units.
        """
        return {'mass': u.M_sun, 'radius': u.R_sun, 'teff': u.K}