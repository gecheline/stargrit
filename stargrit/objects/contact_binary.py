from stargrit.objects.object import Object
from stargrit.structure.potentials import roche
import astropy.units as u 


class ContactBinary(Object):

    def __init__(self, geometry = 'cylindrical', structure='polytropes:tidal', 
                atmosphere='blackbody:gray', **kwargs):

        """
        Initializes a Contact Binary class instance.

        The Contact Binary class initializes an Object with object_type 'contact_binary',
        a cylindrical mesh and choice of hydrodynamical structure and atmosphere model.

        Parameters
        ----------
        geometry: str
            Geometry of the mesh (default is 'cylindrical').
        structure: str
            Hydrodynamical model of the structure (default 'polytropes:tidal').
        atmosphere: str
            Atmosphere model (default 'blackbody:gray').
        **kwargs
            Any choice of parameter values associated with the overall model.


        Attributes
        ----------
        q - mass ratio (M2/M1)
        ff - fillout factor of the envelope
        pot - Roche potential corresponding to the fillout factor


        Methods
        -------
        set_object_attributes
            Generates the attributes associated with the ContactBinary class.
        implemented_geometries
            Returns a list the geometries implemented for ContactBinary class.
        implemented_structures
            Returns a list of the structures compatible with ContactBinary class.
        default_units
            Returns a dictionary of the attributes and their corresponding default units.

        """
        self.__q = 1.
        self.__ff = 0.5

        super(ContactBinary,self).__init__(object_type='contact_binary', geometry=geometry, 
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
            q - defined as M2/M1, dimensionless
            pot - value of the Roche potential of the surface
            ff - fillout factor corresponding to the surface potential

        """

        # collect values if provided in kwargs or set to default
        mass1 = kwargs.get('mass1', 1.)
        qv = kwargs.get('q', 1.)
        ffv = kwargs.get('ff', 0.5)

        # add attributes and set collected values
        self.mass1 = 1.
        self.q = qv 
        self.ff = ffv


    @property
    def q(self):
        """
        Mass of the star (default unit 1 Solar mass).
        """
        return self.__q
    

    @q.setter
    def q(self, value):

        if value <= 1.:
            self.__q = value
        else:
            raise ValueError('Please provide a value for q that is less than 1.\
            The mass ratio is defined as q=mass2/mass1, where mass2 <= mass1.')

    
    @property
    def ff(self):
        """
        Fillout factor of the contact envelope. Must be within range [0,1].
        """
        return self.__ff
    

    @ff.setter
    def ff(self, value):
        if value >= 0. and value <= 1.:
            self.__ff = value
        else:
            raise ValueError('Please provide a value between 0 and 1 \
            for the fillout factor. ff < 0 corresponds to a detached \
            system, while ff > 1 causes mass overflow through L2.')


    @property
    def pot(self):
        """
        Value of the Roche potential based on the fillout factor.
        """
        crit_pots = roche.critical_pots(self.q)
        return roche.ff_to_pot(self.ff,self.q)
        

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
        return {}