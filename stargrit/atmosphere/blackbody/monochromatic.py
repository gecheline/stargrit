import numpy as np 
import astropy.units as u
import astropy.constants as c
from stargrit.atmosphere import interpolate
from stargrit.atmosphere.blackbody.gray import GrayBlackbody
from stargrit.geometry.spherical import ContactBinarySphericalMesh
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class MonochromaticBlackbody(GrayBlackbody):

    # The monochromatic atmosphere retains the same basic functionality
    # as the gray atmosphere, with slight differences in the functions
    # requiring wavelength dependence to be taken into account

    def __init__(self, starinstance, **kwargs):
        """
        Initializes a monochromatic atmosphere.

        TODO: more info on monochromatic atmosphere definition and assumptions

        Parameters
        ----------
        
        starinstance: instance of stargrit.objects.star
            Used to reference and update shared parameters
            and methods.


        Attributes
        ----------
        wavelengths
        opactype
        default_units

        Methods
        -------
        _load_structure
        _compute_absorption_coefficient
        _compute_source_function
        _compute
            Computes and stores the initial atmosphere arrays:
            absorption coefficient (chis) and source function (Ss).
        """

        super(MonochromaticBlackbody,self).__init__(starinstance, **kwargs)
        self.opactype = kwargs.get('opactype','mean')
        self.wavelengths = kwargs.get('wavelengths', np.linspace(4000,8000,1000)*u.Angstrom)


    @property
    def opactype(self):
        return self.__opactype 


    @opactype.setter
    def opactype(self, value):
        self.__opactype = value
        #TODO: will need to recompute the chi array here as well


    @property
    def wavelengths(self):
        return self.__wavelengths


    @wavelengths.setter 
    def wavelengths(self, value):
        if hasattr(value, '_unit'):
            self.__wavelengths = value.to(self.default_units['wavelength'])
        else:
            logging.info('Assuming wavelengths array is in default unit Angstrom.')
            self.__wavelengths = value*u.Angstrom


    def _compute_absorption_coefficient(self, rhos, Ts):

        if self.opactype=='mean':
            opacs = interpolate.opacities(Ts.value, rhos.to('g/cm3').value)*u.m**2/u.kg  
            # this opac is in m^2/kg

            return opacs * rhos * self.default_units['chi']

        elif self.opactype=='monochromatic':
            raise NotImplementedError

        else:
            raise ValueError('Opacity type %s not supported, can only be mean or monochromatic' % self.opactype)


    def _compute_source_function(self, Ts):
        
        # units magic needs to be here to avoid numerical 0s and infinities
        # Ss=((1./np.pi) * 2*c.h*c.c**2/(self.wavelengths.to(u.m))**5 * 1./
        #     (np.exp(c.h.to(u.kg*u.nanometer**2/u.s)*c.c/(c.k_B.to(u.kg*u.nanometer**2/u.s**2/u.K)*Ts[:,None]*(self.wavelengths.to(u.m))))-1.)
        #     ).to(self.default_units['S'])

        Ss=((1./np.pi) * 2*c.h*c.c**2/(self.wavelengths.to(u.m))**5 * 1./
            (np.exp(c.h*c.c/(c.k_B*Ts[:,None]*(self.wavelengths)))-1.)
            ).to(self.default_units['S'])

        return Ss


    def _compute_atmosphere(self, component=''):

        directory = self.star.directory
        mesh = self.star.mesh

        rhos, Ts = self._load_structure(component=component)
        chis = self._compute_absorption_coefficient(rhos=rhos, Ts=Ts)
        Ss = self._compute_source_function(Ts=Ts)
        
        if self.opactype == 'mean':
            np.save(directory+'chi%s_0.npy' % component, chis.value.reshape(mesh.dims))
        else:
            raise NotImplementedError

        np.save(directory+'S%s_0.npy' % component, Ss.value.reshape(tuple(mesh.dims)+(len(self.wavelengths),)))


    @property
    def default_units(self):
        """
        Default units used by the atmosphere arrays.

        Default units throughout the code should not be edited because they strongly
        affect the radiative transfer computations.

        Returns
        -------
        dict
            A dictionary of attributes with assigned default (astropy) units.
        """
        return {'opac': u.R_sun**2/u.M_sun, 'chi': 1./u.R_sun, 
                'S': u.L_sun/u.R_sun**2/u.Angstrom, 'wavelength': u.Angstrom}