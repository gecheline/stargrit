import numpy as np 
import astropy.units as u
import astropy.constants as c
from stargrit.atmosphere import interpolate
from stargrit.geometry.spherical import ContactBinarySphericalMesh

import logging

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class GrayBlackbody(object):

    def __init__(self, starinstance, **kwargs):
        """
        Initializes a gray atmosphere.

        TODO: more info on gray atmosphere definition and assumptions

        Parameters
        ----------
        
        starinstance: instance of stargrit.objects.star
            Used to reference and update shared parameters
            and methods.


        Attributes
        ----------
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
        self.__star = starinstance 

    @property
    def star(self):
        return self.__star 

    @property
    def teff(self):
        return self.__star.teff

    @teff.setter
    def teff(self,value):
        self.__star.teff = value
        logging.info("Recomputing structure with new temperature!")
        self.__star.structure._compute()

    @property
    def atm_range(self):
        return self.__star.mesh.atm_range

    @atm_range.setter
    def atm_range(self,value):
        self.__star.mesh.atm_range = value


    @property
    def dims(self):
        return self.__star.mesh.dims

    @dims.setter
    def dims(self,value):
        self.__star.mesh.dims = value


    def _load_structure(self, component=''):
        
        rhos = np.load(self.__star.directory+'rho%s_0.npy' % component
                        ).flatten() * self.__star.structure.default_units['rho']
        Ts =  np.load(self.__star.directory+'T%s_0.npy' % component
                        ).flatten()* self.__star.structure.default_units['T']

        return rhos, Ts


    def _compute_absorption_coefficient(self, rhos, Ts):

        opacs = np.zeros(len(Ts))*u.m**2/u.kg  
        opacs[(Ts>0) & (rhos>0)] = interpolate.opacities(Ts[(Ts>0) & (rhos>0)].value, rhos[(Ts>0) & (rhos>0)].to('g/cm3').value)*u.m**2/u.kg 
        # this opac is in m^2/kg
        return (opacs * rhos).to(self.default_units['chi'])


    def _compute_source_function(self, Ts):

        Ss = (1. / np.pi) * (c.sigma_sb * Ts ** 4).to(self.default_units['S'])
        return Ss


    def _compute_atmosphere(self, component=''):

        directory = self.__star.directory
        mesh = self.__star.mesh

        rhos, Ts = self._load_structure(component=component)
        chis = self._compute_absorption_coefficient(rhos=rhos, Ts=Ts)
        Ss = self._compute_source_function(Ts=Ts)
        
        np.save(directory+'chi%s_0.npy' % component, chis.value.reshape(mesh.dims))
        np.save(directory+'S%s_0.npy' % component, Ss.value.reshape(mesh.dims))


    def _params(self):
        """
        Returns
        -------
        List of updateable parameters.
        """
        return [key for key in dir(self) if not key.startswith('_')]


    def _compute(self, component='', **kwargs):

        if kwargs:
            newparams = set(kwargs.keys()) & set(self._params())
            if newparams:
                for param in newparams:
                    setattr(self,param,kwargs[param])
                if 'dims' in newparams or 'atm_range' in newparams:
                    logging.info("Recomputing mesh and structure with new atmosphere dims/range!")
                    self.__star.mesh._compute()
                    self.__star.structure._compute()
                    

        if isinstance(self.__star.mesh, ContactBinarySphericalMesh):
            self._compute_atmosphere(component='1')
            self._compute_atmosphere(component='2')

        else:
            self._compute_atmosphere()


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
                'S': u.L_sun/u.R_sun**2}

