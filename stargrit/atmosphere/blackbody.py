import numpy as np 
import astropy.units as u
import astropy.constants as c
import scipy.interpolate as spint
import quadpy.sphere as quadsph
from stargrit.atmosphere import interpolate
import logging
from stargrit import radiative_transfer

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

allowed_atmosphere_types = ['grey', 'gray', 'monochromatic']
implemented_rt_methods = ['cobain']

class Atmosphere(object):

    def __init__(self, atm_type='grey', rt_method='cobain', opactype='mean', **kwargs):

        if atm_type in allowed_atmosphere_types: #and rt_method in implemented_rt_methods:
            self._atm_type=atm_type.replace('e', 'a') if atm_type=='grey' else atm_type 
            self._rt_method = rt_method
            # quadrature and ndir are attributes of the cobain RT method
            self._opactype = opactype

            if atm_type=='monochromatic':
                x = kwargs.get('x', 'wavelength')
                if x=='wavelength':
                    min = kwargs.get('min', 4000*u.Angstrom)
                    max = kwargs.get('max', 8000*u.Angstrom)
                    n = kwargs.get('n', 1000)
                    self._wavelengths = np.linspace(min,max,n)
                    self._frequencies = (c.c/self._wavelengths).to('1/s')
                elif x=='frequency':
                    min = kwargs.get('min', 3.7e+14*u.Hz)
                    max = kwargs.get('max', 7.5e+14*u.Hz)
                    n = kwargs.get('n', 1000)
                    self._frequencies = np.linspace(min,max,n)
                    self._wavelengths = (c.c/self._frequencies).to(u.Angstrom)
                else:
                    raise ValueError('Unsupported atmosphere variable x=%s, must be one of [wavelength, frequency]' % x)
            
        else:
            raise ValueError('Atmosphere type %s not recognized, can only be one of %r' % (atm_type, allowed_atmosphere_types))

    def load_structure(self, directory, component=''):

        rhos = np.load(directory+'rho%s_0.npy' % component).flatten() * u.M_sun/u.R_sun**3
        Ts =  np.load(directory+'T%s_0.npy' % component).flatten() * u.K
        return rhos, Ts

    def compute_chis(self, rhos, Ts, opactype='mean'):

        if opactype=='mean':
            opacs_si = interpolate.opacities(Ts.value, rhos.to('g/cm3').value)*u.m**2/u.kg  # this opac is in m^2/kg

            return opacs_si.to(u.R_sun**2/u.M_sun) * rhos

        elif opactype=='monochromatic':
            raise NotImplementedError

        else:
            raise ValueError('Opacity type %s not supported, can only be mean or monochromatic' % opactype)

    def compute_source_function(self, Ts):
        if self._atm_type=='gray':
            Ss = (1. / np.pi) * (c.sigma_sb * Ts ** 4).to(u.L_sun/u.R_sun**2)
            return Ss

        elif self._atm_type=='monochromatic':
            # not so simple, need to figure out T and wavelength multiplication
            Ss = np.zeros((len(Ts),len(self._wavelengths)))
            for i,T in enumerate(Ts):
                Ss[i] = ((1./np.pi) * 2*c.h*c.c**2/(self._wavelengths.to(u.m)**5 * np.exp(c.h*c.c/(c.k_B*T*self._wavelengths.to(u.m))-1.))).to(u.L_sun/u.R_sun**2/u.Angstrom).value

            return Ss
        
        else:
            raise ValueError('Atmosphere type %s not recognized, can only be one of %r' % (self._atm_type, allowed_atmosphere_types))

    def compute_atmosphere(self, directory, mesh, component=''):

        rhos, Ts = self.load_structure(directory=directory, component=component)
        chis = self.compute_chis(rhos=rhos, Ts=Ts, opactype=self._opactype)
        Ss = self.compute_source_function(Ts=Ts)
        
        if self._opactype == 'mean':
            np.save(directory+'chi%s_0.npy' % component, chis.value.reshape(mesh._dims))
        else:
            raise NotImplementedError
        if self._atm_type == 'gray':
            np.save(directory+'S%s_0.npy' % component, Ss.value.reshape(mesh._dims))
        elif self._atm_type == 'monochromatic':
            #TODO: test if this works with the extra wavelength dimension or will need to be rewritten
            np.save(directory+'S%s_0.npy' % component, Ss.reshape(tuple(mesh._dims)+(len(self._wavelengths),)))
        else:
            raise ValueError('Atmosphere type %s not recognized, can only be one of %r' % (self._atm_type, allowed_atmosphere_types))


class DiffrotStarAtmosphere(Atmosphere):

    def __init__(self, mesh, directory, **kwargs):

        self.mesh = mesh
        self.directory = directory

        atm_type = kwargs.pop('atm_type', 'gray')
        quadrature = kwargs.pop('quadrature', 'lebedev')
        ndir = kwargs.pop('ndir', 15)
        rt_method = kwargs.pop('rt_method', 'cobain')
        opactype = kwargs.pop('opactype', 'mean')
        super(DiffrotStarAtmosphere,self).__init__(atm_type=atm_type, quadrature=quadrature, ndir=ndir, rt_method=rt_method, opactype=opactype, **kwargs)
        
        # rt = getattr(radiative_transfer, self._rt_method)
        
        # if hasattr(rt, 'DiffrotStar%sRadiativeTransfer' % atm_type.title()):
        #     rt_object = getattr(rt, 'DiffrotStar%sRadiativeTransfer' % atm_type.title())
        #     self.RT = rt_object(self, **kwargs) 
        # else:
        #     raise ValueError('RT method %s not supported by Star object' % self._rt_method)

    def compute_atmosphere(self):
        super(DiffrotStarAtmosphere,self).compute_atmosphere(self.directory, self.mesh)

class ContactBinaryAtmosphere(Atmosphere):

    def __init__(self, mesh, directory, **kwargs):

        self.mesh = mesh
        self.directory = directory

        atm_type = kwargs.pop('atm_type', 'gray')
        quadrature = kwargs.pop('quadrature', 'lebedev')
        ndir = kwargs.pop('ndir', 15)
        rt_method = kwargs.pop('rt_method', 'cobain')
        opactype = kwargs.pop('opactype', 'mean')
        super(ContactBinaryAtmosphere,self).__init__(atm_type=atm_type, quadrature=quadrature, ndir=ndir, rt_method=rt_method, opactype=opactype, **kwargs)

        rt = getattr(radiative_transfer, self._rt_method)

        # if hasattr(rt, 'ContactBinaryRadiativeTransfer'):
        #     rt_object = getattr(rt, 'ContactBinaryRadiativeTransfer')
        #     self.RT = rt_object(self, **kwargs) 

        # else:
        #     raise ValueError('RT method %s not supported by Contact Binary object' % self._rt_method)

    def compute_atmosphere(self):

        if self.mesh._geometry == 'cylindrical':
            super(ContactBinaryAtmosphere,self).compute_atmosphere(directory=self.directory, mesh=self.mesh)
            
        elif self.mesh._geometry == 'spherical':
            super(ContactBinaryAtmosphere,self).compute_atmosphere(directory=self.directory, mesh=self.mesh, component='1')
            super(ContactBinaryAtmosphere,self).compute_atmosphere(directory=self.directory, mesh=self.mesh, component='2')

        else:
            raise ValueError('Geometry %s not supported' % self.mesh._geometry)






