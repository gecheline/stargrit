from stargrit.objects.body import Body
import numpy as np 
import os, shutil
import logging
import stargrit as sg
import astropy.units as u
from stargrit import geometry
from stargrit import structure
from stargrit import atmosphere
from stargrit import radiative_transfer

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

implemented_object_types = ['diffrot_star', 'contact_binary']
implemented_structures = ['polytropes']
implemented_atmospheres = ['blackbody']
implemented_geometries = ['cylindrical', 'spherical']
implemented_rt_methods = ['cobain']


class DiffrotStar(Body):


    def __init__(self, geometry='spherical', structure='polytropes', atmosphere='blackbody', directory=os.getcwd()+'/', **kwargs):
        
        super(DiffrotStar, self).__init__(objtype='diffrot_star', geometry=geometry, structure=structure,atmosphere=atmosphere, directory=directory)
        
        # all of these should be converted to hidden and casted to properties
        self._mass = kwargs.get('mass', 1.0)*u.M_sun
        self._radius = kwargs.get('radius', 1.0)*u.R_sun
        self._teff = kwargs.get('teff', 5777.0)*u.K

        # because for Star many of the top-level parameter values depend on the structure
        # a structure object needs to be added in init, without compute
        self.add_structure(compute=False)
        self._pot, self.__scale = self._structure.compute_surface_pot()


    @property
    def scale(self):
        return self.__scale


    def add_mesh(self, atm_range=0.01, dims = [50,100,50], mesh_part='quarter',compute=True):
        super(DiffrotStar,self).add_mesh(atm_range=atm_range, dims=dims, mesh_part=mesh_part, bs=self._structure._bs, pot=self._pot,compute=compute)


    def add_structure(self, compute=False, **kwargs):

        new_structure = kwargs.get('structure', self.params['structure'])

        if new_structure in implemented_structures:

            if new_structure != self.params['structure']:
                self.params['structure'] = new_structure

            get_structure = getattr(structure, '%s' % new_structure)

            if hasattr(get_structure, '%s%s' % (self.params['objtype'].title().replace("_",""),new_structure.title().rstrip('s'))):
                get_structure_object = getattr(get_structure, '%s%s' % (self.params['objtype'].title().replace("_",""),new_structure.title().rstrip('s')))
                # if self._structure != None:
                #     logging.info('Structure object exists: rewriting not enabled, will keep initial configuration.')
                # else:
                self._structure = get_structure_object(**kwargs)
                if compute:
                    self._structure.compute_structure(self.mesh, self.directory)
                    
        else:
            raise ValueError('Structure %s not implemented, must be one of %s' % (new_structure, implemented_structures))
    

        