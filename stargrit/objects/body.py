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

class Body(object):

    '''
    The Body class holds general methods for Star and Contact Binary.
    Upon creation, checks are performed to ensure that the chosen methods are implemented.
    '''

    def __init__(self, objtype='contact_binary', geometry='cylindrical', structure='polytropes', atmosphere='blackbody', rt_method='cobain', directory=os.getcwd()+'/'):
        
        self.params = {}
        # validate whether top-level parameters within implemented and initialize
        if objtype in implemented_object_types:
            self.params['objtype'] = objtype
        else:
            raise ValueError('Objtype \'%s\' not implemented, must be one of %r' % (objtype, implemented_object_types))

        if geometry in implemented_geometries:
            self._mesh = None
            self.params['geometry'] = geometry
        else:
            raise ValueError('%s geometry not implemented, must be one of %r' % (geometry, implemented_geometries))
        
        if structure in implemented_structures:
            self._structure = None
            self.params['structure'] = structure
        else:
            raise ValueError('Structure \'%s\' not implemented, must be one of %r' % (structure, implemented_structures))
        
        if atmosphere in implemented_atmospheres:
            self._atmosphere = None
            self.params['atmosphere'] = atmosphere
        else:
            raise ValueError('Atmosphere \'%s\' not implemented, must be one of %r' % (atmosphere, implemented_atmospheres))

        if rt_method in implemented_rt_methods:
            self._rt = None
            self.params['rt_method'] = rt_method
        else:
            raise ValueError('Atmosphere \'%s\' not implemented, must be one of %r' % (atmosphere, implemented_atmospheres))
        
        # set the directory where all of the structure files will be stored
        self.__directory = directory
        # self.pickle(self.directory + 'body')


    def add_mesh(self, atm_range=0.01, dims=[50,50,50], mesh_part='quarter', compute=True, **kwargs):
        # check for everything else before setting
        # does a mesh exist already: can't change geometry
        # if just parametersets, change the PS values (geometry must be spherical, options for structure, etc.)
        # self._geometry = value

        new_geometry = kwargs.get('geometry', self.params['geometry'])
        if new_geometry != self.params['geometry']:
            self.params['geometry'] = new_geometry
        if new_geometry in implemented_geometries:
            if hasattr(sg.geometry, '%s' % new_geometry):
                get_geometry = getattr(geometry, '%s' % new_geometry)
                if hasattr(get_geometry, '%s%s' % (self.params['objtype'].title().replace("_",""),'Mesh')):
                    get_geometry_object = getattr(get_geometry, '%s%s' % (self.params['objtype'].title().replace("_",""), 'Mesh'))
                    self._mesh = get_geometry_object(atm_range=atm_range, dims=dims, mesh_part=mesh_part, **kwargs)
                    if compute:
                        print 'Computing mesh'
                        self._mesh.compute_mesh()
        else:
            raise ValueError('%s geometry not implemented, must be one of %s' % (new_geometry, implemented_geometries))
    

    def compute_mesh(self):
        self._mesh.compute_mesh()


    def add_structure(self, compute=False, **kwargs):
        if self._mesh == None:
            raise Exception('A mesh required before adding structure. Add and compute with add_mesh, compute_mesh.')
        else:
            new_structure = kwargs.get('structure', self.params['structure'])
            if new_structure in implemented_structures:
                if new_structure != self.params['structure']:
                    self.params['structure'] = new_structure
                get_structure = getattr(structure, '%s' % new_structure)
                if hasattr(get_structure, '%s%s' % (self.params['objtype'].title().replace("_",""),new_structure.title().rstrip('s'))):
                    get_structure_object = getattr(get_structure, '%s%s' % (self.params['objtype'].title().replace("_",""),new_structure.title().rstrip('s')))
                    if self._structure != None:
                        logging.info('Structure object exists: rewriting not enabled, will keep initial configuration.')
                    else:
                        self._structure = get_structure_object(**kwargs)
                    if compute:
                        self._structure.compute_structure(self._mesh, self.directory)
            else:
                raise ValueError('Structure %s not implemented, must be one of %s' % (new_structure, implemented_structures))


    def compute_structure(self):
        self._structure.compute_structure(self._mesh, self.__directory)


    def add_atmosphere(self, compute=False, **kwargs):
        
        if self._mesh == None or self._structure == None:
            raise Exception('A mesh and structure required before adding atmosphere. \
            Add and compute with add_mesh, compute_mesh, add_structure, compute_structure.')
        else:
            new_atmosphere = kwargs.get('atmosphere', self.params['atmosphere'])
            if new_atmosphere in implemented_atmospheres:
                if new_atmosphere != self.params['atmosphere']:
                    self.params['atmosphere'] = new_atmosphere
                get_atmosphere = getattr(atmosphere, '%s' % new_atmosphere)
                if hasattr(get_atmosphere, '%s%s' % (self.params['objtype'].title().replace("_",""),'Atmosphere')):
                    get_atmosphere_object = getattr(get_atmosphere, '%s%s' % (self.params['objtype'].title().replace("_",""), 'Atmosphere'))
                    if self._atmosphere != None:
                        logging.info('Atmosphere object exists: rewriting not enabled, will keep initial configuration.')
                    else:
                        self._atmosphere = get_atmosphere_object(self._mesh, self.__directory, **kwargs)
                        if compute==True:
                            self._atmosphere.compute_atmosphere()


    def compute_atmosphere(self):
        self._atmosphere.compute_atmosphere()


    def add_radiative_transfer(self, compute=False, **kwargs):
        if self._mesh == None or self._structure == None or self._atmosphere == None:
            raise Exception('A mesh, structure and atmosphere required before adding RT object. \
            Add and compute with add_mesh, compute_mesh, add_structure, compute_structure, add_atmosphere.')
        else:
            new_rt_method = kwargs.get('rt_method', self.params['rt_method'])
            if new_rt_method in implemented_rt_methods:
                if new_rt_method != self.params['rt_method']:
                    self.params['rt_method'] = new_rt_method
                get_rt = getattr(radiative_transfer, '%s' % new_rt_method)
                if hasattr(get_rt, '%s%s' % (self.params['objtype'].title().replace("_",""),'RadiativeTransfer')):
                    get_rt_object = getattr(get_rt, '%s%s%s' % (self.params['objtype'].title().replace("_",""),self._atmosphere._atm_type.title(), 'RadiativeTransfer'))
                    if self._rt != None:
                        logging.info('RT object exists: rewriting not enabled, will keep initial configuration.')
                    else:
                        self._rt = get_rt_object(self._atmosphere, **kwargs)
                        if compute==True:
                            raise NotImplementedError


    def compute_radiative_transfer(self, parallel=True):
        # all the paralellization logic will go here
        raise NotImplementedError


    def add_rt(self, compute=False):
        self.add_radiative_transfer(compute=compute)


    def compute_rt(self, parallel=True):
        self.compute_radiative_transfer(parallel=parallel)

        
    @property
    def directory(self):
        return self.__directory


    @directory.setter
    def directory(self, newdir, move = True):
        # create new directory if non-existent
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        if move:
            # move all files in existing directory to new one
            try:
                files = os.listdir(self.__directory)
                for f in files:
                    shutil.move(self.__directory+f, newdir)
            except:
                pass
        self.__directory = newdir

    

    @property
    def mesh(self):
        return self._mesh


    @property
    def structure(self):
        return self._structure


    @property
    def atmosphere(self):
        return self._atmosphere


    @property
    def rt(self):
        return self._rt
