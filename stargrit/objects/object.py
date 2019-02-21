import os
import shutil
from stargrit import structure, geometry, atmosphere, radiative_transfer
from stargrit.structure.polytropes import DiffrotPolytrope, RochePolytrope
from stargrit.atmosphere.blackbody import GrayBlackbody, MonochromaticBlackbody


class Object(object):


    def __init__(self, object_type = 'contact_binary', geometry = 'spherical', 
                structure='polytropes:tidal', atmosphere='blackbody:gray',**kwargs):

        """
        The Object class initializes a star or contact binary object.

        The object is initialized based on the values of the input parameters,
        and checks are performed for compatibility and support upon initialization
        (to avoid checking and raising exceptions further down the line).

        Parameters
        ----------
        object_type: {'star', 'contact_binary'}
            The type of object to be initialized.
        geometry: {'spherical', 'cylindrical'}
            Geometry the mesh is to be computed in.
        structure: {'polytropes:diffrot', 'polytropes:tidal'} 
            Model to be used for the hydrodynamical structure. 
        atmosphere: {'blackbody:gray', 'blackbody:monochromatic'}
            Initial atmosphere model.
        **kwargs
            Arbitrary keyword arguments compatible with the model.

        Raises
        ------
        ValueError
            If any of the top-level parameters are incompatible or not currently supported.

        """

        self.directory = kwargs.get('directory', os.getcwd()+'/')
        
        self._add_object_type(object_type, **kwargs)
        self._add_mesh(geometry, **kwargs)
        self._add_structure(structure, **kwargs)
        self._add_atmosphere(atmosphere, **kwargs)
        
        

    def _add_object_type(self, object_type, **kwargs):
        """
        Assigns object parameter set if object type is implemented.

        Parameters
        ----------
        object_type: str
            The type of object to be initialized (current options: 'star' or 'contact_binary')

        Raises
        ------
        ValueError
            If input object_type not supported by stargrit.
        """

        if object_type in self.implemented_object_types:
            self.__object_type = object_type
            self._set_object_attributes(**kwargs)

        else:
            raise ValueError('Object type {} not supported by stargrit'.format(object_type))


    def _add_mesh(self, geometry_value, **kwargs):
        """
        Assigns mesh parameterset if geometry compatible with object type.

        Parameters
        ----------
        geometry: str
            Geometry of the mesh (current options: 'spherical' or 'cylindrical').
    
        Raises
        ------
        ValueError
            If geometry not supported or incompatible with object type.
        """

        if geometry_value in self.implemented_geometries:
            geometry_module = getattr(geometry, geometry_value)
            geometry_class = getattr(geometry_module, 
                                '%s%s%s' % (self.object_type.title().replace("_",""), 
                                geometry_value.title(), 'Mesh'))
            self.__mesh = geometry_class(self, **kwargs)
        else:
            raise ValueError('Geometry {} incompatible with object type {}'.format(geometry, self.object_type))


    @property
    def mesh(self):
        return self.__mesh

    
    @mesh.setter
    def mesh(self, value):
        self._add_mesh(value)

    
    def _compute_mesh(self, **kwargs):
        if self.mesh == None:
            raise ReferenceError('No geometry attribute detected. Add first with .add_geometry().')
        else:
            self.mesh._compute(**kwargs)


    def _add_structure(self, structure_value, **kwargs):
        """
        Assigns structure parameterset if supported and compatible with object type.

        Parameters
        ----------
        structure: str
            Hydrodynamical model of the structure.
            Always provided in the format 'module:submodule'.
            (current options: 'polytropes:diffrot' and 'polytropes:tidal')

        Raises
        ------
        ValueError
            If structure not supported or incompatible with object type.
        """

        if structure_value in self.implemented_structures:
            structure_levels = structure_value.split(':')
            structure_module = getattr(structure, structure_levels[0])
            structure_class = getattr(structure_module, 
                                '%s%s' % (structure_levels[1].title(),
                                structure_levels[0].title().rstrip('s')))
            self.__structure = structure_class(self, **kwargs)

        else:
            raise ValueError('Structure {} incompatible with object type {}'.format(structure, self.object_type))


    @property
    def structure(self):
        return self.__structure

    
    @structure.setter
    def structure(self, value):
        self._add_structure(value)

    
    def _compute_structure(self, **kwargs):
        if self.structure == None:
            raise ReferenceError('No structure attribute detected. Add first with .add_structure().')
        else:
            self.structure._compute(**kwargs)


    def _add_atmosphere(self, atmosphere_value, **kwargs):
        """
        Assigns atmosphere parameterset if supported and compatible with object type.

        Parameters
        ----------
        atmosphere: str
            Atmosphere model.
            Always provided in the format 'module:submodule'.
            (current options: 'blackbody:gray' and 'blackbody:monochromatic')

        Raises
        ------
        ValueError
            If atmosphere not supported or incompatible with object type.
        """

        if atmosphere_value in self.implemented_atmospheres:
            atmosphere_levels = atmosphere_value.split(':')
            atmosphere_module = getattr(atmosphere, atmosphere_levels[0])
            atmosphere_class = getattr(atmosphere_module, 
                                '%s%s' % (atmosphere_levels[1].title(),
                                atmosphere_levels[0].title().rstrip('s')))
            self.__atmosphere = atmosphere_class(self, **kwargs)

        else:
            raise ValueError('atmosphere {} incompatible with object type {}'.format(atmosphere, self.object_type))

    @property
    def atmosphere(self):
        return self.__atmosphere

    
    @atmosphere.setter
    def atmosphere(self, value):
        self._add_atmosphere(value)

    
    def _compute_atmosphere(self, **kwargs):
        if self.atmosphere == None:
            raise ReferenceError('No atmosphere attribute detected. Add first with .add_atmosphere().')
        else:
            self.atmosphere._compute(**kwargs)


    def _add_radiative_transfer(self, rt_method='cobain', **kwargs):

        if rt_method in self.implemented_rt_methods:
            rt_module = getattr(radiative_transfer, rt_method)

            # this needs to be better handled in the future
            if isinstance(self.structure, DiffrotPolytrope):
                obj = 'DiffrotStar'
            elif isinstance(self.structure, RochePolytrope):
                obj = 'ContactBinary'
            else:
                raise ValueError

            if isinstance(self.atmosphere, GrayBlackbody):
                atm = 'Gray'
            elif isinstance(self.atmosphere, MonochromaticBlackbody):
                atm = 'Monochromatic'
            else:
                raise ValueError

            rt_class = getattr(rt_module, '%s%s%s' % (obj, atm, 'RadiativeTransfer'))
            self.__rt = rt_class(self, **kwargs)


    def _add_rt(self, rt_method, **kwargs):
        self._add_radiative_transfer(rt_method, **kwargs)


    @property
    def rt(self):
        return self.__rt

    
    @rt.setter
    def rt(self, value):
        self._add_radiative_transfer(value)


    def _set_object_attributes(self, object_type, **kwargs):
        """
        Adds parameters associated with object type.

        Handled by subclass.
        """
        raise NotImplementedError


    def _compute_initial(self,**kwargs):

        """
        Computes the intial state of the object.

        After completion, the mesh, structure and intitial atmosphere arrays
        will be computed and attached to their parent objects or saved as files.
        """

        # some attributes are shared between classes so the kwargs need to be
        # properly distributed here, to avoid unnecessary recomputing

        mesh_params = set(kwargs.keys()) & set(self.mesh._params())
        structure_params = (set(kwargs.keys()) & set(self.structure._params())) - mesh_params
        atmosphere_params = (set(kwargs.keys()) & set(self.atmosphere._params())) - mesh_params - structure_params

        kwargs_mesh = {}
        kwargs_structure = {}
        kwargs_atmosphere = {}

        for param in mesh_params:
            kwargs_mesh[param] = kwargs[param]

        for param in structure_params:
            kwargs_structure[param] = kwargs[param]

        for param in atmosphere_params:
            kwargs_atmosphere[param] = kwargs[param]

        self.mesh._compute(**kwargs_mesh)
        self.structure._compute(**kwargs_structure)
        self.atmosphere._compute(**kwargs_atmosphere)


    @property
    def implemented_object_types(self):
        """
        Returns implemented object types.
        """
        return ['star', 'contact_binary']


    @property
    def implemented_geometries(self):
        """
        Returns implemented geometries associated with the object.

        Overriden by subclass.
        """
        return ['spherical', 'cylindrical']


    @property
    def implemented_structures(self):
        """
        Returns implemented structure models associated with the object.
        
        Overriden by subclass.
        """
        return ['polytropes:diffrot', 'polytropes:tidal']

    
    @property
    def implemented_atmospheres(self):
        """
        Returns implemented structure models associated with the object.
        
        Overriden by subclass.
        """
        return ['blackbody:gray', 'blackbody:monochromatic']

    
    @property
    def implemented_rt_methods(self):
        """
        Returns implemented radiative transfer models associated with the object.
        """
        return ['cobain']



    @property
    def object_type(self):
        """
        Object type to be initialized. Can be 'star' or 'contact_binary'.
        """
        return self.__object_type

    @object_type.setter
    def object_type(self):
        return None


    @property
    def directory(self):
        """
        Directory where all the structure and RT files are saved.
        """
        return self.__directory


    @directory.setter
    def directory(self, newdir, move = True):
        # create new directory if non-existent
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        if move:
            # move all files in existing directory to new one
            try:
                files = os.listdir(self.directory)
                for f in files:
                    shutil.move(self.directory+f, newdir)
            except:
                pass
        self.__directory = newdir


    def _save(self,filename):
        
        import pickle

        f = file(self.directory+filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def _load(filename):
        
        import pickle

        with file(filename, 'rb') as f:
            return pickle.load(f)

