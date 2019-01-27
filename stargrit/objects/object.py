import os
import shutil
from stargrit import structure, geometry, atmosphere


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
        
        self.add_object_type(object_type, **kwargs)
        self.add_mesh(geometry, **kwargs)
        self.add_structure(structure, **kwargs)
        
        

    def add_object_type(self, object_type, **kwargs):
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

        if object_type in self.implemented_object_types():
            self.__object_type = object_type
            self.set_object_attributes(**kwargs)

        else:
            raise ValueError('Object type {} not supported by stargrit'.format(object_type))


    def add_mesh(self, geometry_value, **kwargs):
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

        if geometry_value in self.implemented_geometries():
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
    def mesh(self, value, **kwargs):
        self.add_mesh(value, **kwargs)

    
    def _compute_mesh(self, **kwargs):
        if self.mesh == None:
            raise ReferenceError('No geometry object detected. Add first with .add_geometry().')
        else:
            self.mesh._compute(**kwargs)


    def add_structure(self, structure_value, **kwargs):
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

        if structure_value in self.implemented_structures():
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
    def structure(self, value, **kwargs):
        self.add_structure(value, **kwargs)

    
    def compute_structure(self, **kwargs):
        if self.structure == None:
            raise ReferenceError('No structure object detected. Add first with .add_structure().')
        else:
            self.structure._compute(**kwargs)


    def set_object_attributes(self, object_type, **kwargs):
        """
        Adds parameters associated with object type.

        Handled by subclass.
        """
        raise NotImplementedError


    def implemented_object_types(self):
        """
        Returns implemented object types.
        """
        return ['star', 'contact_binary']


    def implemented_geometries(self):
        """
        Returns implemented geometries associated with the object.

        Overriden by subclass.
        """
        return ['spherical', 'cylindrical']


    def implemented_structures(self):
        """
        Returns implemented structure models associated with the object.
        
        Overriden by subclass.
        """
        return ['polytropes:diffrot', 'polytropes:tidal']


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