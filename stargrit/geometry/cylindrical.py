import numpy as np
import logging

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class CylindricalMesh(object):

    def __init__(self, **kwargs):
        super(CylindricalMesh,self).__init__(geometry='cylindrical', **kwargs)

    def set_geometry_attributes(self, **kwargs):
        raise NotImplementedError

    @property 
    def default_units(self):
        return {'rs': u.R_sun, 'normals': u.dimensionless}


class ContactBinaryCylindricalMesh(CylindricalMesh):
    # TODO: call the relevant potential and radius computation 
    # from potentials instead of implementing here

    def __init__(self, **kwargs):
        super(ContactBinaryCylindricalMesh,self).__init__(geometry='cylindrical', **kwargs)

    def set_geometry_attributes(self, **kwargs):
        # TODO: final level, this is where it should be fully implemented
        # some shared attributes can be set in super
        raise NotImplementedError