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

class ContactBinary(Body):


    def __init__(self, geometry='cylindrical', structure='polytropes', atmosphere='blackbody', directory=os.getcwd()+'/', **kwargs):

        super(ContactBinary, self).__init__(objtype='contact_binary', geometry=geometry, structure=structure,atmosphere=atmosphere,
                                directory=directory)

        # check which parameters are provided by user and use defaults if not provided
        self._q = kwargs.get('q', 1.0)
        self._teff = kwargs.get('teff', 5777.0)*u.K
        self._mass1 = kwargs.get('mass1', 1.0)*u.M_sun

        # check if potential or ff provided
        if 'ff' in kwargs.keys() and 'pot' not in kwargs.keys():
            ff = kwargs['ff']
            pot = sg.potentials.roche.ff_to_pot(ff,self._q)
        elif 'pot' in kwargs.keys() and 'ff' not in kwargs.keys():
            pot = kwargs['pot']
            ff = sg.potentials.roche.pot_to_ff(pot,self._q)
        elif 'ff' in kwargs.keys() and 'pot' in kwargs.keys():
            if kwargs['ff'] == sg.potentials.roche.pot_to_ff(kwargs['pot'],self._q):
                pot = kwargs['pot']
                ff = kwargs['ff']
            else:
                raise ValueError('Value mismatch for pot and ff, provide only one.')
        else:
            ff = 0.5
            pot = sg.potentials.roche.ff_to_pot(ff,self._q)

        self._pot = pot
        self._ff = ff


    def add_mesh(self, atm_range=0.01, dims = [50,100,50], mesh_part='quarter',compute=True):
        super(ContactBinary, self).add_mesh(atm_range=atm_range, dims=dims, mesh_part=mesh_part, q=self._q, pot=self._pot, compute=compute)


    def add_structure(self, **kwargs):
        super(ContactBinary, self).add_structure(pot=self._pot, q=self._q, **kwargs)