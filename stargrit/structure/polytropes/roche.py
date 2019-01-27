from stargrit.structure.polytropes.spherical import Polytrope
import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import astropy.constants as const 
import astropy.units as u


class RochePolytrope(object):

    def __init__(self):
        return None