import numpy as np 
import quadpy 

class Lebedev(object):

    def __init__(self, deg='15'):

        quad = quadpy.sphere.Lebedev(deg)

        self.azimuthal_polar = quad.azimuthal_polar
        self.points = quad.points
        self.weights = quad.weights 
        self.nI = len(self.weights)


    def integrate_over_4pi(self, function):
        if function.shape == self.weights.shape:
            return np.sum(function*self.weights)
        else:
            raise ValueError('Shape mismatch: function needs to have shape %s' % self.weights.shape)


    def integrate_outer_m_inner(self, function):
        
        cond_out = self.azimuthal_polar[:,1] <= np.pi/2
        cond_in = self.azimuthal_polar[:,1] > np.pi/2

        ws = self.weights
        thetas = self.azimuthal_polar[:,1]

        return 2*np.pi*(np.sum(ws[cond_out]*function[cond_out]*np.cos(thetas[cond_out])) - np.sum(ws[cond_in]*function[cond_in]*np.cos(thetas[cond_in])))
