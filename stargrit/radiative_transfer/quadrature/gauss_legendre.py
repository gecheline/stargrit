import numpy as np 

class Gauss_Legendre(object):

    def __init__(self, ntheta=10, nphi=20):

        # compute thetas from Gauss-Legendre quadrature
        (x, weights) = np.polynomial.legendre.leggauss(ntheta)
        thetas = np.arccos(x)
        phis = np.linspace(0,2*np.pi,nphi)

        theta, phi = np.meshgrid(thetas, phis, indexing='ij')

        points = np.zeros((ntheta,nphi,3))
        points[:,:,0] = np.sin(theta)*np.cos(phi)
        points[:,:,1] = np.sin(theta)*np.sin(phi)
        points[:,:,2] = np.cos(theta)

        self.azimuthal_polar = np.array([theta.flatten(), phi.flatten()]).T
        self.points = points.reshape((ntheta*nphi,3))
        self.weights = weights/np.sin(thetas)
        self.thetas = thetas
        self.phis = phis

        # the Gaussian quadrature doesn't encompass theta=0 and theta=pi, 
        # so the outward normal and inward normal directions are left out.
        # We're artificially adding them here but will skip them when integrating
        # by setting their weights to 0

        normal_dirs = np.array([[0.,0.,1.],[0.,0.,-1]])
        az_pol_norm = np.array([[0.,0.],[0.,np.pi]])

        self.points = np.vstack((normal_dirs,self.points))
        self.azimuthal_polar = np.vstack((az_pol_norm, self.azimuthal_polar))
        self.nI = len(self.points)


    def integrate_over_4pi(self, function):
        
        ntheta, nphi = len(self.thetas), len(self.phis)
        # check that function is of the right shape
        if function.shape == (ntheta, nphi):
            function = function.T 
        elif function.shape == (nphi, ntheta):
            function = function
        else:
            raise ValueError('Shape mismatch: function needs to have shape: (%s,%s)' % (ntheta,nphi))

        f_theta = np.zeros(nphi)
        for i in range(nphi):
            f_theta[i] = np.sum(function[i]*self.weights)

        return 1./(4*np.pi)*np.trapz(f_theta*np.sin(self.phis),self.phis)


    def integrate_outer_m_inner(self, function):

        ntheta, nphi = len(self.thetas), len(self.phis)
        # check that function is of the right shape
        if function.shape == (ntheta, nphi):
            function = function.T 
        elif function.shape == (nphi, ntheta):
            function = function
        else:
            raise ValueError('Shape mismatch: function needs to have shape: (%s,%s)' % (ntheta,nphi))

        nphi = len(self.phis)
        cond_out = self.thetas <= np.pi/2
        cond_in = self.thetas > np.pi/2

        f_theta_out = np.zeros(nphi)
        for i in range(nphi):
            f_theta_out[i] = np.sum(function[i][cond_out]*self.weights[cond_out]*np.cos(self.thetas[cond_out]))

        f_theta_in = np.zeros(nphi)
        for i in range(nphi):
            f_theta_in[i] = np.sum(function[i][cond_in]*self.weights[cond_in]*np.cos(self.thetas[cond_in]))

        return np.abs(np.trapz(f_theta_out*np.sin(self.phis),self.phis))-np.abs(np.trapz(f_theta_in*np.sin(self.phis),self.phis))


        