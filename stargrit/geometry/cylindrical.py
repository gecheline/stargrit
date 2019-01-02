import numpy as np 
import os, shutil
import logging
import scipy.interpolate as spint
from scipy.optimize import newton
from stargrit import potentials

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class ContactBinaryMesh(object):

    def __init__(self, dims=[50,100,50], atm_range=0.01, mesh_part='quadratic', **kwargs):#dims=[50,100,50], atm_range=0.01, mesh_part='quadratic', pot=3.75, q=1.0):

        self._dims = dims
        self._atm_range = atm_range
        self._mesh_part = mesh_part
        self._q = kwargs.get('q', 1.)
        self._pot = kwargs.get('pot', 3.75)
        self._nekmin, _ = potentials.roche.nekmin(self._pot,self._q)
        self._geometry = 'cylindrical'

        theta_end = np.pi if self._mesh_part == 'half' else (2*np.pi if self._mesh_part == 'full' else np.pi/2)

        pots = self._pot*np.linspace(1.,1.+self._atm_range,self._dims[0])
        xs_norm = np.linspace(0.,1.,self._dims[1])
        thetas = np.linspace(0.,theta_end, self._dims[2])

        self.coords = {'pots': pots, 'xs_norm': xs_norm, 'thetas': thetas}

    @staticmethod
    def xnorm_to_xphys_cb(xnorm,xRB1,xRB2):

        # converts normalized to Roche x-values
        return xnorm*(xRB2-xRB1)+xRB1

    @staticmethod
    def xphys_to_xnorm_cb(xphys,xRB1,xRB2):

        # converts Roche to normalized x-values
        return (xphys-xRB1)/(xRB2-xRB1)

    @staticmethod
    def get_rbacks(pots,q,comp):

        # find the value of the radius at the back of a star by interpolating in ff and q

        dir_local = os.path.dirname(os.path.abspath(__file__)) + '/'
        if comp == 1:
            rbacks = np.load(dir_local+'tables/rbacks_forinterp_negff_prim.npy')
        elif comp==2:
            rbacks = np.load(dir_local+'tables/rbacks_forinterp_negff_sec.npy')
        else:
            raise ValueError
        qs = rbacks[0].copy()[1:]
        # print qs.min(), qs.max()
        ffs = rbacks[:, 0].copy()[1:]
        rs_rowr = np.delete(rbacks, [0], axis=0)
        rs = np.delete(rs_rowr, [0], axis=1)

        RGI = spint.RegularGridInterpolator
        f = RGI(points=[qs, ffs], values=rs)
        qs_int = q*np.ones(len(pots))
        # print qs_int
        # print qs_int.min(), qs_int.max()
        ffs_int = potentials.roche.pot_to_ff(pots,q)
        # print ffs_int
        # print pots.min(), pots.max()
        # print ffs_int.min(), ffs_int.max()
        rs_int = f((qs_int, ffs_int))
        return rs_int

    @staticmethod
    def find_root(coeffs):

        # finds the one root of a polynomial that is physical

        roots = np.roots(coeffs)
        reals_i = roots[np.isreal(roots)]
        # print reals_i

        if reals_i.size != 0:
            reals = np.real(reals_i)
            # print reals

            if np.all(reals >= 0.):
                # print 'all positive', reals
                return np.min(reals)
            elif np.all(reals < 0.):
                return np.nan
            else:
                reals_pos = reals[reals >= 0.]
                if reals_pos.size > 1:
                    return np.max(reals_pos)
                else:
                    return reals_pos[0]
        else:
            return np.nan

    def find_r0(self,x,q,pot):

        # find the local polar radius, to be used as intial value for a given x

        P = ( pot - 0.5 * ( 1. + q ) * x ** 2. + q * x ) ** 2.
        A = P * ( 2. * x ** 2. - 2. * x + 1. ) - ( 1. + q ** 2.)
        B = ( x - 1. ) ** 2. * ( P * x ** 2. - 1. ) - x ** 2. * q ** 2.
        C = 4. * q ** 2.
        D = C * ( 2. * x ** 2. - 2. * x + 1. )
        E = C * x ** 2. * ( x - 1. ) ** 2.

        u = 2. * A / P
        v = ( 2. * B * P + A ** 2 - C ) / P ** 2.
        w = ( 2. * A * B - D ) / P ** 2.
        t = ( B ** 2. - E ) / P ** 2.

        # find the roots of the polynomial r^8 + u r^6 + v r^4 + w r^2 + t
        root = self.find_root((1,0,u,0,v,0,w,0,t))

        return root

    @staticmethod
    def radius_newton(r0, x, theta, pot, q):

        def Roche(r,x=x, theta=theta, q=q):
            return pot - potentials.roche.BinaryRoche_cylindrical(r,x=x,theta=theta,q=q)

        try:
            return newton(Roche, r0, args=(x,theta,q), maxiter=100000, tol=1e-8)

        except:
            return np.nan

    def compute_mesh(self,parallel=False,**kwargs):
        
        if parallel:
            # paralellize mesh creation
            raise NotImplementedError

        else:
            # compute mesh as usual
            crit_pot = potentials.roche.critical_pots(self._q)['pot_L1']

            if self.coords['pots'][-1] > crit_pot:
                raise ValueError('Boundary potential cannot be higher than critical value %s.' % crit_pot)
            
            else:

                pots2 = self.coords['pots'] / self._q + 0.5 * (self._q - 1) / self._q
                
                rs = np.zeros((self._dims[0]*self._dims[1]*self._dims[2],3))
                normals = np.zeros((self._dims[0]*self._dims[1]*self._dims[2], 3))
                n = 0

                xrb1s = - self.get_rbacks(self.coords['pots'], self._q, 1)
                xrb2s = np.ones(len(self.coords['pots'])) + self.get_rbacks(self.coords['pots'], self._q, 2)

                for i,pot in enumerate(self.coords['pots']):

                    for j,xnorm in enumerate(self.coords['xs_norm']):

                        # convert the normalized x to physical
                        xphys = self.xnorm_to_xphys_cb(xnorm, xrb1s[i], xrb2s[i])
                        if xnorm == 0. or xnorm==1.:
                            r0 = 0.
                        else:
                            r0 = self.find_r0(xphys, self._q, pot)

                        for k,theta in enumerate(self.coords['thetas']):
                            print 'pot %s, xnorm %s, theta %s' % (i, j, k)
                            # find the true radius for the given theta
                            if xnorm == 0. or xnorm == 1.:
                                rho = 0.
                            else:
                                rho = self.radius_newton(r0, xphys, theta, pot, self._q)

                            # print 'rho at pot=%s, x=%s, theta=%s: %s' % (pot,xnorm,theta,rho)
                            if np.isnan(rho):
                                # now only happens at points where r0 = nan because it's probably = 0.
                                rs[n] = np.array([xphys, 0., 0.])
                                # print 'nan, r0', r0
                            else:
                                rs[n] = np.array([xphys, rho * np.sin(theta), rho * np.cos(theta)])

                            nx = -potentials.dBinaryRochedx(rs[n], 1., self._q, 1.)
                            ny = -potentials.dBinaryRochedy(rs[n], 1., self._q, 1.)
                            nz = -potentials.dBinaryRochedz(rs[n], 1., self._q, 1.)
                            nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                            normals[n] = np.array([nx / nn, ny / nn, nz / nn])

                            n+=1

                self.rs = rs
                self.ns = normals