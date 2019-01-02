import numpy as np 
import os, shutil
import logging
import scipy.interpolate as spint
from stargrit import potentials

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

class SphericalMesh(object):

    def __init__(self, dims=[50,100,50], atm_range=0.01, mesh_part='quadratic', pot=3.74):
        self._geometry = 'spherical'
        self._dims = dims
        self._atm_range = atm_range
        self._mesh_part = mesh_part
        self._pot = pot

        theta_end = np.pi if self._mesh_part == 'half' else (2*np.pi if self._mesh_part == 'full' else np.pi/2)
        phi_end = np.pi if (self._mesh_part == 'half' or self._mesh_part == 'quarter') else 2*np.pi

        pots = self._pot*np.linspace(1.,1.+self._atm_range,self._dims[0])
        thetas = np.linspace(0., theta_end, self._dims[1])
        phis = np.linspace(0., phi_end, self._dims[2])

        self.coords = {'pots': pots, 'thetas': thetas, 'phis': phis}


class StarMesh(SphericalMesh):

    def __init__(self, dims=[50,100,50], atm_range=0.01, mesh_part='quadratic', **kwargs):
        self._bs = kwargs.pop('bs')
        pot = kwargs.pop('pot')
        # self.__scale = scale
        super(StarMesh,self).__init__(dims=dims, atm_range=atm_range, mesh_part=mesh_part, pot=pot)
        

    def compute_mesh(self):
        meshsize = self._dims[0]*self._dims[1]*self._dims[2]
        rs = np.zeros((meshsize, 3))
        normals = np.zeros((meshsize, 3))

        n = 0
        for i, pot in enumerate(self.coords['pots']):
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(self.coords['thetas']):
                for k, phi in enumerate(self.coords['phis']):
                    r = potentials.diffrot.radius_newton(pot, self._bs, theta)
                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

                    # compute the normal to the surface

                    nx = potentials.dDiffRotRochedx(r_cs, self._bs)
                    ny = potentials.dDiffRotRochedy(r_cs, self._bs)
                    nz = potentials.dDiffRotRochedz(r_cs, self._bs)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs[n] = r_cs
                    normals[n] = np.array([nx / nn, ny / nn, nz / nn])

        self.rs = rs#*self.__scale
        self.normals = normals


class ContactBinaryMesh(SphericalMesh):

    def __init__(self,dims=[50,100,50], atm_range=0.01, mesh_part='quadratic',**kwargs):#dims=[50,100,50], atm_range=0.01, mesh_part='quadratic', q=1.0, pot=3.75):
        
        self._q = kwargs.pop('q')
        pot = kwargs.pop('pot')
        super(ContactBinaryMesh,self).__init__(dims=dims, atm_range=atm_range, mesh_part=mesh_part, pot=pot)
        
        self._nekmin, _ = potentials.roche.nekmin(self._pot,self._q)
        self.__pot2 = self._pot / self._q + 0.5 * (self._q - 1) / self._q
        # self.__scale = scale
        self.__rpole1 = potentials.roche.radius_pot_contact_approx(self._pot, self._q, 0., 1.)
        self.__rpole2 = potentials.roche.radius_pot_contact_approx(self.__pot2, 1./self._q, 0., 1.)

    def build_component(self, pots, rpole, q, nekmin):
        
        meshsize = self._dims[0]*self._dims[1]*self._dims[2]
        rs = np.zeros((meshsize, 3))
        normals = np.zeros((meshsize, 3))

        n=0
        for i, pot in enumerate(pots):
            logging.info('Building equipotential surface at pot=%s' % pot)
            for j, theta in enumerate(self.coords['thetas']):
                for k, phi in enumerate(self.coords['phis']):

                    lam = np.sin(theta) * np.cos(phi)
                    mu = np.sin(theta) * np.sin(phi)
                    nu = np.cos(theta)

                    try:                        
                        r = potentials.roche.radius_newton_spherical(rpole, lam, mu, nu, pot, q, nekmin)
                        # print r
                        if np.isnan(r):
                            r = 0.
                    except:
                        r = 0.

                    r_cs = np.array(
                        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])
                    # compute the normal to the surface

                    nx = -potentials.dBinaryRochedx(r_cs, 1., q, 1.)
                    ny = -potentials.dBinaryRochedy(r_cs, 1., q, 1.)
                    nz = -potentials.dBinaryRochedz(r_cs, 1., q, 1.)
                    nn = np.sqrt(nx * nx + ny * ny + nz * nz)

                    rs[n] = r_cs
                    normals[n] = np.array([nx / nn, ny / nn, nz / nn])
                    n+=1

        return rs, normals

    def compute_mesh_breaks(self):

            potlen = self._dims[1] * self._dims[2]
            dimlen = self._dims[0] * self._dims[1] * self._dims[2]
            breaks1 = np.zeros((self._dims[0], 3))
            breaks2 = np.zeros((self._dims[0], 3))

            for i in range(self._dims[0]):
                pot_slice1 = self.rs[i * potlen: (i + 1) * potlen]
                pot_slice2 = self.rs[dimlen + i * potlen: dimlen + (i + 1) * potlen]

                argzeros1 = np.argwhere(np.all(pot_slice1 == 0., axis=1))
                argzeros2 = np.argwhere(
                    (pot_slice2[:, 0] == 1.0) & (pot_slice2[:, 1] == 0.0) & (pot_slice2[:, 2] == 0.0))

                thetas_break1 = self.coords['thetas'][argzeros1 / self._dims[1]]
                phis_break1 = self.coords['phis'][argzeros1 % self._dims[2]]

                thetas_break2 = self.coords['thetas'][argzeros2 / self._dims[1]]
                phis_break2 = self.coords['phis'][argzeros2 % self._dims[2]]

                if thetas_break1.size == 0:
                    th1min = np.pi / 2
                else:
                    th1min = thetas_break1.min()

                if phis_break1.size == 0:
                    ph1max = 0.
                else:
                    ph1max = phis_break1.max()

                if thetas_break2.size == 0:
                    th2min = np.pi / 2
                else:
                    th2min = thetas_break2.min()

                if phis_break2.size == 0:
                    ph2max = 0.
                else:
                    ph2max = phis_break2.max()

                breaks1[i] = np.array([self.coords['pots'][i], th1min, ph1max])
                breaks2[i] = np.array([self.coords['pots'][i], th2min, ph2max])

            self.breaks1 = breaks1
            self.breaks2 = breaks2

    def compute_mesh(self):

        pots2 = self.coords['pots'] / self._q + 0.5 * (self._q - 1) / self._q

        rs1, normals1 = self.build_component(pots=self.coords['pots'], q=self._q, rpole=self.__rpole1, nekmin=self._nekmin)
        rs2, normals2 = self.build_component(pots = pots2, q=1./self._q, rpole=self.__rpole2, nekmin=1.-self._nekmin)

        rs2[:,0] = 1.-rs2[:,0]
        normals2[:,0] = -normals2[:,0]

        self.rs = np.vstack((rs1, rs2))#*self.__scale
        self.normals = np.vstack((normals1, normals2))
        self.compute_mesh_breaks()

    

        


