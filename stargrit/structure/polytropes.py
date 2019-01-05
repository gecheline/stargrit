import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from stargrit.utils.constants import *
import astropy.units as u
import astropy.constants as const
from stargrit import utils

class Polytrope(object):

    def __init__(self, n=3.0):
        self._n = n

    def lane_emden(self, n=3, dt=1e-4, endt=10., **kwargs):
        '''
        Numerical solution of the Lane-Emden equation for any n, adapted from 
        http://www.astro.utu.fi/~cflynn/Stars/laneem.f
        
        # initial values
        thetas = np.zeros(10000)
        dthetas = np.zeros(10000)
        ts = np.arange(1e-6,endt,dt)

        ibreak = 0
        for i,t in enumerate(ts):
            while thetas[i] >= 0.0:
                if i == 0:
                    thetas[i] = 1.0
                    dthetas[i] = 0.0
                else:
                    dthetas[i] = dthetas[i-1] - (2.0*dthetas[i-1]/ts[i-1] + thetas[i-1]**n)*dt
                    thetas[i] = thetas[i-1]+dthetas[i]*dt
            else:
                ibreak = i
                break

        thetas = thetas[:ibreak]
        dthetas = dthetas[:ibreak]
        ts = ts[:ibreak]

        theta_t_interp = interp1d(thetas, ts)
        t_dtheta_interp = interp1d(ts, dthetas)

        t_surface = float(theta_t_interp(0.))
        dtheta_surface = float(t_dtheta_interp(t_surface))

        return t_surface, dtheta_surface, theta_t_interp
        '''
        def f(y, t):
            return [y[1], -y[0] ** n - 2 * y[1] / t]

        y0 = [1., 0.]
        if n <= 1:
            tmax = 3.5
        elif n <= 2:
            tmax = 5.
        else:
            tmax = 10.
        ts = np.arange(1e-120, tmax, 1e-4)
        soln = odeint(f, y0, ts)

        if np.isnan(soln[:,0]).any():
            nonans = np.argwhere(~np.isnan(soln[:,0])).flatten()
            t_surface = float(ts[nonans[-1]])
            dtheta_surface = float(soln[:,1][nonans[-1]])
            theta_interp = interp1d(ts[:nonans[-1]+1], soln[:,0][:nonans[-1]+1])
        else:
            theta_interp = interp1d(ts, soln[:,0])
            dtheta_interp = interp1d(ts, soln[:,1])

            # compute the value of t and dthetadt where theta falls to zero
            ts_theta_interp = interp1d(soln[:,0], ts)
            t_surface = float(ts_theta_interp(0.))
            dtheta_surface = float(dtheta_interp(t_surface))

        return t_surface, dtheta_surface, theta_interp

    
class DiffrotStarPolytrope(Polytrope):

    def __init__(self, **kwargs):#n=3., bs = [0.,0.,0.], M = 1.0, R = 1.0, teff=5777, XYZ=(0.7381,0.2485,0.0134)):
        
        self._n = kwargs.get('n', 3.)
        self._bs = kwargs.get('bs', [0.,0.,0.])
        self._M = kwargs.get('M', 1.0)*u.M_sun
        self._R = kwargs.get('R', 1.0)*u.R_sun
        self._teff = kwargs.get('teff',5777.0)*u.K

        (X, Y, Z) = kwargs.get('XYZ',(0.7381,0.2485,0.0134))
        self._mass_fractions = (X, Y, Z)
        self._mu = 1./(2.*X + 0.75*Y + 0.5*Z)

        self.compute_lane_emden()


    def le_solution(self, xiu):
        
        bs = self._bs
        n = self._n
        
        def f(y, t):
            def A(t):
                return t ** 2 * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 - 8. / 105. * bs[0] * bs[1] * t ** 8 - (
                    16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 - 196. / 525. * bs[0] ** 2 * bs[
                                    1] * t ** 11 - 128. / 3405. * bs[1] * bs[2] * t ** 12)

            def dA(t):
                return 2 * t * (1 - 1. / 15. * bs[0] ** 2 * t ** 6 - 8. / 105. * bs[0] * bs[1] * t ** 8 - (
                    16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 10 - 196. / 525. * bs[0] ** 2 * bs[
                                    1] * t ** 11 - 128. / 3405. * bs[1] * bs[2] * t ** 12) + t ** 2 * (
                    -6. / 15. * bs[0] ** 2 * t ** 5 - 8. * 8. / 105. * bs[0] * bs[1] * t ** 7 - 10 * (
                        16. / 315. * bs[0] * bs[2] + 8. / 315. * bs[1] ** 2) * t ** 9 - 196. * 11. / 525. * bs[0] ** 2 * bs[
                        1] * t ** 10 - 128. * 12. / 3405. * bs[1] * bs[2] * t ** 11)

            def B(t):
                return 1 + 2 * bs[0] * t ** 3 + 16. / 15. * bs[1] * t ** 5 + 24. / 5. * bs[0] ** 2 * t ** 6 + \
                    16. / 21. * bs[2] * t ** 7 + 44. / 7. * bs[0] * bs[1] * t ** 8 + (1664./315. * bs[0] * bs[2] + 208./105. * bs[1]**2) * t ** 10 + \
                    2912./105. * bs[0]**2 * bs[1] * t ** 11 + 2240./693. * bs[1] * bs[2] * t ** 12

            return [y[1],(-xiu ** 2 * t ** 2 * B(t) * y[0] ** n - dA(t) * y[1])/A(t)]

        # rewrite the diffrot solution to not use odeint but simpler

        y0 = [1., 0.]
        ts = np.arange(1e-120, 8., 1e-4)
        soln = odeint(f, y0, ts)

        # func has the form (xi, theta, dtheta/dxi)
        theta_interp = interp1d(ts, soln[:,0], fill_value='extrapolate')
        dtheta_interp = interp1d(ts, soln[:,1], fill_value='extrapolate')

        negtheta = int(np.argwhere(theta_interp(ts) < 0)[1])
        # compute the value of t and dthetadt where theta falls to zero
        ts_theta_interp = interp1d(soln[:,0][:negtheta], ts[:negtheta])
        t_surface = float(ts_theta_interp(0.))
        dtheta_surface = float(dtheta_interp(t_surface))

        return t_surface, dtheta_surface, np.array([ts, soln[:, 0]])

    def compute_lane_emden(self):
        # R1 is the undistorted radius of the outermost potential surface
        xi_s, dthetadxi_s, theta_interp_xi = self.lane_emden(n=self._n)

        rn = self._R / xi_s
        rhoc = (-1) * self._M / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)
        # Pc = (4 * np.pi * GG_sol * rhoc.value ** 2 * rn ** 2) / (self._n + 1)
        Tc = 1. / ((self._n + 1) * xi_s * ((-1) * dthetadxi_s)) * GG_sol * self._M.value * self._mu * mp_kB_sol / self._R.value * u.K

        # compute the polytrope of the distorted star
        r0_s, dthetadr0_s, theta_interp = self.le_solution(xi_s)

        self._le = {'xi_s': xi_s, 'r0_s': r0_s, 'theta_interp': theta_interp, 'rhoc': rhoc, 'Tc': Tc, 'pot_max': 1e+120}
        
    def compute_surface_pot(self):

        pot = 1./self._le['r0_s']
        r0s = np.linspace(1./pot - 0.1, 1./pot, 10000)
        op_depths = np.zeros(len(r0s))

        theta_interp = interp1d(self._le['theta_interp'][0],self._le['theta_interp'][1])
        for i, r0 in enumerate(r0s):
            T = self._le['Tc'] * theta_interp(r0)
            if T > 0:
                op_depths[i] = ((T / self._teff) ** 4 - 0.5) * 4.0 / 3.0
            else:
                op_depths[i] = 0.0
            # print 'diffrot pot = %.2f T = %.2f, op_depth = %.2f' % (1. / r0, T, op_depths[i])

        r0_intp = interp1d(op_depths, r0s)

        r0_tau1 = r0_intp([2.0 / 3.0])[0]

        return 1./r0_tau1

    def compute_structure(self, mesh, directory, parallel=False):

        theta_interp = interp1d(self._le['theta_interp'][0],self._le['theta_interp'][1], fill_value='extrapolate')
        
        # this logic needs to be replaced based on the point indices
        # for the first iteration when populating with polytropes, all we need is the
        # potential corresponding to the given index for interpolation

        if mesh == None:
            raise ValueError('Mesh object empty, needs to be computed before structure.')
        else:
            points = np.arange(0,len(mesh.rs),1)
            theta_pots = theta_interp(1./mesh.coords['pots'])
            Ts_pots = self._le['Tc'] * theta_pots
            rhos_pots = self._le['rhoc'] * theta_pots ** self._n
            pots_inds = points//(mesh._dims[1]*mesh._dims[2])
            
            np.save(directory+'T_0', Ts_pots[pots_inds].value.reshape(mesh._dims))
            np.save(directory+'rho_0', rhos_pots[pots_inds].value.reshape(mesh._dims))
            self.compute_bb(mesh.coords['pots'], theta_interp, directory)

    def compute_bb(self, pots, theta_interp, directory):
        # make pots that go beyond the mesh for black body interpolation outside of mesh
        pots_bb = np.linspace(pots[0], pots[1]+10, 10000)
        theta_pots_bb = theta_interp(1./pots_bb)
        Ts_bb = self._le['Tc']*pots_bb
        rhos_bb = self._le['rhoc'] * theta_pots_bb ** self._n
        bb_file = np.array([pots_bb, Ts_bb, rhos_bb]).T
        np.save(directory+'potTrho_bb', bb_file)


class TidalStarPolytrope(Polytrope):

    def __init__(self, **kwargs):#n=3.0, M=1.0, q=1.0, pot=3.75, XYZ=(0.7381,0.2485,0.0134), component=1):

        self._component = kwargs.get('component', 1)
        self._n = kwargs.get('n', 3.0)
        self._M = kwargs.get('M', 1.0)*u.M_sun
        self._q = kwargs.get('q', 1.0)
        self._pot = kwargs.get('pot', 3.75)
        zeta, nu = utils.main_sequence.return_MS_factors(self._M.value)
        self._R = self._M.value ** zeta * u.R_sun
        self._teff = self._M.value ** (0.25 * (nu + 1. / (2 * zeta))) * 5777.* u.K
        (X, Y, Z) = kwargs.get('XYZ', (0.7381,0.2485,0.0134))
        self._mass_fractions = (X,Y,Z)
        self._mu = 1./(2.*X + 0.75*Y + 0.5*Z)

        self._k = self.compute_equivalent_radius()
        
        self.compute_lane_emden()

        pot_tau = self.compute_surface_pot()
        # set the scale so that potx_tau = user provided pot
        r0_tau = 1. / (pot_tau - self._q)
        r0_pot = 1. / (self._pot - self._q)

        self.r0_scale = r0_tau / r0_pot  # this scales the structure from polytropes to the outer potential


    def compute_equivalent_radius(self):
        # dimensionless

        n = 0.5 * (self._q + 1.)

        r0 = 1. / (self._pot - self._q)
        return r0 * (1. + 2. * self._n / 3 * r0 ** 3 + (4. / 5. * self._q ** 2 + 
                8. / 15 * self._n * self._q + 76. / 45 * self._n ** 2) * r0 ** 6 + 
                5. / 7 * self._q ** 2 * r0 ** 8 + 2. / 3 * self._q ** 2 * r0 ** 10)


    def compute_surface_pot(self):

        pot = 1./self._le['r0_s'] + self._q
        r0s = np.linspace(1./(pot-self._q) - 0.1, 1./(pot-self._q), 10000)
        op_depths = np.zeros(len(r0s))
        theta_interp = interp1d(self._le['theta_interp'][0], self._le['theta_interp'][1])

        for i, r0 in enumerate(r0s):
            T = self._le['Tc'] * theta_interp(r0)
            if T > 0:
                op_depths[i] = ((T / self._teff) ** 4 - 0.5) * 4.0 / 3.0
            else:
                op_depths[i] = 0.0

            # print 'contact pot = %.2f, r0 = %.2f, T = %.2f, op_depth = %.2f' % (1./r0 + q, r0, T, op_depths[i])

        r0_intp = interp1d(op_depths, r0s)

        r0_tau1 = r0_intp([2.0 / 3.0])[0]

        return 1./r0_tau1 + self._q

    def le_solution(self, xiu):
        n = self._n
        q = self._q
        k = self._k

        def f(y, t):
            def A(t):
                return t ** 2 * (1 + (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (
                    0.5 * (q + 1.)) ** 2) * t ** 6 + 9. / 14. * q ** 2 * t ** 8 + 8. / 9. * q ** 2 * t ** 10)

            def dA(t):
                return 2 * t * (1 + (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (
                    0.5 * (q + 1.)) ** 2) * t ** 6 + 9. / 14. * q ** 2 * t ** 8 + 8. / 9. * q ** 2 * t ** 10) + \
                    t ** 2 * (6 * (2. / 5. * q ** 2 + 4. / 15. * 0.5 * (q + 1) * q + 16. / 15. * (0.5 * (
                        q + 1.)) ** 2) * t ** 5 + 8. * 9. / 14. * q ** 2 * t ** 7 + 10. * 8. / 9. * q ** 2 * t ** 9)

            def B(t):
                return 1 + 4 * (0.5 * (q + 1)) * t ** 3 + (36. / 5. * q ** 2 + 24. / 5. * 0.5 * (
                    q + 1) * q + 96. / 5. * (0.5 * (
                    q + 1)) ** 2) * t ** 6 + 55. / 7 * q ** 2 * t ** 8 + 26. / 3 * q ** 2 * t ** 10

            return [y[1], (-(xiu / k) ** 2 * t ** 2 * B(t) * y[0] ** n - dA(t) * y[1]) / A(t)]

        y0 = [1., 0.]
        ts = np.arange(1e-120, 10., 1e-4)
        soln = odeint(f, y0, ts)

        if np.isnan(soln[:, 0]).any():
            nonans = np.argwhere(~np.isnan(soln[:, 0])).flatten()

            ts_new = np.linspace(ts[nonans[-1]],ts[nonans[-1]+1], 1e4)
            soln_new = odeint(f, y0, ts_new)
            nonans_new = np.argwhere(~np.isnan(soln_new[:, 0])).flatten()

            t_surface = float(ts_new[nonans_new[-1]])
            dtheta_surface = float(soln_new[:, 1][nonans_new[-1]])

            soln[:, 0][np.isnan(soln[:, 0])] = 0.0
            soln_new[:, 0][np.isnan(soln_new[:, 0])] = 0.0

            ts_stack = np.hstack((ts[:nonans[-1]+1], ts_new, ts[nonans[-1]+1:]))
            soln_stack = np.hstack((soln[:,0][:nonans[-1]+1], soln_new[:,0], soln[:,0][nonans[-1]+1:]))
            theta_interp_return = np.array([ts_stack, soln_stack])
        else:

            # func has the form (xi, theta, dtheta/dxi)
            theta_interp = interp1d(ts, soln[:, 0], fill_value='extrapolate')
            dtheta_interp = interp1d(ts, soln[:, 1])

            negtheta = int(np.argwhere(theta_interp(ts) < 0)[1])
            # compute the value of t and dthetadt where theta falls to zero
            ts_theta_interp = interp1d(soln[:, 0][:negtheta], ts[:negtheta])
            t_surface = float(ts_theta_interp(0.))
            dtheta_surface = float(dtheta_interp(t_surface))
            theta_interp_return = np.array([ts, soln[:, 0]])

        return t_surface, dtheta_surface, theta_interp_return


    def compute_lane_emden(self):

        xi_s, dthetadxi_s, theta_interp_xi = self.lane_emden(self._n)

        rn = self._R / xi_s
        rhoc = (-1) * self._M / (4 * np.pi * rn ** 3 * xi_s ** 2 * dthetadxi_s)
        Tc = 1. / ((self._n + 1) * xi_s * ((-1) * dthetadxi_s)) * GG_sol * self._M.value * self._mu * mp_kB_sol / self._R.value * u.K

        # compute the polytrope of the tidally distorted star
        r0_s, dthetadr0_s, theta_interp = self.le_solution(xi_s)
    
        self._le = {'xi_s': xi_s, 'dthetadxi_s': dthetadxi_s, 'r0_s': r0_s, 'theta_interp': theta_interp, 'rhoc': rhoc, 'Tc': Tc}

    def compute_structure_pots(self, pots):
        
        theta_interp = interp1d(self._le['theta_interp'][0],self._le['theta_interp'][1], fill_value='extrapolate')

        theta_pots = theta_interp(self.r0_scale / (pots - self._q))
        Ts_pots = self._le['Tc'] * theta_pots
        rhos_pots = self._le['rhoc'] * theta_pots ** self._n
        
        return Ts_pots, rhos_pots, theta_interp

    def compute_structure(self, rs, pots, dims, directory, parallel=False):

        theta_interp = interp1d(self._le['theta_interp'][0],self._le['theta_interp'][1], fill_value='extrapolate')
        
        # this logic needs to be replaced based on the point indices
        # for the first iteration when populating with polytropes, all we need is the
        # potential corresponding to the given index for interpolation

        points = np.arange(0,len(rs),1)
        theta_pots = theta_interp(self.r0_scale / (pots - self._q))
        Ts_pots = self._le['Tc'] * theta_pots
        rhos_pots = self._le['rhoc'] * theta_pots ** self._n
        pots_inds = points//(dims[1]*dims[2])
        
        np.save(directory+'T%s_0' % int(self._component), Ts_pots[pots_inds].reshape(dims))
        np.save(directory+'rho%s_0' % int(self._component), rhos_pots[pots_inds].reshape(dims))

        self.compute_bb(pots,directory,theta_interp)

    def compute_bb(self, pots, theta_interp, directory):
        # make pots that go beyond the mesh for black body interpolation outside of mesh
        pots_bb = np.linspace(pots[0], pots[1]+10, 10000)
        theta_pots_bb = theta_interp(1./pots_bb)
        Ts_bb = self._le['Tc']*pots_bb
        rhos_bb = self._le['rhoc'] * theta_pots_bb ** self._n
        bb_file = np.array([pots_bb, Ts_bb, rhos_bb]).T
        np.save(directory+'potTrho_bb%s' % self._component, bb_file)

    
class ContactBinaryPolytrope(object):

    def __init__(self, **kwargs):#n1 = 3.0, n2=3.0, mass1=1.0, q=1.0, pot=3.75, XYZ1 = (0.7381,0.2485,0.0134), XYZ2 = (0.7381,0.2485,0.0134)):

        # The contact binary polytrope is composed of two different subclasses (each its own TidalStarPolytrope)
        pot = kwargs.get('pot', 3.75)
        q = kwargs.get('q', 1.0)
        n1 = kwargs.get('n1', 3.0)
        n2 = kwargs.get('n2', 3.0)
        mass1 = kwargs.get('mass1', 1.0)
        XYZ1 = kwargs.get('XYZ1', (0.7381,0.2485,0.0134))
        XYZ2 = kwargs.get('XYZ2', (0.7381,0.2485,0.0134))

        pot2 = pot / q + 0.5 * (q - 1) / q
        self.primary = TidalStarPolytrope(n=n1, M=mass1, q=q, pot=pot, XYZ=XYZ1, component=1)
        self.secondary = TidalStarPolytrope(n=n2, M=q*mass1, q=1./q, pot=pot2, XYZ=XYZ2, component=2)
        # TODO:to avoid having to compute the same thing again, check if secondary==primary and copy if so

    def compute_structure(self, mesh, directory, parallel=False):

        if mesh == None:
            raise ValueError('Mesh object empty, needs to be computed before structure.')
        else:
            pots2 = mesh.coords['pots'] / mesh._q + 0.5 * (mesh._q - 1) / mesh._q
            Ts1_pots, rhos1_pots, theta_interp1 = self.primary.compute_structure_pots(mesh.coords['pots'])
            Ts2_pots, rhos2_pots, theta_interp2 = self.secondary.compute_structure_pots(pots2)

            # compute and save files for blackbody interpolation out of mesh
            self.primary.compute_bb(mesh.coords['pots'],theta_interp1,directory)
            self.secondary.compute_bb(pots2,theta_interp2,directory)
                        
            # compute structure needs to know about the geometry of the mesh to assign values correctly
            complen = np.prod(mesh._dims)
            points = np.arange(0,complen,1)
            pots_inds = points//(mesh._dims[1]*mesh._dims[2])
            
            # populate structure based on geometry and component values per potential
            if mesh._geometry == 'spherical':
                for i,Ts_pots,rhos_pots in zip([1,2],[Ts1_pots,Ts2_pots],[rhos1_pots,rhos2_pots]):
                    np.save(directory+'T%s_0' % i, Ts_pots.value[pots_inds].reshape(mesh._dims))
                    np.save(directory+'rho%s_0' % i, rhos_pots.value[pots_inds].reshape(mesh._dims))

            
            elif mesh._geometry == 'cylindrical':
                # part based on nekmin value
                primcond = mesh.rs[:,0] <= mesh._nekmin
                seccond = mesh.rs[:,0] > mesh._nekmin
                pots1_inds = pots_inds[primcond]
                pots2_inds = pots_inds[seccond]
                
                # structure population magic
                Ts, rhos = np.zeros(complen), np.zeros(complen)
                Ts1, rhos1 = Ts1_pots[pots1_inds], rhos1_pots[pots1_inds]
                Ts2, rhos2 = Ts2_pots[pots2_inds], rhos2_pots[pots2_inds]
                Ts[primcond], rhos[primcond] = Ts1, rhos1
                Ts[seccond], rhos[seccond] = Ts2, rhos2

                # save files
                np.save(directory+'T_0', Ts.reshape(mesh._dims))
                np.save(directory+'rho_0', rhos.reshape(mesh._dims))


            else:
                raise NotImplementedError('Geometry %s not implemented with polytropic structure.' % mesh._geometry)
            