import numpy as np 
import os, shutil
import logging
from scipy.optimize import newton
from scipy.special import legendre

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


def compute_diffrot_potential(vertex, pot_r, bs, scale):
    rabs = np.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
    theta = np.arccos(vertex[2] / rabs)
    r = radius_newton(pot_r, bs, theta)

    if rabs <= r+1e-12:
        return DiffRotRoche(vertex / scale, bs)
    else:
        return 0.


def DiffRotRoche(r, bs):
    if r.shape == (3,):
        return (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** (-0.5) + 0.5 * (r[0] ** 2 + r[1] ** 2) * (
            bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2)
    else:
        return (r[:,0] ** 2 + r[:,1] ** 2 + r[:,2] ** 2) ** (-0.5) + 0.5 * (r[:,0] ** 2 + r[:,1] ** 2) * (
            bs[0] + 0.5 * bs[1] * (r[:,0] ** 2 + r[:,1] ** 2) + 1. / 3. * bs[2] * (r[:,0] ** 2 + r[:,1] ** 2) ** 2)


def dDiffRotRochedx(r, bs):
    return r[0] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5 + r[0] * (
        bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2) + 0.5 * r[
        0] * (r[0] ** 2 +r[1] ** 2) * (bs[1] + 4. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2))


def dDiffRotRochedy(r, bs):
    return r[1] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5 + r[1] * (
        bs[0] + 0.5 * bs[1] * (r[0] ** 2 + r[1] ** 2) + 1. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2) ** 2) + 0.5 * r[
        1] * (r[0] ** 2 + r[1] ** 2) * ( bs[1] + 4. / 3. * bs[2] * (r[0] ** 2 + r[1] ** 2))


def dDiffRotRochedz(r, bs):
    return r[2] / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 0.5


def radius_newton(pot, bs, theta):
    r0 = 1. / pot
    x = 1. - (np.cos(theta)) ** 2

    r_start = r0*(1. + (bs[0]*r0**3*x)/2. + (bs[1]*r0**5*x**2)/4. + (3*bs[0]**2*r0**6*x**2)/4. +
    (bs[2]*r0**7*x**3)/6. + bs[0]*bs[1]*r0**8*x**3 + (3*bs[0]**3*r0**9*x**3)/2. +
    (5*(3*bs[1]**2 + 8*bs[0]*bs[2])*r0**10*x**4)/48. + (55*bs[0]**2*bs[1]*r0**11*x**4)/16. +
    (13*bs[0]*(3*bs[1]**2 + 4*bs[0]*bs[2])*r0**13*x**5)/16. +
    (35*bs[0]**2*(9*bs[1]**2 + 8*bs[0]*bs[2])*r0**16*x**6)/24. +
    (r0**12*x**4*(55*bs[0]**4 + 8*bs[1]*bs[2]*x))/16. +
    (51*bs[0]*r0**18*x**6*(7*bs[0]**5 + 2*bs[1]**3*x + 8*bs[0]*bs[1]*bs[2]*x))/16. +
    (7*r0**15*x**5*(78*bs[0]**5 + 5*bs[1]**3*x + 40*bs[0]*bs[1]*bs[2]*x))/64. +
    (7*r0**14*x**5*(117*bs[0]**3*bs[1] + 2*bs[2]**2*x))/72. +
    (17*r0**17*x**6*(315*bs[0]**4*bs[1] + 12*bs[1]**2*bs[2]*x + 16*bs[0]*bs[2]**2*x))/144. +
    (19*r0**19*x**7*(51*bs[0]**3*bs[1]**2 + 34*bs[0]**4*bs[2] + bs[1]*bs[2]**2*x))/16.)

    def Roche(r):
        return pot - (1./r + 0.5 * r**2 * x * (bs[0] + 0.5*bs[1]*x*r**2 + 1./3. * bs[2] * x**2 * r**4))

    r = newton(Roche, r_start, maxiter = 1000)

    return r