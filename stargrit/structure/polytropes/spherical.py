import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import odeint

class Polytrope(object):

    def __init__(self, n):

        """
        Creates a spherical polytrope with polytropic index n.

        Parameters
        ----------
        n: float
            Polytropic index of the model.

        Methods
        -------
        lane_emden
            Solves the basic (spherical) Lane-Emden equation of the polytrope.
        __call__
            Returns the solution of the Lane-Emden equation.
        """
        self.__n = n

    @staticmethod
    def lane_emden(n=3, dt=1e-4):
        """
        Numerical solution of the standard Lane-Emden equation.

        Parameters
        ----------
        n: float
            Polytropic index of the model
        dt: float
            Discretization step of the dimensionless radial variable

        Returns
        -------
        t_surface: float
            Dimensionless radial value of the surface.
        dtheta_surface: float
            Derivative of the theta function at the surface.
        theta_interp: scipy.interpolate.interp1d object
            1D interpolation function of (t,theta)
            TODO: maybe replace this with just the array?
        """
        def f(y, t, n):
            return [y[1], -np.abs(y[0]) ** n - 2 * y[1] / t]

        y0 = [1., 0.]
        
        if n <= 1:
            tmax = 3.5
        elif n <= 2:
            tmax = 5.
        elif n <= 3.5:
            tmax = 10.
        else:
            tmax = 20.
            
        ts = np.arange(1e-120, tmax, 1e-4)
        soln = odeint(f, y0, ts, args=(n,))

        theta_interp = interp1d(ts, soln[:,0])
        dtheta_interp = interp1d(ts, soln[:,1])

        # compute the value of t and dthetadt where theta falls to zero
        ts_theta_interp = interp1d(soln[:,0], ts)
        t_surface = float(ts_theta_interp(0.))
        dthetadt_surface = float(dtheta_interp(t_surface))
                
        return t_surface, dthetadt_surface, theta_interp


    def __call__(self, dt=1e-4):
        return self.lane_emden(n=self.__n, dt=dt)