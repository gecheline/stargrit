import numpy as np

def return_MS_factors(mass):

    if mass <= 1.:
        zeta = 0.8
    else:
        zeta = 0.57

    if mass <= 0.25:
        nu = 2.5
    elif mass <= 10.:
        nu = 4.
    else:
        nu = 3.

    return zeta, nu