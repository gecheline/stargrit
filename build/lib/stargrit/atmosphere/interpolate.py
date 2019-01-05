import numpy as np
import os
from scipy.interpolate import interp2d

def opacities(Ts,rhos,XYZ=None):

    # load opal table and prepare for interpolation
    opaldir_local = os.path.dirname(os.path.abspath(__file__))+'/'
    opal = np.loadtxt(opaldir_local+'tables/opal_sun.csv', delimiter=',')
    x = opal.T[:, 0]
    y = opal[:, 0]
    x = np.delete(x, 0)
    y = np.delete(y, 0)
    opac = np.delete(opal, 0, 0)
    opac = np.delete(opac.T, 0, 0).T
    opac_interp = interp2d(x, y, opac)

    logTs = np.log10(Ts)
    logRs = np.log10(rhos / ((Ts * 1e-6) ** 3))

    logk = np.zeros(len(logTs))
    for i,logT in enumerate(logTs):
        logk[i] = opac_interp(logRs[i], logTs[i])[0]

    logk[(logk==0.) & (logk > 9.99)] = -np.inf

    return 0.1 * (10 ** logk)  # in m^2/kg (multiplied by 0.1)