"""
    Handling custom density profile
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.special import hyp2f1

from src.input_parser import read_form


def prepare_form(form_path):
    """

    Prepares interpolator for form factor

    """

    x, F = form_table(form_path)

    logx_ = np.log(x)
    logF_ = np.log(F)

    log_interp = interp1d(logx_,logF_,bounds_error=False,fill_value='extrapolate')

    def fun(r,r_s,rho_s):
        logx = np.log(r/r_s)

        return np.exp(np.piecewise(
            logx,
            [logx<logx_[-1],logx>=logx_[-1]],
            [log_interp,lambda x: logF_[-1]],
            )) * rho_s * r_s**3

    return fun


def postencounter_density_profile(x):
  
    """
    
    The universal halo density profile after impulsive tidal stripping, as
    defined in arXiv:xxxx.xxxxx.
    
    """
    
    alpha = 0.78
    beta = 5.
    
    q = (1./3*x**alpha)**beta
    return np.exp(-1./alpha * x**alpha * (1+q)**(1-1/beta) * hyp2f1(1,1,1+1/beta,-q))/x


def form_table(form_path=None):
    
    if form_path is not None and len(form_path) > 0:
        x, F = read_form(form_path)
        
    else: # use default
        x = np.geomspace(1e-12,1e15,10000)
        rho = postencounter_density_profile(x)
        F = cumtrapz(4*np.pi*x**3 * rho,x=np.log(x),initial=0) + 2*np.pi*x[0]**2
        
    return x, F