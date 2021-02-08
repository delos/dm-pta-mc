"""
    Handling custom density profile
"""

import numpy as np
from scipy.interpolate import interp1d

from src.input_parser import read_form


def prepare_form(form_path):
    """

    Prepares interpolator for form factor

    """

    x, F = read_form(form_path)

    logx_ = np.log(x)
    logF_ = np.log(F)

    log_interp = interp1d(logx_,logF_,bounds_error=False,fill_value='extrapolate')

    def fun(r,r_s,rho_s):
        logx = np.log(r/r_s)

        return np.exp(np.piecewise(
            logx,
            [logx<logF[-1],logx>=logx_[-1]],
            [log_interp(logx),logF_[-1]],
            )) * rho_s * r_s**3

    return fun

