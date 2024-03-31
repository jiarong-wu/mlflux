''' Utility functions '''

import numpy as np
import xarray as xr
from numpy import copy, asarray, exp

''' Physical quantities. '''

def qsat(t,p):
    ''' TAKEN FROM COARE PACKAGE. Usage: es = qsat(t,p)
        Returns saturation vapor pressure es (mb) given t(C) and p(mb).
        After Buck, 1981: J.Appl.Meteor., 20, 1527-1532
        Returns ndarray float for any numeric object input.
    '''

    t2 = copy(asarray(t, dtype=float))  # convert to ndarray float
    p2 = copy(asarray(p, dtype=float))
    es = 6.1121 * exp(17.502 * t2 / (240.97 + t2))
    es = es * (1.0007 + p2 * 3.46e-6)
    return es

def rhcalc(t,p,q):
    ''' TAKEN FROM COARE PACKAGE. usage: rh = rhcalc(t,p,q)
        Returns RH(%) for given t(C), p(mb) and specific humidity, q(kg/kg)
        Returns ndarray float for any numeric object input.
    '''
    
    q2 = copy(asarray(q, dtype=float))    # conversion to ndarray float
    p2 = copy(asarray(p, dtype=float))
    t2 = copy(asarray(t, dtype=float))
    es = qsat(t2,p2)
    em = p2 * q2 / (0.622 + 0.378 * q2)
    rh = 100.0 * em / es
    return rh

def rhcalc_xr(ds):
    ''' xarray wrapper for rhcalc, requires the name to match. '''
    
    xr.apply_ufunc(
        rhcalc,
        ds.ta,
        ds.p,
        ds.qa, # remember to divide by 1000 is unit is g/kg
        input_core_dims=[()] * 3,
        output_core_dims=[()] * 1,
        dask="parallelized",
        output_dtypes=[ds.ta.dtype] * 1,  # deactivates the 1 element check which aerobulk does not like
)
    
    
''' Some statistical functions. '''
def mse_r2(ypred, ytruth):
    mse = np.average((ypred-ytruth)**2)
    r2 = 1 - np.average((ypred-ytruth)**2)/np.var(ytruth)
    return (mse,r2)