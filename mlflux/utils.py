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



import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_animation(fun, idx, filename='my-animation.gif', dpi=200, FPS=18, loop=0):
    '''
    See https://pythonprogramming.altervista.org/png-to-gif/
    fun(i) - a function creating one snapshot, has only one input:
        - number of frame i
    idx - range of frames, i in idx
    FPS - frames per second
    filename - animation name
    dpi - set 300 or so to increase quality
    loop - number of repeats of the gif
    '''
    frames = []
    for i in idx:
        fun(i)
        plt.savefig('.frame.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        frames.append(Image.open('.frame.png').convert('RGB'))
        print(f'Frame {i} is created', end='\r')
    os.system('rm .frame.png')
    # How long to persist one frame in milliseconds to have a desired FPS
    duration = 1000 / FPS
    print(f'Animation at FPS={FPS} will last for {len(idx)/FPS} seconds')
    frames[0].save(
        filename, format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop)