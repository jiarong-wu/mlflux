''' Utility functions '''

import numpy as np
import xarray as xr
from numpy import copy, asarray, exp

def save_ds_compressed(ds, filename):
    encoding = {}
    for var_name in ds.data_vars:
        encoding[var_name] = {'dtype': 'float32', 'zlib': True}

    ds.to_netcdf(filename, encoding=encoding)
    
''' Get cruise info, return a list of dictionary '''
def get_cruise ():
    metz = {'name':'metz', 'pcode':77, 'months':['199301','199302','199303','199304','199305','199306','199307','199308','199309',
            '199612','199801','199905','199906','199907','199908','199909','199910','199911','199912']} # checked
    calwater = {'name':'calwater', 'pcode':67, 'months':['201501','201502']}
    hiwings = {'name':'hiwings', 'pcode':72, 'months':['201309','201310','201311']}
    capricorn = {'name':'capricorn', 'pcode':73, 'months':['201603','201604']}
    dynamo = {'name':'dynamo', 'pcode':68, 'months':['201109','201110','201111','201112']}
    stratus = {'name':'stratus', 'pcode':83, 'months':['200110','200412','200510','200610','200710','200711',
                                                      '200810','200811','200812','201001']}
    epic = {'name':'epic', 'pcode':69, 'months':['199911','199912',
            '200004','200005','200006','200007','200008','200009','200010','200011',
            '200103','200104','200105','200106','200107','200108','200109','200110','200111','200112',
            '200203','200204','200205','200206','200207','200208','200209','200210','200211',
            '200311','200410','200411']}
    whots = {'name':'whots', 'pcode':87, 'months':['200907','201107','201206','201307','201407','201507']} #checked
    neaqs = {'name':'neaqs', 'pcode':78, 'months':['200407','200408']} #checked
    gasex = {'name':'gasex', 'pcode':71, 'months':['200803','200804']} #checked
    cruises = [metz, calwater,hiwings,capricorn,dynamo,stratus,epic,whots,neaqs,gasex]
    return cruises


''' Adjust longitude in datasets '''

def sort_longitude(x):
    x['lon'] = np.where(x['lon']>0, x['lon'], 360+x['lon'])
    x = x.sortby('lon')
    return x

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