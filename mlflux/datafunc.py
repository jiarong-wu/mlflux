''' Helper functions related to data. '''

# We split into training, validation, and testing
# plt.plot(ds_clean.pcode) # 77, 69, 83, 78, 87, 72, 71, 68, 67, 73

import datetime
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
# For when we don't have aerobulk installed then applybulk can't be used 
try:
    from aerobulk.flux import noskin 
except ImportError:
    pass


def load_psd (filepath, algo='coare3p6'):
    ''' Load the psd data and compute bulk '''
    ds = xr.load_dataset(filepath)
    
    # Remove nan
    ds_psd = ds.dropna(dim="time", how="any", 
                       subset=['taucx','taucy','hsc','hlc','U','tsnk','ta','qa'])
    print('Number of samples: %g' %(len(ds_psd.U.values)))
    
    # Rename fields (dummy2 is relative humidity)
    ds_psd = ds_psd.rename_vars({'tsnk':'tsea','ta':'tair','qa':'qair','dummy1':'p','dummy2':'rh','dir':'wdir'})
    # Drop the not used variables
    ds_psd = ds_psd[['taucx','taucy','hsc','hlc','U','tsea','tair','qair',
                     'p', 'rh', 'pcode','zu','zt','zq','lon','lat','wdir']] # Update: added the wind direction
    
    # A few more adjustments that are data set specific 
    ds_psd['qair'] = ds_psd['qair']/1000. # Make it into unit kg/kg
    ds_psd['p'] = ds_psd['p']*100. # from millibar to pascal
    ds_psd['hsc'] = -ds_psd['hsc'] # Heat flux is positive when it's from air to ocean
    ds_psd['hlc'] = -ds_psd['hlc']
    
    # Compute bulk using COARE3.6 and then append to dataset
    # Here when zq and zt are different height we use zt
    ds_psd = applybulk(ds_psd, algo=algo)
    return ds_psd

def load_atomic(filepath):
    ''' Atomic - loading and processing '''
    ds = xr.open_dataset(filepath)
    ds_atomic = ds.dropna(dim="obs", how="any",
                          subset=["tau_streamwise_cov","tau_crossstream_cov",
                                  "tau_bulk","hl_cov","hs_cov","wspd",'tsea','tair','qair'])
    print('Number of samples: %g' %(len(ds_atomic.wspd.values)))
    
    # Rename fields
    ds_atomic = ds_atomic.rename_vars({'wspd':'U','tau_streamwise_cov':'taucx','tau_crossstream_cov':'taucy',
                                       'hl_cov':'hlc','hs_cov':'hsc','rhair':'rh'})
    
    # A few more adjustments that are data set specific 
    ds_atomic = ds_atomic.reset_coords('ztq')
    ds_atomic = ds_atomic.reset_coords('zu')
    ds_atomic = ds_atomic.rename_vars({'ztq':'zt'})
    ds_atomic = ds_atomic.assign(zq=ds_atomic.zt) # zt and zq are the same for this one
    ds_atomic['qair'] = ds_atomic['qair']/1000. # Make it into unit kg/kg
    
    # # Drop the not used variables (for atomic zu and ztq are coordinates)
    ds_atomic = ds_atomic[['taucx','taucy','hsc','hlc','U','tsea','tair','qair','rh','zu','zt','zq',
                           'wdir','cspd','cdir','wave_height','wave_period','wave_flag']] # Update: added the current info
    
    # Compute bulk using COARE3.6 and then append to dataset
    # Here zq and zt are the same height 
    ds_atomic = applybulk(ds_atomic, algo='coare3p6')
    return ds_atomic

def applybulk(ds, algo='coare3p6'):
    ''' Dependence: aerobulk-python
        https://github.com/jbusecke/aerobulk-python 
        Installation through conda works on Greene but not on MacBook yet.
        Update: use measured pressure instead of constant pressure!
    '''
    hl, hs, taux, tauy, evap = noskin(sst=ds.tsea+273.15, t_zt=ds.tair+273.15, 
                                      hum_zt=ds.qair, u_zu=ds.U, v_zu=ds.U*0, 
                                      slp=ds.p, algo=algo, 
                                      zt=ds.zt, zu=ds.zu)  
    ds = ds.assign(hlb=hl,hsb=hs,taubx=taux)
    return ds

# If we need to compute relative humidity
# from utils import rh

# This can be obsolete now because we have a better way
def assemble_var (ds, choice='U_Tdiff_rh'):
    ''' Here ds has to have variable names as ['U','tsea','tair','qair','rh','taucx','hsc','hlc'] '''
    # U-Ta-To-q_absolute
    if choice == 'U_To_Ta_q':
        X = np.hstack([np.reshape(ds.U.values.astype('float32'),(-1,1)), 
                        np.reshape(ds.tsea.values.astype('float32'),(-1,1)),
                        np.reshape(ds.tair.values.astype('float32'),(-1,1)), 
                        np.reshape(ds.qair.values.astype('float32'),(-1,1))])
    # U-Ta-To-q_relative
    if choice == 'U_To_Ta_q':
        X = np.hstack([np.reshape(ds.U.values.astype('float32'),(-1,1)), 
                        np.reshape(ds.tsea.values.astype('float32'),(-1,1)),
                        np.reshape(ds.tair.values.astype('float32'),(-1,1)), 
                        np.reshape(ds.rh.values.astype('float32'),(-1,1))])  
    # U-Tdiff-q_relative
    if choice == 'U_Tdiff_rh':
        X = np.hstack([np.reshape(ds.U.values.astype('float32'),(-1,1)), 
                       np.reshape((ds.tair-ds.tsea).values.astype('float32'),(-1,1)),
                       np.reshape(ds.rh.values.astype('float32'),(-1,1))])

    Y = np.hstack([np.reshape(ds.taucx.values.astype('float32'),(-1,1)),
                   np.reshape(ds.hsc.values.astype('float32'),(-1,1)),
                   np.reshape(ds.hlc.values.astype('float32'),(-1,1))])
    
    return (X,Y)
    
def data_split_psd(ds, split, PLOT=True, XVIS='time', VERBOSE=True):
    ''' Split the data into training, validation, and testing. 
        This function is specific to the PSD data set with the cruise 
        labeled by [77, 69, 83, 78, 87, 72, 71, 68, 67, 73].
        Arguments: 
            Split: is specified in the form of list of list, e.g. 
                   [[77, 69, 83, 78], [87, 72, 71], [68, 67, 73]]
            PLOT: if True, also visualize the splitting
            XVIS: if 'time', plot x axis as time; if 'samples', plot x axis as samples
    '''
    colors = ['Blue','Purple','Pink']
    psd_train = ds.where(ds.pcode.isin(split[0]), drop=True)
    psd_valid = ds.where(ds.pcode.isin(split[1]), drop=True)
    psd_test = ds.where(ds.pcode.isin(split[2]), drop=True)
    
    if VERBOSE:
        print('Training samples: %g' %len(psd_train.U.values))
        print('Validating samples: %g' %len(psd_valid.U.values))
        print('Testing samples: %g' %len(psd_test.U.values))

    if PLOT:
        fig = plt.figure(figsize=[6,4], dpi=200) 
        for i in range(3):
            for pcode in split[i]:
                if XVIS == 'time':
                    plt.plot(ds.where(ds.pcode==pcode).time,
                            ds.where(ds.pcode==pcode).pcode, c=colors[i]) 
                    plt.xlim([datetime.date(1996,1,1), datetime.date(2020,1,1)])
                    plt.xlabel('Year')
                if XVIS == 'samples':
                    plt.plot(ds.where(ds.pcode==pcode).pcode, c=colors[i])
                    plt.xlim([0,10000]); plt.xlabel('Samples')    
        
        plt.yticks([77, 69, 83, 78, 87, 72, 71, 68, 67, 73])
        plt.ylabel('Cruise code')
        plt.annotate('Training %g' %len(psd_train.U.values), 
                    xy=(0.01,0.95), xycoords='axes fraction', color=colors[0])
        plt.annotate('Validating %g' %len(psd_valid.U.values), 
                    xy=(0.01,0.9), xycoords='axes fraction', color=colors[1])
        plt.annotate('Testing %g' %len(psd_test.U.values), 
                    xy=(0.01,0.85), xycoords='axes fraction', color=colors[2])
        plt.show()

    return (psd_train, psd_valid, psd_test)

def data_split_psd_rand(ds, seed=7, ratio=0.2):
    ''' Split the data into training, validation, and testing randomly. 
    '''
    np.random.seed(seed)
    N = ds.sizes['time']
    test_indices = np.random.choice(N, size=int(N*ratio), replace=False)
    train_indices = np.setdiff1d(np.arange(0,N), test_indices)

    psd_train = ds.isel(time=train_indices)
    psd_valid = ds.isel(time=test_indices)
    psd_test = ds.isel(time=test_indices)
    
    return (psd_train, psd_valid, psd_test)