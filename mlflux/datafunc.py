''' Helper functions related to data. '''

# We split into training, validation, and testing
# plt.plot(ds_clean.pcode) # 77, 69, 83, 78, 87, 72, 71, 68, 67, 73

import datetime
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from aerobulk.flux import noskin 

def load_psd (filepath):
    ''' Load the psd data and compute bulk '''
    ds = xr.load_dataset(filepath)
    
    # Remove nan
    ds_psd = ds.dropna(dim="time", how="any", 
                       subset=['taucx','taucy','hsc','hlc','U','tsnk','ta','qa'])
    print('Number of samples: %g' %(len(ds_psd.U.values)))
    
    # Rename fields (dummy2 is relative humidity)
    ds_psd = ds_psd.rename_vars({'tsnk':'tsea','ta':'tair','qa':'qair','dummy2':'rh'})
    # Drop the not used variables
    ds_psd = ds_psd[['taucx','taucy','hsc','hlc','U','tsea','tair','qair',
                     'rh', 'pcode','zu','zt','zq']]
    
    # A few more adjustments that are data set specific 
    ds_psd['qair'] = ds_psd['qair']/1000. # Make it into unit kg/kg
    ds_psd['hsc'] = -ds_psd['hsc'] # Heat flux is positive when it's from air to ocean
    ds_psd['hlc'] = -ds_psd['hlc']
    
    # Compute bulk using COARE3.6 and then append to dataset
    # Here when zq and zt are different height we use zt
    ds_psd = applybulk(ds_psd, algo='coare3p6')
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
    ds_atomic = ds_atomic[['taucx','taucy','hsc','hlc','U','tsea','tair','qair','rh','zu','zt','zq']]
    
    # Compute bulk using COARE3.6 and then append to dataset
    # Here zq and zt are the same height 
    ds_atomic = applybulk(ds_atomic, algo='coare3p6')
    return ds_atomic

def applybulk(ds, algo='coare3p6'):
    ''' Dependence: aerobulk-python
        https://github.com/jbusecke/aerobulk-python 
        Installation through conda works on Greene but not on MacBook yet.
    '''
    hl, hs, taux, tauy, evap = noskin(sst=ds.tsea+273.15, t_zt=ds.tair+273.15, 
                                      hum_zt=ds.qair, u_zu=ds.U, v_zu=ds.U*0, 
                                      slp=ds.U/ds.U*101000.0, algo=algo, 
                                      zt=ds.zt, zu=ds.zu)
    ds = ds.assign(hlb=hl,hsb=hs,taubx=taux)
    return ds

# If we need to compute relative humidity
# from utils import rh

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
    
def data_split_psd(ds, split, PLOT=True, XVIS='time'):
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
    print('Training samples: %g' %len(psd_train.U.values))

    psd_valid = ds.where(ds.pcode.isin(split[1]), drop=True)
    print('Validating samples: %g' %len(psd_valid.U.values))

    psd_test = ds.where(ds.pcode.isin(split[2]), drop=True)
    print('Testing samples: %g' %len(psd_test.U.values))

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