''' Generate dataset ready for training. 
    Example use: python gen_dataset.py --bulkalg=coare3p0 --weight=1 --wave=0 '''

import argparse
import xarray as xr
import numpy as np
from mlflux.datafunc import load_psd
from mlflux.gotm import read2010
import os

ows_papa_path = '/home/jw8736/code-5.2.1/cases/ows_papa/' # path where ows_papa distribution can be read
filepath = '../data/PSD/fluxes_all_cruises_compilation.nc' # absolute or relative path to the original PSD dataset

##### This function computes weights based on three variables: U, To-Ta, RH. Can be expanded to more flexible choices. #######
from adapt.instance_based import KLIEP
def weighting (ds_sample, ds_target, vars_list = ['U','tdiff','rh']):
    # Createan array by stacking the specified variables
    X_sample = np.hstack([np.array(ds_sample[var]).reshape(-1, 1) for var in vars_list])
    X_target = np.hstack([np.array(ds_target[var]).reshape(-1, 1) for var in vars_list])
    kliep = KLIEP(kernel="rbf", gamma=[10**(i-3) for i in range(7)], random_state=0)
    kliep_weights = kliep.fit_weights(X_sample, X_target)
    ds_sample['weight'] = (['time'], kliep_weights)  # Mean of weights is 1
    return ds_sample
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the training dataset from PSD, with optional weighting.')
    parser.add_argument('--bulkalg', type=str, help='Bulk algorithm used. Options: coare3p6, coare3p0, ncar.')
    parser.add_argument('--weight', type=int, help='Weighting towards target dist. (0: no weighting, 1: ocean papa, 2: global)')
    parser.add_argument('--wave', type=int, help='Whether to include interpolated wave info in dataset. (0: no wave vars, 1: with wave vars)')
    parser.add_argument('--outpath', type=str, default='../data/Processed/', help='Path for the output netcdf file.')
    args = parser.parse_args()
    
    # compute bulk with a given algorithm
    ds_psd = load_psd(filepath, algo=args.bulkalg)
    ds_psd['tauby'] = xr.zeros_like(ds_psd['taubx']) # bulk formula doesn't predict tauy
    
    # Optional weighting based on different target
    if args.weight == 1:
        # Read in target files
        df = read2010(ows_papa_path, datetimeformat='%Y-%m-%d %H:%M:%S') 
        df_ = df.set_index('datetime'); ds = xr.Dataset.from_dataframe(df_)
        ds_papa = ds.sel(datetime=slice('2012-01-01','2012-12-31'))
        # Adjust variable names
        ds_papa['U'] = ((ds_papa.ux)**2+(ds_papa.uy)**2)**0.5
        ds_papa['tdiff'] = ds_papa.sst - ds_papa.t
        ds_psd['tdiff'] = ds_psd.tsea - ds_psd.tair
        # Compute weights
        ds_psd = weighting (ds_psd, ds_papa, vars_list = ['U','tdiff','rh'])       
    elif args.weight == 2:
        print('Global weighting not yet implemented!')       
    else:
        print('No weighting!')

    # TODO: include co-located or interpolated wave info if wave=1
              
    # Same to desired directory and name
    if not os.path.exists(args.outpath):
        # Create the directory and any necessary parent directories
        os.makedirs(args.outpath)
        print(f"Directory {args.outpath} created.")
    else:
        print(f"Directory {args.outpath} already exists.")
   
    ds_psd.attrs['bulkalg'] = args.bulkalg
    ds_psd.attrs['weight'] = args.weight
    ds_psd.attrs['wave'] = args.wave
    ds_psd.to_netcdf(args.outpath + f'psd_{args.bulkalg}_weight%d_wave%d.nc' %(args.weight,args.wave))