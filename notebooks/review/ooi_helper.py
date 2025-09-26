import numpy as np
from collections import Counter
import xarray as xr
import glob
import scipy.io
import os

# Suggested by copilot to avoid misalignment in lengths of variables
# ds_ooi = xr.Dataset({key: ([], val) if not hasattr(val, 'shape') else (['dim'], val)
#                 for key, val in mat_data.items() if not key.startswith('__')})
def convert_mat_to_xarray(mat_data):
    # mat_data is from scipy.io.loadmat(...)
    # 1) Inspect lengths quickly to decide a reference length
    lengths = {}
    for k, v in mat_data.items():
        if k.startswith('__'):
            continue
        if hasattr(v, 'shape') and np.shape(v) != ():
            # treat column/row vectors as 1D
            lengths[k] = np.atleast_1d(v).ravel().shape[0]

    # Prefer 'Sbytes' as reference if present (the data quality variable for filtering),
    # otherwise use the most common length among arrays.
    if 'Sbytes' in lengths:
        ref_len = lengths['Sbytes']
    else:
        ref_len = Counter(lengths.values()).most_common(1)[0][0]

    # 2) Build dataset: put variables with ref_len on shared dim, others get their own dim
    data_vars = {}
    coords = {}

    for k, v in mat_data.items():
        if k.startswith('__'):
            continue

        # Scalars or empty shapes -> store as scalar (no dim)
        if not hasattr(v, 'shape') or np.shape(v) == ():
            data_vars[k] = v
            continue

        arr = np.atleast_1d(v).ravel()  # ensure 1D
        if arr.shape[0] == ref_len:
            data_vars[k] = (['dim'], arr)
        else:
            # give the variable its own dim name to avoid conflicts
            dimname = f"{k}_dim"
            data_vars[k] = ([dimname], arr)
            coords[dimname] = np.arange(arr.shape[0])

    # 3) create the dataset
    ds_ooi = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds_ooi

# Clean and preprocess the dataset
# Convert units, compute fluxes, rename variables, drop NaNs
# stress	Bulk stress (N/m2)
# shf		Bulk sensible heat flux (W/m2)
# lhf		Bulk latent heat flux (W/m2)
# bhf		Bulk virtual temperature flux (W/m2) or buoyancy flux
# bshf	Bulk sonic temperature flux (W/m2)
# moL		Bulk Monin-Obukhov length (m)
def clean (ds):
    ds = ds.where(ds.Sbytes==0, drop=True)
    ds['Pair'] = ds['Pair'] * 100.
    ds['hsc'] = -ds['WT']*ds['cpa']*ds['rhoair']
    ds['taucx'] = -ds['UW']*ds['rhoair']
    ds['taucy'] = -ds['VW']*ds['rhoair']
    ds['taubx'] = ds['stress']; ds['hsb']= -ds['shf']; ds['hlb']=-ds['lhf']
    # rename some variables
    # seems like shf is sensible heat flux but bshf is sonic temperature flux
    # Sonic temperature is virtual temperature (corrected with 1+0.61q)
    ds = ds.rename({'Ue':'U', 'Tair':'tair', 'Tsea':'tsea', 'RH':'rh', 'Pair':'p'})
    # keep only useful variables
    keep_vars = ['U', 'tair', 'tsea', 'rh', 'p', 'zq', 'zt', 'zu', 
                 'taucx', 'taucy', 'hsc', 'taubx', 'hsb', 'hlb', 'yyyy', 'yday']
    ds = ds[keep_vars]
    ds = ds.dropna(dim='dim', how='any')
    # ds = ds.assign(weight=xr.DataArray(np.ones(ds.dims['dim'])/ds.sizes['dim'], dims=['dim']))
    return ds

if __name__ == "__main__":
    # Example usage
    path = '/scratch/jw8736/mlflux/data/OOI/'
    campaign = 'Endurance'
    pattern = [f'Transfer_{campaign}_*_v3.mat']
    filenames = sorted(glob.glob(os.path.join(path, pattern[0])))

    for i, filename in enumerate(filenames):
        mat_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        ds = convert_mat_to_xarray(mat_data)
        ds = clean(ds)
        print(f'Processed {i+1}/{len(filenames)}: {filename}, shape: {ds.dims}')
        if i == 0:
            ds_all = ds
        else:
            ds_all = xr.concat([ds_all, ds], dim='dim')
        ds_all.to_netcdf(os.path.join(path, f'{campaign}.nc'))
    