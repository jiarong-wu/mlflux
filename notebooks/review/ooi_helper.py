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

def clean (ds):
    ds = ds.where(ds.Sbytes==0, drop=True)
    ds['weight'] = ds['Beta'] / ds['Beta'] 
    ds['Pair'] = ds['Pair'] * 100.
    ds['SH'] = ds['WT']*ds['cpa']*ds['rhoair']
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
    