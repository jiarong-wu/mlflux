###### Read in tprof, sprof, interpolate to grid, and save to netcdf (takes about 10 min) ########
# Used for analysis in gotm/Eval5
# file_path = path + 'tprof_papa_hourly' # not uniform (upper 300m) verqtical layers, hourly
# file_path = path + 'tprof_woa.dat' # uniform vertical fqull depth but monthly

from mlflux.gotm import process_file
from mlflux.utils import save_ds_compressed
import xarray as xr
import numpy as np

path = '/scratch/jw8736/gotm/ensem/shared/' # need to specify
zgrid = np.linspace(-200,-1,200) + 0.5 # need to specify

file_path = path + 'tprof.dat' # uniform vertical (upper 300m), roughly 6 hourly
t, temp_full = process_file(file_path, zgrid)
file_path = path + 'sprof.dat' # uniform vertical (upper 300m), roughly 6 hourly
t, salinity_full = process_file(file_path, zgrid)

ds = xr.Dataset({'T':(['t','z'], np.array(temp_full)),
                 'S':(['t','z'], np.array(salinity_full))},
                 coords={'t': t,
                         'z': zgrid})

save_ds_compressed(ds, path+'profs.nc')    