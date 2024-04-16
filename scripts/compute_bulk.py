import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mlflux.datafunc import load_psd, load_atomic, assemble_var, data_split_psd

print('Loading PSD data')
ds_psd = load_psd('../data/PSD/fluxes_all_cruises_compilation.nc')
# train, valid, test = data_split_psd(ds_psd, split=[[77, 69, 83, 78], [87, 72, 71], [68, 67, 73]],
#                                     PLOT=True, XVIS='time')
# X_train, Y_train = assemble_var(train, choice='U_Tdiff_rh')
# X_valid, Y_valid = assemble_var(valid, choice='U_Tdiff_rh')
# X_test, Y_test = assemble_var(test, choice='U_Tdiff_rh')
ds_psd.to_netcdf('../data/Processed/psd.nc')

print('Loading ATOMIC data')
ds_atomic = load_atomic('../data/WHOI/EUREC4A_ATOMIC_RonBrown_10min_nav_met_sea_flux_20200109-20200212_v1.3.nc')
# X_test_atomic, Y_test_atomic = assemble_var(ds_atomic, choice='U_Tdiff_rh')
ds_atomic.to_netcdf('../data/Processed/atomic.nc')