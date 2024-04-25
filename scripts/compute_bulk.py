import xarray as xr
from mlflux.datafunc import load_psd, load_atomic

print('Loading PSD data and computing bulk!')
ds_psd = load_psd('../data/PSD/fluxes_all_cruises_compilation.nc') # This load_psd function includes applybulk
ds_psd['tauby'] = xr.zeros_like(ds_psd['taubx']) # bulk formula doesn't predict tauy
ds_psd.to_netcdf('../data/Processed/psd.nc')

print('Loading ATOMIC data and computing bulk!')
ds_atomic = load_atomic('../data/WHOI/EUREC4A_ATOMIC_RonBrown_10min_nav_met_sea_flux_20200109-20200212_v1.3.nc')
ds_atomic['tauby'] = xr.zeros_like(ds_atomic['taubx']) 
ds_atomic.to_netcdf('../data/Processed/atomic.nc')