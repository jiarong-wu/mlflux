import xarray as xr

# Path where the wave reanalysis are stored
wavepath = '/Users/jwu/Work/Dabble/Waves/'

# Dictionary contain each cruise and the corresponding months
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

# First load the cleaned up PSD file and drop all data before 1993 because the wave data set only contains 1993 and after. 
psd = xr.load_dataset('../data/Processed/psd.nc')
ds = psd.where(psd.time.dt.year >= 1993, drop=True) # Reference for datetime: https://docs.xarray.dev/en/latest/user-guide/time-series.html

# Adding new variables
new_vars = ['phs0', 'plp0', 'pdir0', 'phs1', 'plp1', 'pdir1']
var_descriptions = ['Wind wave height [m]', 'Wind wave peak wavelength [m]', 'Wind wave direction [degree]',
                    'Swell height [m]', 'Swell peak wavelength [m]', 'Swell direction [degree]']
for var,var_description in zip(new_vars,var_descriptions):
    ds[var] = xr.zeros_like(ds.U)
    ds[var].attrs['long_name'] = var_description

# Iterate over cruises
for c in cruises:
    # Assemble the wave reanalysis
    month = c['months'][0]
    file = wavepath + 'data/LOPS_WW3-GLOB-30M_' + month +'.nc' 
    ds_ERA5 = xr.open_dataset(file,chunks={'time':'500MB'})
    for month in c['months'][1:]:
        file = wavepath + 'data/LOPS_WW3-GLOB-30M_' + month +'.nc' 
        dsmonth = xr.open_dataset(file,chunks={'time':'500MB'})
        ds_ERA5 = xr.concat([ds_ERA5,dsmonth],dim='time')
    # Interpolate
    ds_ERA5 = ds_ERA5.fillna(0)
    condition = ds.pcode == c['pcode']
    for var in new_vars:
        # https://docs.xarray.dev/en/stable/generated/xarray.DataArray.where.html
        # Value to use for locations in this object where cond is _False_.
        replacement = ds_ERA5[var].interp(time=ds.time.where(condition),longitude=ds.lon.where(condition),latitude=ds.lat.where(condition)) # here where selection does not drop
        replacement = replacement.fillna(0) # fill nan with 0 (shouldn't change results) 
        ds[var] = ds[var].where(~condition, replacement.values) # use values means we move from dask to numpy

# Saving the new data set with wave information        
ds.to_netcdf('../data/Processed/psd_wave.nc')