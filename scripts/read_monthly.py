''' Read profiles from monthly restarted GOTM runs and save to netcdf files (separated by years and by ensem label). '''

from mlflux.utils import save_ds_compressed
from mlflux.gotm import read_monthly
import numpy as np


if __name__ == "__main__":
    masterdir = '/scratch/jw8736/gotm/ensem/2011-01-01_2020-01-01_heat/'
    method = 'kepsilon' # {'kepsilon', 'kpp'}
    MINUTE = 10 # dt
    OUTMINUTE = 60 # outp ut dt
    n2 = 200 # number of vertical points
    n1 = 31 # run for 31 days regardless of months
    n1_ = int(n1*24*60/OUTMINUTE) + 1 # depends on output frequency
    nensem = 20 # number of ensemble members
    
    ylist = [2011,2012,2015,2016] # the years that we ran
    ENSEM = np.arange(1,nensem) # tags for ensembles
    for year in ylist:
        folder = masterdir +  f'out_{method}_dt{MINUTE}_{year}/' # directory e.g. out_kpp_dt60_2011
        for i in ENSEM:
            filename_start = 'out'
            filename_end = 'ensem%g' % i
            filename = folder + filename_start + f'_%g_' + filename_end # file name e.g. out_1_ensem9 (leave place holder for month)
            ds = read_monthly (filename, year, n1_, n2, DELETE=False)
            netcdf_name = folder + filename_start + f'_ensem%g.nc' %i
            save_ds_compressed(ds, netcdf_name)
        # Deterministic runs
        for filename_start in ['out_ann_mean', 'out_ensem_mean', 'out_bulk']:
            filename = folder + filename_start + f'_%g' # filename e.g. out_ensem_mean_1, out_ann_mean_1, out_bulk_1
            ds = read_monthly (filename, year, n1_, n2, DELETE=False)
            netcdf_name = folder + filename_start + f'.nc'
            save_ds_compressed(ds, netcdf_name)


    ####### Reading single year without ensemble runs ##########
    # folder = '/scratch/jw8736/gotm/test/'
    # MINUTE = 10
    # n2 = 200 # number of vertical points
    # n1 = 31 # run for 31 days regardless of months
    # n1_ = int(n1*24*60/MINUTE) + 1 # depends on output frequency    
    # method = 'kpp'
    
    # f_start = 'out_test_'
    # f_end = 'dt%g' % MINUTE
    # year = 2012
    
    # filename = folder + f_start + f'{method}_%g_%g_' + f_end
    # ds = read_monthly (filename, year, n1_, n2, DELETE=False)
    # netcdf_name = folder + f'{method}_%g_dt%g.nc' %(year, MINUTE)
    # save_ds_compressed(ds, netcdf_name) 
