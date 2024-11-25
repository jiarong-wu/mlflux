
from mlflux.utils import save_ds_compressed
from mlflux.gotm import read_monthly
from mlflux.gotm import compute_MLD
import numpy as np
import xarray as xr


''' Compute metrics with columns of month and row of combos, averaged over years '''

def open_ds(path, SUBSET=True):
    ''' SUBSET: only open deterministic cases. '''
    ds_bulk = xr.open_dataset(path+'out_bulk.nc')
    ds_ann_mean = xr.open_dataset(path+'out_ann_mean.nc')
    if SUBSET:
        return ds_bulk, ds_ann_mean
    else:
        ds_ensem_mean = xr.open_dataset(path+'out_ensem_mean.nc')
        ds = xr.open_mfdataset(path+f'out_ensem*.nc', combine='nested', concat_dim='ensem')
    return ds_bulk, ds_ann_mean, ds_ensem_mean, ds
    
class Distance():
    ''' Distance between two configs across years and months.
        Attributs:
            label: a string that identify the two configs
            MLD_diffs: 2D arrays of MLD distance of dimension N_years * N_months
            SST_diffs: 2D arrays of SST distance of dimension N_years * N_months
    '''
    def __init__ (self, years, combo, combo_ref, dir):
        self.years = years
        self.combo = combo
        self.combo_ref = combo_ref
        self.dir = dir

    def read_compute(self):
        print ('Compare between \n' + str(self.combo) + '\n' + 'and\n' + str(self.combo_ref))
        self.MLD_criterion = 'density'
        ''' REF: which combo to use as reference, default first on in the list. '''
        self.label = []; self.MLD_diffs = []; self.SST_diffs = []       
        MLD_diffs = []; SST_diffs = []; Q_diffs = [] # over years and then months
        for year in self.years:
            # First read ref numerical configuration
            if self.combo_ref['flux'] != 'Bulk':
                print('Reference case should be bulk formula!')
            else:
                path = self.dir + f"out_{self.combo_ref['method']}_dt{self.combo_ref['MINUTE']}_{year}/"
                ds_ref = xr.open_dataset(path+'out_bulk.nc')
            # Then read combo to compare
            path = self.dir + f"out_{self.combo['method']}_dt{self.combo['MINUTE']}_{year}/"
            if self.combo['flux'] == 'Bulk':
                ds = xr.open_dataset(path+'out_bulk.nc')
            elif self.combo['flux'] == 'ANN':
                ds = xr.open_dataset(path+'out_ann_mean.nc')

            ds_ref = compute_MLD(ds_ref, self.MLD_criterion)
            ds = compute_MLD(ds, self.MLD_criterion)                      
            label, MLD_diff, SST_diff = self.distance_metrics_monthly (ds, ds_ref, self.combo)
            Q_diff = self.compute_Q_diff (ds, ds_ref)
            
            self.label = label
            MLD_diffs.append(MLD_diff)
            SST_diffs.append(SST_diff)
            Q_diffs.append(Q_diff)

        self.MLD_diffs = np.array(MLD_diffs)
        self.SST_diffs = np.array(SST_diffs)
        self.Q_diffs = np.array(Q_diffs)

    def distance_metrics_monthly(self, ds, ds_ref, combo):
        # NOTICE: absolute sign!
        MLD_diff = abs(ds.MLD.resample(t='1M').mean()) - abs(ds_ref.MLD.resample(t='1M').mean())       
        SST_diff = ds.T.isel(z=-1).resample(t='1M').mean() - ds_ref.T.isel(z=-1).resample(t='1M').mean()
        label = f"{combo['flux']}-{combo['method']}-{combo['MINUTE']}min - REF"
        return label, MLD_diff.values, SST_diff.values 

    def compute_Q_diff(self, ds, ds_ref):
        Q_diff = ds.Q.resample(t='1M').mean() - ds_ref.Q.resample(t='1M').mean()  
        return Q_diff

    def checksigns(self, name):
        # target is a string, referring to a 2D array with axis1 = year and axis2 = month
        # we want to check if sign is consistent across years
        target = getattr(self, name)
        sign_agree = np.zeros(12)
        for month in range(12):
            first_sign = np.sign(target[0,month])
            sign_agree[month] = np.all(np.sign(target[:,month]) == first_sign)
        return sign_agree


class Stoch():
    ''' Properties of stochastic runs across years and months.
        Attributs:
            
    '''
    def __init__ (self, years, combo, dir):
        self.years = years
        self.combo = combo
        self.dir = dir
        
    def read_compute (self):
        self.MLD_criterion = 'density'
        MLD_diffs = []; MLD_stds = []; SST_diffs = []; SST_stds = []
        for year in self.years:
            print(year)
            path = self.dir + f"out_{self.combo['method']}_dt{self.combo['MINUTE']}_{year}/"
            ds_determ = xr.open_dataset(path+'out_ensem_mean.nc')
            ds = xr.open_mfdataset(path+f'out_ensem*.nc', combine='nested', concat_dim='ensem')

            ds_determ = compute_MLD(ds_determ, self.MLD_criterion)
            ds = compute_MLD(ds, self.MLD_criterion)
            print('Done computing MLD!')
        
            MLD_diff, MLD_std, SST_diff, SST_std = self.metrics_monthly(ds, ds_determ)
            MLD_diffs.append(MLD_diff); MLD_stds.append(MLD_std); SST_diffs.append(SST_diff); SST_stds.append(SST_std) 
            
        self.MLD_diffs = np.array(MLD_diffs)
        self.MLD_stds = np.array(MLD_stds)
        self.SST_diffs = np.array(SST_diffs)
        self.SST_stds = np.array(SST_stds)
        
    def metrics_monthly(self, ds, ds_determ):
        # NOTICE: absolute sign!
        MLD_diff = abs(ds.MLD.mean(dim='ensem').resample(t='1M').mean()) - abs(ds_determ.MLD.resample(t='1M').mean())  
        MLD_std = ds.MLD.std(dim='ensem').resample(t='1M').mean()
        SST_diff = ds.T.isel(z=-1).mean(dim='ensem').resample(t='1M').mean() - ds_determ.T.isel(z=-1).resample(t='1M').mean()
        SST_std = ds.T.isel(z=-1).std(dim='ensem').resample(t='1M').mean()
        return MLD_diff.values, MLD_std.values, SST_diff.values, SST_std.values 
        
    def checksigns(self, name):
        # target is a string, referring to a 2D array with axis1 = year and axis2 = month
        # we want to check if sign is consistent across years
        target = getattr(self, name)
        sign_agree = np.zeros(12)
        for month in range(12):
            first_sign = np.sign(target[0,month])
            sign_agree[month] = np.all(np.sign(target[:,month]) == first_sign)
        return sign_agree
        

# if __name__ == "__main__":
# ...