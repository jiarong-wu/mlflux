''' Functions used for GOTM data file operations. '''
import pandas as pd
import numpy as np
import torch
from mlflux.predictor import FluxANNs, Fluxdiff
import xarray as xr
import gc
from tqdm import tqdm
from scipy.interpolate import interp1d
from mlflux.eval import open_case
from dask.diagnostics import ProgressBar

from scipy.interpolate import interp1d
import gsw
import os

''' Read outputs '''
from io import StringIO
def read (filename, n1, n2, start_date):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    # Exclude the last two lines
    lines_to_keep = lines[:-2]
    # Join valid lines into a single string with newline characters
    valid_data = "\n".join(lines_to_keep)
    df = pd.read_csv(StringIO(valid_data), sep='\s+', header=None, 
                     names=['t','z','ux','uy','T','S','nn','nu','taux','tauy','Q','swr'])
    
    # By default, reshape uses row-major (C-style) order, meaning it fills the array row by row.
    time = pd.to_timedelta(df['t'][::n2], unit='s') # from second to hour 
    datetime = start_date + time
    depth = df['z'][:n2] # depth are 1 m apart
    
    ux = np.reshape(df['ux'],(n1,n2))
    uy = np.reshape(df['uy'],(n1,n2))
    T = np.reshape(df['T'],(n1,n2))
    S = np.reshape(df['S'],(n1,n2))
    taux = np.reshape(df['taux'],(n1,n2))[:,0]
    tauy = np.reshape(df['tauy'],(n1,n2))[:,0]
    Q = np.reshape(df['Q'],(n1,n2))[:,0]
    swr = np.reshape(df['swr'],(n1,n2))
    nn = np.reshape(df['nn'],(n1,n2))
    nu = np.reshape(df['nu'],(n1,n2))

    xrdf = xr.Dataset(
        {'ux':(['t','z'], ux),
        'uy':(['t','z'], uy),
        'T':(['t','z'], T),
        'S':(['t','z'], S),
        'swr':(['t','z'], swr),
        'nn':(['t','z'], nn), 
        'nu':(['t','z'], nu),
        'taux':(['t'], taux),
        'tauy':(['t'], tauy),
        'Q':(['t'], Q)}, 
        coords={
            't': datetime,
            'z': depth
        })

    del(df)
    gc.collect()
    
    return xrdf

''' Read outputs of vertical profiles from monthly restarted GOTM outputs. 
    Only need years because we need to put a time stamp. ''' 
def read_monthly (filename_, year, n1, n2, DELETE=False):
    ''' Auguments:
        filename: file name with placeholder for year and month
        n1: number of data points in time
        n2: number of depth points
    '''
    # days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ds_months = []

    for i,month in enumerate(range(1,13)):
        filename = filename_ %month
        print(filename)
        start_date = pd.Timestamp(year=year, month=month, day=1)
        ds = read(filename, n1, n2, start_date)
        ds = ds.where(ds.t.dt.month==month, drop=True)
        ds_months.append(ds)
        if DELETE:
            os.remove(filename)
    
    ds_full = xr.concat(ds_months, dim="t")
    return ds_full

''' interpolate to find where the depth is that any quantity profile is of value q '''
def interp(zprofile, qprofile, q):
    fz = interp1d(qprofile, zprofile, kind='linear', fill_value='extrapolate')
    z = fz(q)  
    return z

''' Compute the mixed layer depth. 
    Temperature and salinity need to be names S and T respectively. '''
def compute_MLD (ds, criterion='density'):
    ds['rho_p'] = gsw.density.rho(ds.S, ds.T, 0).compute() # potential density
    if criterion == 'temperature':
        diff = 0.2
        result = xr.apply_ufunc(
            interp,
            ds['z']*ds['T']/ds['T'],
            ds['T'],
            ds['T'].isel(z=-1) - diff,
            input_core_dims=[['z'], ['z'], []],  # Specify input dimensions (not to be broadcasted)
            output_core_dims=[[]],               # Specify output dimensions
            vectorize=True,                                
            dask='allowed'                       
        )
        
    elif criterion == 'density':
        diff = 0.1 # 0.03 or 0.1???
        result = xr.apply_ufunc(
            interp,
            ds['z']*ds['rho_p']/ds['rho_p'],
            ds['rho_p'],
            ds['rho_p'].isel(z=-1) + diff,
            input_core_dims=[['z'], ['z'], []],  # Specify input dimensions (not to be broadcasted)
            output_core_dims=[[]],               # Specify output dimensions
            vectorize=True,                                
            dask='allowed'                       
        )
        
    ds['MLD'] = result.compute()
    return ds

''' Read observations of profiles. '''
def process_file(file_path, zgrid):

    var_full = []
    t = []
    count = 0 # only used when a finite number of lines is desired
    MAX = 1e+30
    
    with open(file_path, 'r') as file:
        while count < MAX:
            # Read the main line (containing time, num1, num2)
            main_line = file.readline()
            if not main_line:  # Stop if we reach the end of the file
                break
            
            # Split the main line to extract the time and the two numbers
            parts = main_line.split()
            timestamp = pd.to_datetime(f"{parts[0]} {parts[1]}")  # Parse the date and time
            num1 = int(parts[2])  # Number of lines to read next
            num2 = int(parts[3])  # Not used directly here, but can be stored if needed

            # Read the next `num1` lines and store them in an array
            zz = []
            varz = []
            for _ in range(num1):
                data_line = file.readline()
                if not data_line:  # Stop if we reach the end of the file early
                    break
                z, var = data_line.split()
                zz.append(float(z)); varz.append(float(var))
                
            f = interp1d(np.array(zz), np.array(varz), kind='linear', fill_value='extrapolate')
            vargrid = f(zgrid)   
            var_full.append(vargrid)
            t.append(timestamp)

            count += 1         

        return t, var_full

''' Read multiple files and return pandas dataframe. '''

def read_vars (path, files, datetimeformat='%Y/%m/%d %H:%M:%S'):
    
    for i,file in enumerate(files):
        data_file = path + file['filename'] # the one used in basilisk, from 1961 to 1962
        df = pd.read_csv(data_file, sep='\s+', header=None, 
                          names=['date', 'time']+file['columns'])
        # Combine the 'date' and 'time' columns into a single 'datetime' column
        df['datetime'] = df['date'] + ' ' + df['time']
        # Convert 'datetime' column to datetime format
        df['datetime'] = pd.to_datetime(df['datetime'], format=datetimeformat)
        # Drop the separate 'date' and 'time' columns if they are no longer needed
        df.drop(columns=['date', 'time'], inplace=True)
        # Ensure 'datetime' column is in the correct format
        df['datetime'] = pd.to_datetime(df['datetime'], format=datetimeformat)

        if i == 0:
            df_ = df[['datetime']+file['columns']] # exchange order
        else:
            df_ = df_.merge(df, on='datetime')
            
    return df_

''' The following files are hourly state variables from 2010 to 2020 (provided by GOTM github). 
    Give the path where they are stored.'''

from mlflux.utils import rhcalc
def read2010 (path, datetimeformat='%Y/%m/%d %H:%M:%S'):

    file_sst = {'filename':'sst_hourly.dat', 'columns':['sst']} # in degree C
    file_wind = {'filename':'u10.dat', 'columns':['ux','uy']} # in m/s
    file_tair = {'filename':'airt.dat', 'columns':['t']} # in degree C
    file_tp = {'filename':'airp.dat', 'columns':['p']} # in Pa
    file_hum = {'filename':'hum.dat', 'columns':['q']} # in kg/kg
    
    file_tau = {'filename':'momentum_flux_papa.dat', 'columns':['taux','tauy']} 
    file_Q = {'filename':'heat_flux_papa.dat', 'columns':['Q']}  # this is the total Q with qh+ql+lwr
    file_swr = {'filename':'swr_papa.dat', 'columns':['swr']}
    file_lwr = {'filename':'lwr.dat', 'columns':['lwr']}
    
    files = [file_sst, file_wind, file_tair, file_tp, file_hum, file_tau, file_Q, file_swr, file_lwr]
    df = read_vars (path, files, datetimeformat)

    # Some additional fields
    df['U'] = (df.ux**2 + df.uy**2)**0.5
    df['rh'] = rhcalc(df.t,df.p/100.,df.q) # millibar to pascal
    df['cos'] = df.ux/df.U
    df['sin'] = df.uy/df.U  
    
    return df


''' Given a dataset containing time series of input variables, predict time series of 
    mean and variance. This needs to be better factored to work with more ANNs. 
    # TODO: make it compatible with ANNs of four outputs
    # TODO: add convariance
'''

# TODO: figure out how to make this function know about Fluxdiff
from mlflux.predictor import ensem_predict

''' An ad-hoc function for reading in an ensemble of ANNs. 
    N: number of ensemble member 
    filename: the shared part of saved model name
    X: inputs (should already be in torch tensor of size Nsample*Nfeature '''

def Q (ds, model_SH, model_LH):
    input_keys = model_SH.config['ikeys']
    X = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in input_keys]).astype('float32'))    
    mean = model_SH.pred_mean(X).detach().numpy().squeeze()
    std =  model_SH.pred_var(X).detach().numpy().squeeze() ** 0.5
    ds['qh_ann'] = mean*ds.Q/ds.Q
    ds['qh_std'] = std*ds.Q/ds.Q

    input_keys = model_LH.config['ikeys']
    X = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in input_keys]).astype('float32'))
    mean = model_LH.pred_mean(X).detach().numpy().squeeze()
    std =  model_LH.pred_var(X).detach().numpy().squeeze() ** 0.5
    ds['ql_ann'] = mean*ds.Q/ds.Q
    ds['ql_std'] = std*ds.Q/ds.Q

    return ds

def tau (ds, model_M):
    input_keys = model_M.config['ikeys']
    X = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in input_keys]).astype('float32'))    
    mean = model_M.pred_mean(X).detach().numpy().squeeze()
    std =  model_M.pred_var(X).detach().numpy().squeeze() ** 0.5
    ds['taux_ann'] = mean*ds.cos
    ds['tauy_ann'] = mean*ds.sin
    ds['taux_std'] = abs(std*ds.cos) # is this valid
    ds['tauy_std'] = abs(std*ds.sin) # is this valid
    return ds
    
def predict (ds, SHmodel_dir, LHmodel_dir, Mmodel_dir, rand_seed):
    model_name = 'model_rand%g.p' %rand_seed
    SHmodel = open_case (SHmodel_dir, model_name)  
    LHmodel = open_case (LHmodel_dir, model_name)
    Mmodel = open_case (LHmodel_dir, model_name)

    print ('Predicting fluxes and stds based on ANNs in \n SH directory ' + SHmodel_dir + 
           '\n LH directory ' + LHmodel_dir + 
           '\n M directory ' + Mmodel_dir + ' ...')
    ds = Q(ds, SHmodel, LHmodel)
    ds = tau(ds, Mmodel)

    print ('Finished!')
    return ds

''' General function to generate an ensemble of stochastic red noise by auto-regressive process.
    When ENSEM=1, it's one instance
    Aruguments:
        ENSEM: number of ensemble members
        N: length of series 
        alpha: coefficient for auto-regressive process 
        std: standard deviation. Can be an array of the same size as N
    Retun:
        eps_ensem: list of dimension ENSEM*N
'''

def gen_epsilon (ENSEM, N, alpha, std):

    eps_ensem = []
    
    for j in range(0, ENSEM):
        epsilon = np.zeros(N)
        epsilon[0] = np.random.normal(loc=0, scale=std[0])    
        for i in range(1, N):
            epsilon[i] = alpha*epsilon[i-1] + (1-alpha**2)**0.5*np.random.normal(loc=0, scale=std[i])        
        eps_ensem.append(epsilon)

    return eps_ensem
    
''' Generate stochastic perturbation for particular fluxes.
    Arguments:
        T: correlation time for a particular flux, may be different for heat and momentum
        dt: time stepping of flux input, by default 3 hrs
        ENSEM: number of ensemble members
    Returns: 
        eps_ensem: numpy array of dimension ENSEM*N 
'''
def gen_epsilon_flux (ds, FLUX='heat', T=50, dt=3, ENSEM=100):
    print (f'Generating an ensemble of {FLUX} flux. Size=%g.' %ENSEM)
    alpha = 1 - dt/T
        
    if FLUX == 'heat':
        interval = (ds.qh_std**2 + ds.ql_std**2)**0.5
        mean = ds.qh_ann + ds.ql_ann
        
    elif FLUX == 'taux':
        interval = ds.taux_std
        mean = ds.taux_ann 

    elif FLUX == 'tauy':
        interval = ds.tauy_std
        mean = ds.tauy_ann 

    eps_ensem = gen_epsilon (ENSEM=ENSEM, N=ds.sizes['datetime'], alpha=alpha, std=interval)
    eps_ensem = np.array(eps_ensem)
    print ('Finished! eps_ensem array shape: ' + str(eps_ensem.shape))
    return eps_ensem
    
''' Write stochstic ensemble flux as well as mean flux to specified floder using GOTM format. '''

def write_datetime (output_file, datetime, values):
    ''' output_file: output file path
        datatime: object of pandas datetime format
        values: a value array to write '''
    # Write the datetime and values to the file, row by row
    with open(output_file, 'w') as file:
        for t, val_row in zip(datetime, values):
            datetime_string = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S')
            values_string = '\t'.join(f"{val:.8f}" for val in val_row)
            file.write(f"{datetime_string}\t{values_string}\n")
    print('Finish writing to ' + output_file)
    
def write_stoch_flux (path, datetime, mean, eps_ensem, bulk, prefix='heatflux_ann'):
    ENSEM = eps_ensem.shape[0]

    # write the ensembles
    for i in range(0,ENSEM):
        output_file = path + prefix + 'ann_ensem%g.dat' %(i+1)
        flux = mean + eps_ensem[i]
        write_datetime (output_file, datetime, flux)  
                
    # Write ANN predicted mean
    output_file = path + prefix + 'ann_mean.dat' 
    flux = mean
    write_datetime (output_file, datetime, flux)
            
    # Write the finite ensemble mean (this should be close to but slightly different from ANN predicted mean)        
    output_file = path + prefix + 'ann_ensem_mean.dat' 
    eps_mean = eps_ensem.mean(axis=0)
    flux = mean + eps_mean
    write_datetime (output_file, datetime, flux)

    # write bulk flux
    output_file = path + prefix + 'bulk.dat'
    flux = bulk
    write_datetime (output_file, datetime, flux)



def read_ensem (filename, ENSEM=10, n1=361, n2=200, 
                start_date=pd.Timestamp('2012-03-21')):
    ''' Auguments:
        filename: shared filename
        ENSEM: number of ensemble member
        n1: number of time steps
        n2: number of depth points
    '''
    datasets = []
    for i in tqdm(range(ENSEM)):
        filename_ = filename + '%d' %(i+1)
        ds = read(filename_, n1, n2, start_date)
        ds = ds.assign_coords(ensem=(i+1))
        datasets.append(ds)       
    ds_ensem = xr.concat(datasets, dim='ensem')       
    return ds_ensem




def read_fluxes_old (heat_data_file='heatflux.dat', momentum_data_file='momentumflux.dat',
                 datetimeformat='%Y/%m/%d %H:%M:%S'):

    # Use read_csv to read the file
    df1 = pd.read_csv(heat_data_file, sep='\s+', header=None, names=['date', 'time', 'Q'])
    # Combine the 'date' and 'time' columns into a single 'datetime' column
    df1['datetime'] = df1['date'] + ' ' + df1['time']
    # Convert 'datetime' column to datetime format
    df1['datetime'] = pd.to_datetime(df1['datetime'], format=datetimeformat)
    # Drop the separate 'date' and 'time' columns if they are no longer needed
    df1.drop(columns=['date', 'time'], inplace=True)
    # Combine the date and time columns if necessary
    # Ensure 'datetime' column is in the correct format
    df1['datetime'] = pd.to_datetime(df1['datetime'], format=datetimeformat)
    
    df2 = pd.read_csv(momentum_data_file, sep='\s+', header=None, names=['date', 'time', 'taux', 'tauy'])
    df2['datetime'] = df2['date'] + ' ' + df2['time']
    df2['datetime'] = pd.to_datetime(df2['datetime'], format=datetimeformat)
    df2.drop(columns=['date', 'time'], inplace=True)
    df2['datetime'] = pd.to_datetime(df2['datetime'], format=datetimeformat)

    df = pd.merge(df1, df2, on='datetime')
    df = df[['datetime','Q','taux','tauy']]
    
    xrdf = xr.Dataset(
    {'taux': (['t'], df.taux),
    'tauy': (['t'], df.tauy),
    'Q': (['t'], df.Q)},
    coords={
        "t": df.datetime,
    },)
    
    return xrdf

    


    