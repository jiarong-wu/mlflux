''' Functions used for GOTM data file operations. '''
import pandas as pd
import numpy as np
import torch
from mlflux.predictor import Fluxdiff

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



def predict (ds):
    model_dir = '/home/jw8736/mlflux/saved_model/one_output_anns/'  
    # Assemble input X
    input_keys = ['U','sst','t','rh','q']
    X = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in input_keys]).astype('float32'))

    # Three fluxes if they are separate ANNs
    # TODO: make it compatible with ANNs of four outputs
    model_names = ['Flux51_momentum_3layers_split','Flux51_sensible_3layers_split', 'Flux51_latent_3layers_split']
    print ('Predicting fluxes and stds based on ANNs in directory ' + model_dir + '...')
    
    for i, model_name in enumerate(model_names):
        name = model_dir + model_name
        mean, std, Sigma_mean, Sigma_std = ensem_predict(X=X, N=6, modelname=name)
        mean = mean.squeeze(); std = std.squeeze(); 
        Sigma_mean = Sigma_mean.squeeze(); Sigma_std = Sigma_std.squeeze()
        if i == 0: # Momentum
            ds['taux_ann'] = mean*ds.cos
            ds['tauy_ann'] = mean*ds.sin
            ds['taux_ann_sigma'] = abs(Sigma_mean*ds.cos) # is this valid
            ds['tauy_ann_sigma'] = abs(Sigma_mean*ds.sin) # is this valid
        elif i == 1: # Sensible heat
            ds['qh_ann'] = mean*ds.Q/ds.Q
            ds['qh_ann_sigma'] = Sigma_mean*ds.Q/ds.Q 
        elif i == 2: # Latent heat 
            ds['ql_ann'] = mean*ds.Q/ds.Q
            ds['ql_ann_sigma'] = Sigma_mean*ds.Q/ds.Q 

    ds['Q_ann'] = ds.qh_ann + ds.ql_ann + ds.lwr
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
        interval = (ds.qh_ann_sigma + ds.ql_ann_sigma)**0.5
        mean = ds.qh_ann + ds.ql_ann
        
    elif FLUX == 'taux':
        interval = ds.taux_ann_sigma
        mean = ds.taux_ann 

    elif FLUX == 'tauy':
        interval = ds.tauy_ann_sigma
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
    
def write_stoch_flux (path, datetime, mean, eps_ensem, prefix='heatflux_ann'):
    ENSEM = eps_ensem.shape[0]

    # write the ensembles
    for i in range(0,ENSEM):
        output_file = path + prefix + '_ensem%g.dat' %(i+1)
        flux = mean + eps_ensem[i]
        write_datetime (output_file, datetime, flux)  
                
    # Write ANN predicted mean
    output_file = path + prefix + '_mean.dat' 
    flux = mean
    write_datetime (output_file, datetime, flux)
            
    # Write the finite ensemble mean (this should be close to but slightly different from ANN predicted mean)        
    output_file = path + prefix + '_ensem_mean.dat' 
    eps_mean = eps_ensem.mean(axis=0)
    flux = mean + eps_mean
    write_datetime (output_file, datetime, flux)


    


    