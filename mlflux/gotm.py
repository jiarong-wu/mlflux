''' Functions used for GOTM data file operations. '''
import pandas as pd

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