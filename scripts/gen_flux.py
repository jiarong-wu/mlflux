''' Generate flux from ANN. Use in combination with flux.sh '''

import argparse
import os
import xarray as xr
import numpy as np
from mlflux.gotm import read2010, predict, gen_epsilon_flux, write_stoch_flux
from aerobulk.flux import noskin_np

# REMEMBER TO CHANGE ACCORDINGLY
SHmodel_dir = '/scratch/jw8736/mlflux/saved_model/final/SH5_1/NW_tr2/'
LHmodel_dir = '/scratch/jw8736/mlflux/saved_model/final/LH5_1/NW_tr2/'
Mmodel_dir = '/scratch/jw8736/mlflux/saved_model/final/M5_1/NW_tr2/'
rand = 4

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate an ensemble of flux (heat or momentum) based on ANN prediction and write to specified folder using GOTM data file format.")
    parser.add_argument('--sd', type=str, required=True, help="Start date in format YYYY-MM-DD")
    parser.add_argument('--ed', type=str, required=True, help="End date in format YYYY-MM-DD")
    parser.add_argument('--corrtime', type=float, default=60, help="Correlation time in hours")
    parser.add_argument('--dt', type=float, default=3, help="Flux time stepping interval in hours")
    parser.add_argument('--ensem', type=int, default=10, help="Number of ensemble members")
    parser.add_argument('--flux', type=str, default='heat', help="Flux to write. Options: \'heat\' or \'momentum\'") 
    parser.add_argument('--input_folder', '-i', type=str, required=True, help="Path to state variable holder")
    parser.add_argument('--output_folder', '-o', type=str, required=True, help="Parent path to write flux")

    # Parse the arguments
    args = parser.parse_args()
                        
    # Path
    # input_folder = '/home/jw8736/code-5.2.1/cases/ows_papa/'
    # output_folder = '/home/jw8736/test-gotm/ensem/' 
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Read and interpolate to hourly and then coarsen according to dt
    df = read2010(input_folder, datetimeformat='%Y-%m-%d %H:%M:%S')    
    df_ = df.set_index('datetime')
    ds = xr.Dataset.from_dataframe(df_)
    ds_uniform = ds.resample(datetime='H').interpolate('linear') # Interpolation non-uniform to hourly
    ds = ds_uniform.sel(datetime=slice(args.sd,args.ed))
    print ('Coarsening/Resampling inputs from hourly to %d hourly!' %args.dt) # Coarsening to dt
    dt_str = str(int(args.dt)) + 'H' 
    ds = ds.resample(datetime=dt_str).mean() 

    # These are artificially assigned, for models that use these features
    ds['zu'] = 10*ds['U']/ds['U']
    ds['zt'] = 10*ds['U']/ds['U']
    ds['zq'] = 10*ds['U']/ds['U']
    ds = ds.rename({'t': 'tair', 'sst': 'tsea', 'q':'qair'})
    
    # Predict by ANNs 
    from mlflux.predictor import FluxANNs
    ds = predict(ds, SHmodel_dir, LHmodel_dir, Mmodel_dir, rand)

    # Write to file
    output_path = output_folder + f'{args.sd}_{args.ed}/'
    print(output_path)
    os.system(f'mkdir -p {output_path}')

    if args.flux == 'heat':
        Q_eps_ensem = gen_epsilon_flux (ds, FLUX='heat', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        mean = (ds.qh_ann.values + ds.ql_ann.values + ds.lwr.values).reshape(-1, 1)
        eps_ensem = Q_eps_ensem.reshape(args.ensem, -1, 1) # of shape ensem*time*number_of_quantities_per_row
        # Old version: take bulk from GOTM
        # bulk = ds.Q.values.reshape(-1,1) 
        # New version: compute bulk assuming certain height
        ql, qh, taux, tauy, evap = noskin_np(sst=ds.tsea.to_numpy()+273.15, t_zt=ds.tair.to_numpy()+273.15, 
            hum_zt=ds.qair.to_numpy(), u_zu=ds.ux.to_numpy(), 
            v_zu=ds.uy.to_numpy(), slp=ds.p.to_numpy(), 
            algo='coare3p6', zt=18., zu=15., niter=6, input_range_check=True)
        print(ql.shape)
        bulk = (ql + qh + ds.lwr.values).reshape(-1,1)
        write_stoch_flux (path=output_path, datetime=ds.datetime.values,
                          mean=mean, eps_ensem=eps_ensem, bulk=bulk, prefix='heatflux_')

    if args.flux == 'momentum':
        taux_eps_ensem = gen_epsilon_flux (ds, FLUX='taux', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        tauy_eps_ensem = gen_epsilon_flux (ds, FLUX='tauy', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        mean1 = ds.taux_ann.values
        mean2 = ds.tauy_ann.values
        mean = np.concatenate((mean1[..., np.newaxis], mean2[..., np.newaxis]), axis=-1)
        eps_ensem = np.concatenate((taux_eps_ensem[..., np.newaxis], tauy_eps_ensem[..., np.newaxis]), axis=-1)  
        # bulk1 = ds.taux.values
        # bulk2 = ds.tauy.values
        # bulk = np.concatenate((bulk1[..., np.newaxis], bulk2[..., np.newaxis]), axis=-1)
        ql, qh, taux, tauy, evap = noskin_np(sst=ds.tsea.to_numpy()+273.15, t_zt=ds.tair.to_numpy()+273.15, 
            hum_zt=ds.qair.to_numpy(), u_zu=ds.ux.to_numpy(), 
            v_zu=ds.uy.to_numpy(), slp=ds.p.to_numpy(), 
            algo='coare3p6', zt=18., zu=15., niter=6, input_range_check=True)
        print(taux.shape)
        bulk = np.concatenate((taux[..., np.newaxis], tauy[..., np.newaxis]), axis=-1)
        write_stoch_flux (path=output_path, datetime=ds.datetime.values, 
                          mean=mean, eps_ensem=eps_ensem, bulk=bulk, prefix='momentumflux_') 

    # Some optional moving data around
    # os.system(f'cp {output_folder}shared/* {output_path}')

