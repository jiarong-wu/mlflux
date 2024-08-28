import argparse
import os
import xarray as xr
import numpy as np
from mlflux.gotm import read2010, predict, gen_epsilon_flux, write_stoch_flux

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

    if args.corrtime is None:
        args.corrtime = input("Please enter correlation time: ")


    # parser.add_argument('--stencil_size', type=int, default=3)
    # parser.add_argument('--hidden_layers', type=str, default='[20]')
    # parser.add_argument('--collocated', type=str, default='False')
    # parser.add_argument('--short_waves_dissipation', type=str, default='False')
    # parser.add_argument('--short_waves_zero', type=str, default='False')
    # parser.add_argument('--jacobian_trace', type=str, default='False')
    # parser.add_argument('--perturbed_inputs', type=str, default='False')
    # parser.add_argument('--grid_harmonic', type=str, default='plane_wave')
    # parser.add_argument('--jacobian_reduction', type=str, default='component')

    # args.factors = eval(args.factors)
                        
    # Path
    # input_folder = '/home/jw8736/code-5.2.1/cases/ows_papa/'
    # output_folder = '/home/jw8736/test-gotm/ensem/' 
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Read and interpolate to hourly
    df = read2010(input_folder, datetimeformat='%Y-%m-%d %H:%M:%S')    
    df_ = df.set_index('datetime')
    ds = xr.Dataset.from_dataframe(df_)
    ds_uniform = ds.resample(datetime='H').interpolate('linear') # Interpolation non-uniform to hourly
    # ds_hat = ds_uniform.resample(datetime='3H').mean() # Coarsening to 3-hourly
    ds = ds_uniform.sel(datetime=slice(args.sd,args.ed))
    
    # Predict by ANNs 
    from mlflux.predictor import Fluxdiff
    ds = predict(ds)

    output_path = output_folder + f'{args.sd}_{args.ed}/'
    print(output_path)
    os.system(f'mkdir -p {output_path}')

    if args.flux == 'heat':
        Q_eps_ensem = gen_epsilon_flux (ds, FLUX='heat', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        mean = ds.Q.values.reshape(-1,1) 
        eps_ensem = Q_eps_ensem.reshape(args.ensem, -1, 1) # of shape ensem*time*number_of_quantities_per_row
        write_stoch_flux (path=output_path, datetime=ds.datetime.values, mean=mean, eps_ensem=eps_ensem, prefix='heatflux_ann')

    if args.flux == 'momentum':
        taux_eps_ensem = gen_epsilon_flux (ds, FLUX='taux', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        tauy_eps_ensem = gen_epsilon_flux (ds, FLUX='tauy', T=args.corrtime, dt=args.dt, ENSEM=args.ensem)
        mean1 = ds.taux.values
        mean2 = ds.tauy.values
        mean = np.concatenate((mean1[..., np.newaxis], mean2[..., np.newaxis]), axis=-1)
        eps_ensem = np.concatenate((taux_eps_ensem[..., np.newaxis], tauy_eps_ensem[..., np.newaxis]), axis=-1)  
        write_stoch_flux (path=output_path, datetime=ds.datetime.values, mean=mean, eps_ensem=eps_ensem, prefix='momentumflux_ann') 

    
    os.system(f'cp {output_folder}shared/* {output_path}')

