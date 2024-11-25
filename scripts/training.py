import argparse
import sys
import json
import torch
import numpy as np
import xarray as xr
from mlflux.datafunc import data_split_psd_rand
from mlflux.ann import RealFluxDataset
from mlflux.predictor import FluxANNs

# Is this needed? Predictor with four inputs and four outputs, and with fixed first layer taking temperature difference. ###
# from mlflux.ann import ANNdiff  
# class Fluxdiff(FluxANNs):
#     def __init__(self,params={}):
#         super().__init__(params)
#         self.mean_func = ANNdiff(**self.mean_ann_para)
#         self.var_func = ANNdiff(**self.var_ann_para)  


################
# Design space:
# DATA:
#   WEIGHT: weighted or not
# NETWORK:
#   ann_size: a list of neurons for each layer
#   var_activation: ['no','square','exponential','softplus'] 
#   ikeys: ['U','tsea','tair','rh','qair','zu','zt','zq','p'] and its subset
#   okeys: ['taucx', 'hsc', 'hlc'] together or separate (and corresponding ['taubx', 'hsb', 'hlb'])
#   compute_norm: True if specify manually 
# TRAINING:
#   two_steps: if training mean first
#   training_paras: a dictionary
################

def train_model (config, path, rand_seed):
    datapath = config['datapath'] # 'home/jw8736/mlflux/data/Processed/'
    datafile = config['datafile'] # 'psd_coare3p0_weight1_wave0.nc'
    ann_size = config['ann_size'] # [64,32,16]
    mean_activation = config['mean_activation'] # 'no'
    var_activation = config['var_activation'] # 'exponential'
    dropout_rate = config['dropout_rate'] # 0.5 (-1 for no dropout)
    ikeys = config['ikeys'] # ['U','tsea','tair','rh']
    okeys = config['okeys'] # ['taucx']
    bkeys = config['bkeys'] # ['taubx']
    RATIO = config['RATIO'] # ratio of validation/total, default 20%
    WEIGHT = config['WEIGHT'] # True: use importance weighting
    RESIDUAL = config['RESIDUAL'] # True: optional training on residual, TODO: implement
    training_paras = config['training_paras']
    compute_norm = config['compute_norm'] # True: compute (weighted norm) from data; False: read the norms below
    if not compute_norm:
        norms = config['norms'] # only used if compute_norm is False, need to match ikeys and okeys   
    two_steps = config['two_steps'] # True: first train mean model according to mse. 
    VERBOSE = config['VERBOSE'] # True: print out loss

    ###### Load data #######
    ds = xr.load_dataset(datapath + datafile)
    if WEIGHT:
        if 'weight' not in ds.data_vars:
            print(f"Weights don't exist! Check config file.")
            sys.exit(1)
    else:
        ds['weight'] = ds['U']/ds['U'] # manually set to 1 

    ###### Put together network parameters #######
    para_mean = {'n_in':len(ikeys),'n_out':len(okeys),'hidden_channels':ann_size,'ACTIVATION':mean_activation,'dropout_rate':dropout_rate}
    para_var = {'n_in':len(ikeys),'n_out':len(okeys),'hidden_channels':ann_size,'ACTIVATION':var_activation,'dropout_rate':dropout_rate}

    ###### Assemble dataloader #######
    training_ds, validating_ds, testing_ds = data_split_psd_rand(ds, seed=rand_seed, ratio=RATIO)
    training_data = RealFluxDataset(training_ds, input_keys=ikeys, output_keys=okeys, bulk_keys=bkeys)
    validating_data = RealFluxDataset(validating_ds, input_keys=ikeys, output_keys=okeys, bulk_keys=bkeys)
    testing_data = RealFluxDataset(testing_ds, input_keys=ikeys, output_keys=okeys, bulk_keys=bkeys)
    ###### Initialize model weights ######
    model = FluxANNs({'mean_ann_para':para_mean, 'var_ann_para':para_var})
    ###### Compute or read scales #######
    if compute_norm:
        xmean_value = (training_data.X*training_data.W).mean(dim=0) # scales also need to be weighted
        xscale_value = ((training_data.X - xmean_value)**2*training_data.W).mean(dim=0)**0.5            
        ymean_value = (training_data.Y*training_data.W).mean(dim=0) # scales also need to be weighted
        yscale_value = ((training_data.Y - ymean_value)**2*training_data.W).mean(dim=0)**0.5
        model.Xscale = {'mean':xmean_value.reshape(1,-1),
                        'scale':xscale_value.reshape(1,-1)}          
        model.Yscale = {'mean':ymean_value.reshape(1,-1),
                        'scale':yscale_value.reshape(1,-1)}
    else:   
        if (len(norms['imean']) == len(ikeys)) and (len(norms['omean']) == len(okeys)):
            model.Xscale = {'mean':torch.tensor(norms['imean'].reshape(1,-1).astype('float32')),
                            'scale':torch.tensor(norms['iscale'].reshape(1,-1).astype('float32'))}      
            model.Yscale = {'mean':torch.tensor(norms['omean'].reshape(1,-1).astype('float32')),
                            'scale':torch.tensor(norms['oscale'].reshape(1,-1).astype('float32'))}
        else:
            print(f"Read in normalization does not match network size!")
            sys.exit(1)                           
        
    log = model.fit(training_data, validating_data, training_paras, VERBOSE=VERBOSE, TWOSTEPS=two_steps)
    model.save(fname=path + "model_rand%g" %rand_seed)                                     
         
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training ANNs. Read yaml from a directory and save models to the same directory.')
    parser.add_argument('--path', type=str, help='Directory of config file.')
    parser.add_argument('--rand', type=int, help='Random seeding (for data split).')
    args = parser.parse_args()
    
    with open(args.path + 'config.json', 'r') as f:
        config = json.load(f)

    if not config['compute_norm']:
        if 'norms' in config:
            config['norms'] = {key: np.array(value) for key, value in config['norms'].items()}

    print(config)
    model = train_model (config, args.path, args.rand)



        
