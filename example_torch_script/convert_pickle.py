
import pickle
import json
import os 
import torch
from mlflux.predictor import FluxANNs

def open_case (model_dir, model_name):
    with open(model_dir + 'config.json', 'r') as f:
        config = json.load(f)   
    filename = model_dir + model_name
    with open(filename, "rb") as input_file:
        model = pickle.load(input_file)   
    model.config = config 
    return model


def convert_and_save_model(old_model_file, dir_name):
    old_model = open_case(old_model_file) # this was a class not good for torch script
    os.makedirs(dir_name, exist_ok=True)

    # Don't use this that save scale and weights seperately, use the one that saves scale and weights together
    # torch.save(LH.Xscale, dir_name + "Xscale.pt")
    # torch.save(LH.Yscale, dir_name + "Yscale.pt")
    # torch.save(LH.mean_func.state_dict(), dir_name + "mean_weights.pt")
    # torch.save(LH.var_func.state_dict(), dir_name + "var_weights.pt")

    from flux_model import ANN_online

    ####### Convert mean network and save #######
    # Hard code configuration for now 
    n_in = 5
    n_out = 1
    hidden_channels = [32, 16] 
    activation = 'no' # 'exponential' for var
    mean_ann = ANN_online(n_in=n_in, n_out=n_out, hidden_channels=hidden_channels, ACTIVATION=activation, 
                        Xmean=old_model.Xscale['mean'], Xscale=old_model.Xscale['scale'], 
                        Ymean=old_model.Yscale['mean'], Yscale=old_model.Yscale['scale'])
    # This will miss ['Xmean', 'Xscale', 'Ymean', 'Yscale'] because they were not registered as buffer in the old class
    missing, unexpected = mean_ann.load_state_dict(old_model.mean_func.state_dict(), strict=False)
    torch.save(mean_ann.state_dict(), dir_name + "mean_weights.pt")
    # print(missing)
    # print(unexpected)

    ####### Convert var network and save #######
    n_in = 5
    n_out = 1
    hidden_channels = [32, 16] 
    activation = 'exponential' # 'exponential' for var
    var_ann = ANN_online(n_in=n_in, n_out=n_out, hidden_channels=hidden_channels, ACTIVATION=activation, 
                        Xmean=old_model.Xscale['mean'], Xscale=old_model.Xscale['scale'], 
                        Ymean=old_model.Yscale['mean'], Yscale=old_model.Yscale['scale'])
    # This will miss ['Xmean', 'Xscale', 'Ymean', 'Yscale'] because they were not registered as buffer in the old class
    missing, unexpected = var_ann.load_state_dict(old_model.var_func.state_dict(), strict=False)
    torch.save(var_ann.state_dict(), dir_name + "var_weights.pt")
    
if __name__ == "__main__":
    source_path = '/glade/work/jiarongw/mlflux/example_python/'
    target_path = '/glade/u/home/jiarongw/MLFLUX/example_torch_script/'
    # Apparently change M/SH/LH
    old_model_file = source_path + 'SH/', 'sh.p'
    dir_name = target_path + 'SH/'
    print("Converting SH model... from \\", old_model_file, " to \\", dir_name)
    convert_and_save_model(old_model_file, dir_name)

