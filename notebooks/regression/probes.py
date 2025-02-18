import numpy as np
from matplotlib import pyplot as plt
import torch
from mlflux.ann import RealFluxDataset

def gen_grid_U_dT (model):
    U = np.linspace(0,30,201)  # 1D array for U
    Ta = np.linspace(0,20,101)  # 1D array for Ta
    To = np.ones(101)*10 # 1D array for To
    U_grid, Ta_grid = np.meshgrid(U, Ta)
    U_flat = U_grid.flatten().reshape(-1, 1)  # Flatten to 2D column
    Ta_flat = Ta_grid.flatten().reshape(-1, 1)  # Flatten to 2D column
    To_flat = np.ones_like(U_flat) * 10    
    zu_flat = np.ones_like(U_flat) * 10
    zt_flat = np.ones_like(U_flat) * 5
    RH_flat = np.ones_like(U_flat) * 80
    p_flat = np.ones_like(U_flat) * 101000
    # zu_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,5].numpy()
    # zt_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,6].numpy()
    # zq_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,7].numpy()
    if model.config['ikeys'] == ["U", "tsea", "tair", "zu", "zt"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,zu_flat,zt_flat]).astype('float32')
    elif model.config['ikeys'] == ["U", "tsea", "tair", "rh"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,RH_flat]).astype('float32')  
    elif model.config['ikeys'] == ["U", "tsea", "tair", "rh", "p"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,RH_flat,p_flat]).astype('float32')

    mean_pred = model.pred_mean(torch.tensor(X)).detach().numpy()
    mean_2D = mean_pred.reshape(101, 201)
    std_pred = model.pred_var(torch.tensor(X)).detach().numpy()**0.5
    std_2D = std_pred.reshape(101, 201)
    
    return X.reshape(101, 201, -1), mean_2D, std_2D

def gen_grid_U_RH (model):
    U = np.linspace(0,30,201)  # 1D array for U
    RH = np.linspace(50,110,101)
    U_grid, RH_grid = np.meshgrid(U, RH)
    U_flat = U_grid.flatten().reshape(-1, 1)  # Flatten to 2D column
    RH_flat = RH_grid.flatten().reshape(-1, 1)  # Flatten to 2D column
    To_flat = np.ones_like(U_flat) * 15    
    Ta_flat = np.ones_like(U_flat) * 15
    zu_flat = np.ones_like(U_flat) * 10
    zt_flat = np.ones_like(U_flat) * 5
    p_flat = np.ones_like(U_flat) * 101000
    # zu_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,5].numpy()
    # zt_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,6].numpy()
    # zq_flat = np.ones_like(U_flat) * model.Xscale['mean'][:,7].numpy()
    if model.config['ikeys'] == ["U", "tsea", "tair", "zu", "zt"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,zu_flat,zt_flat]).astype('float32')
    elif model.config['ikeys'] == ["U", "tsea", "tair", "rh"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,RH_flat]).astype('float32')  
    elif model.config['ikeys'] == ["U", "tsea", "tair", "rh", "p"]:
        X = np.hstack([U_flat,To_flat,Ta_flat,RH_flat,p_flat]).astype('float32')

    mean_pred = model.pred_mean(torch.tensor(X)).detach().numpy()
    mean_2D = mean_pred.reshape(101, 201)
    std_pred = model.pred_var(torch.tensor(X)).detach().numpy()**0.5
    std_2D = std_pred.reshape(101, 201)
    
    return X.reshape(101, 201, -1), mean_2D, std_2D

''' Plot the data density '''
from scipy.stats import gaussian_kde

def plot_data_density_U_dT (ax, ds, Xgrid, model):
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    # Weighted
    # w_index = np.random.choice(len(vd.X), 10000, p=vd.W.squeeze()/vd.W.sum())
    # kde = gaussian_kde(np.vstack([vd.X[w_index,0], vd.X[w_index,2]-vd.X[w_index,1]]))
    kde = gaussian_kde(np.vstack([vd.X[:,0], vd.X[:,2]-vd.X[:,1]]), weights=None)    

    # Not weighted
    U = Xgrid[:,:,0]; Tdiff = Xgrid[:,:,2] - Xgrid[:,:,1]
    zi = kde(np.vstack([U.flatten(), Tdiff.flatten()]))  # Evaluate KDE on the grid
    zi = zi.reshape(101, 201)*len(vd.X)  # Reshape to match grid, AND! convert density to counts
    levels = (10, 100)
    contour = ax.contour(U, Tdiff, zi, levels=levels, colors='gray')  # Filled contours
    ax.clabel(contour, inline=True, fontsize=10)
    return contour

def plot_data_density_U_RH (ax, ds, Xgrid, model):
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    # Weighted
    # w_index = np.random.choice(len(vd.X), 10000, p=vd.W.squeeze()/vd.W.sum())
    # kde = gaussian_kde(np.vstack([vd.X[w_index,0], vd.X[w_index,2]-vd.X[w_index,1]]))
    # kde = gaussian_kde(np.vstack([vd.X[:,0], vd.X[:,3]]), weights=vd.W[:,0])  
    kde = gaussian_kde(np.vstack([vd.X[:,0], vd.X[:,3]]), weights=None)   

    # Not weighted
    U = Xgrid[:,:,0]; RH = Xgrid[:,:,3]
    zi = kde(np.vstack([U.flatten(), RH.flatten()]))  # Evaluate KDE on the grid
    zi = zi.reshape(101, 201)*len(vd.X)  # Reshape to match grid, AND! convert density to counts
    levels = (10, 100)
    contour = ax.contour(U, RH, zi, levels=levels, colors='gray')  # Filled contours
    ax.clabel(contour, inline=True, fontsize=10)
    return contour