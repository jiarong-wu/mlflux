from utils import rhcalc
from ann import open_case
import numpy as np
import torch

abs_path = '/scratch/jw8736/mlflux/example/'

''' 
Inputs: 1D numpy arrays
    ux: Wind speed in x (m/s)
    uy: Wind speed in y (m/s)               
    Ta: Air temperature (celcius)
    To: Ocean temperature (celcius)
    p: Sea surface pressure (pascal), e.g. around 1e+6
    q: Specific humidity (kg/kg)
    
Outputs: 
    taux: Momentum flux in x (N/m^2)
    tauy: Momentum flux in y (N/m^2)
    Qs: Sensible heat flux (W/m^2)
    Ql: Latent heat flux (W/m^2)
'''

def ann_flux_xarray(ds):
    """
    Xarray wrapper for ann_flux.
    Expects ds to be an xarray.Dataset with variables:
        ux, uy, To, Ta, p, q
    Returns a new xarray.Dataset with taux, tauy, Qs, Ql.
    """
    ux = ds['ux'].values
    uy = ds['uy'].values
    To = ds['To'].values
    Ta = ds['Ta'].values
    p = ds['p'].values
    q = ds['q'].values

    taux, tauy, Qs, Ql = ann_flux(ux, uy, To, Ta, p, q)

    out_ds = ds.copy()
    out_ds['taux'] = (ds['ux'].dims, taux.reshape(ds['ux'].shape))
    out_ds['tauy'] = (ds['ux'].dims, tauy.reshape(ds['ux'].shape))
    out_ds['Qs'] = (ds['ux'].dims, Qs.reshape(ds['ux'].shape))
    out_ds['Ql'] = (ds['ux'].dims, Ql.reshape(ds['ux'].shape))
    return out_ds

def ann_flux (ux, uy, To, Ta, p, q):

    U = (ux**2 + uy**2)**0.5                   # Wind speed in m/s
    epsilon = 1e-8
    U_safe = np.where(U == 0, epsilon, U)
    cos = ux/U_safe
    sin = uy/U_safe                  
    rh = rhcalc(Ta, p/100. , q)                
    
    # Reshape into sample * features ["U","tsea","tair","rh","p"]
    X = np.hstack([U.reshape(-1,1), To.reshape(-1,1), Ta.reshape(-1,1), 
                   rh.reshape(-1,1), p.reshape(-1,1)]).astype('float32')
    X = torch.tensor(X)

    # Read models
    M = open_case (abs_path + 'M/', 'm.p')
    SH = open_case (abs_path + 'SH/', 'sh.p')
    LH = open_case (abs_path + 'LH/', 'lh.p')
    
    # Predict fluxes
    M_mean = M.pred_mean(X)
    M_std = M.pred_var(X) ** 0.5
    SH_mean = SH.pred_mean(X)
    SH_std = SH.pred_var(X) ** 0.5
    LH_mean = LH.pred_mean(X)
    LH_std = LH.pred_var(X) ** 0.5
    
    taux = M_mean.detach().numpy().squeeze() * cos.ravel()
    tauy = M_mean.detach().numpy().squeeze() * sin.ravel()
    Qs = SH_mean.detach().numpy().squeeze()
    Ql = LH_mean.detach().numpy().squeeze()
    
    return taux, tauy, Qs, Ql

if __name__ == "__main__":
    ux = np.array([4, 10, -5]) 
    uy = np.array([8, 2, 10])
    q = np.array([0.005, 0.007, 0.006]) 
    To = np.array([12, 10, 12])
    Ta = np.array([10, 12, 14])
    p = np.array([1.01, 1.01, 1.0])*10**5
    taux, tauy, Qs, Ql = ann_flux(ux, uy, To, Ta, p, q)
    print(taux)
    print(tauy)
    print(Qs)
    print(Ql)
    