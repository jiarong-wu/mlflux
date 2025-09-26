''' Compute ANN or bulk predicted fluxes and exchange coefficients
'''
import xarray as xr
import torch
from mlflux.predictor import FluxANNs
from mlflux.utils import qsea, qsat
from mlflux.ann import RealFluxDataset

### Load bulk
from mlflux.COARE.COARE3p6 import coare36vn_zrf_et 

### Load ANN?
# For now ANN are loaded outside 

### Compute fluxes (with minimal amount of required input variables)
@torch.no_grad
def compute(X, label='ANN'):
    global model_LH, model_SH, model_M
    zt = 10 # only used for small corrections 
    zq = 10 # only used for small corrections 
    
    U = X[0]  # m/s
    tsea = X[1] # degree C
    tair = X[2] # degree C
    rh = X[3] # relative humidity (%)
    pair = X[4] # pascal
    Rgas = 287.1
    kappa = 0.4
    g = 9.8
    qs = qsea(tsea, pair/100.) / 1000. # kg/kg
    e = qsat(tair, pair/100.) * (rh/100.)
    qair = 0.622*e / (pair/100. - 0.378*e)
    Le = (2.501 - 0.00237*tsea) * 1e6
    cpa = 1004.67
    thetav = (tair + 273.15) * (1 + 0.61*qair) # virtual potential (?) temperature
    rhoair = (pair - 12.5*zq) / (Rgas * thetav) # density with height correction

    if label == 'ANN':
        LH = model_LH.pred_mean(X) 
        SH = model_SH.pred_mean(X) 
        M = model_M.pred_mean(X)
        M_cross = model_M_cross.pred_mean(X)
        CE = LH / U / (qair - qs) / Le / rhoair
        CT = SH / U / (tair + 0.0098*zt - tsea) / cpa / rhoair
        CD = (M**2 + M_cross**2) ** 0.5 / U**2 / rhoair
        return M.numpy()[0][0],-SH.numpy()[0][0],-LH.numpy()[0][0],CD.numpy()[0][0],CT.numpy()[0][0],CE.numpy()[0][0]
        
    elif label == 'COARE3p6':
        zu = 10
        zt = 10
        zq = 10
        A = coare36vn_zrf_et(u=np.array([U]), zu=np.array([zu]), t=np.array([tair]), 
                                            zt=np.array([zt]), lon=np.array([0]), lat=np.array([45]),
                                            rh=np.array([rh]), zq=np.array([zq]), P=np.array([pair])/100., 
                                            ts=np.array([tsea]), sw_dn=0, lw_dn=400, jd=10, zi=600, rain=0, Ss=35, 
                         cp=None, sigH=None, zrf_u=10.0, zrf_t=10.0, zrf_q=10.0) 
        return A[0][0],A[0][1],A[0][2],A[0][3],A[0][4],A[0][5]
        
    else:
        print('Algorithm not recognized!')
        
           
### An xarray wrapper to compute fluxes and append fields as LH, SH, M
def add_field(ds, model_M, model_Mcross, model_SH, model_LH):
    vd = RealFluxDataset(ds, input_keys=model_M.config['ikeys'], 
                         output_keys=model_M.config['okeys'], bulk_keys=model_M.config['bkeys'])
    LH = model_LH.pred_mean(vd.X)
    LH_var = model_LH.pred_var(vd.X)   
    SH = model_SH.pred_mean(vd.X)
    SH_var = model_SH.pred_var(vd.X)
    M = model_M.pred_mean(vd.X)
    M_var = model_M.pred_var(vd.X)
    M_cross = model_Mcross.pred_mean(vd.X)
    M_cross_var = model_Mcross.pred_var(vd.X)

    ds["LH"] = xr.DataArray(LH.detach().numpy().squeeze(), dims=("time"))
    ds["LH_var"] = xr.DataArray(LH_var.detach().numpy().squeeze(), dims=("time"))
    ds["SH"] = xr.DataArray(SH.detach().numpy().squeeze(), dims=("time"))
    ds["SH_var"] = xr.DataArray(SH_var.detach().numpy().squeeze(), dims=("time"))
    ds["M"] = xr.DataArray(M.detach().numpy().squeeze(), dims=("time"))
    ds["M_var"] = xr.DataArray(M_var.detach().numpy().squeeze(), dims=("time"))
    ds["M_cross"] = xr.DataArray(M_cross.detach().numpy().squeeze(), dims=("time"))
    ds["M_cross_var"] = xr.DataArray(M_cross_var.detach().numpy().squeeze(), dims=("time"))
    return ds

### Compute transfer coefficients and append fields as CD, CT, CE, MOL, etc.
### With fluxes already pre-computed
def compute_coeff(ds):
    # physical properties
    # Le = 2.5 * 10 ** 6
    # Check all the required fields exist
    Rgas = 287.1
    kappa = 0.4
    g = 9.8
    ds['qsea'] = xr.DataArray(qsea(ds['tsea'].values, ds['p'].values/100.)/1000., dims=("time"))
    # ds['cpa'] = 1004.67*(1-ds['qair']) + 1850*ds['qair'] 
    ds['Le'] = (2.501 - 0.00237*ds['tsea']) * 1e6
    ds['cpa'] = ds['qair'] / ds['qair'] * 1004.67
    ds['thetav'] = (ds['tair'] + 273.15) * (1 + 0.61*ds['qair']) # virtual potential (?) temperature
    # rhoa = P*100. / (Rgas * (t + tdk) * (1 + 0.61*Q))
    # P - (0.125 * zt)
    # ds['rhoair'] = ds['p'] / (Rgas * ds['thetav']) # density 
    ds['rhoair'] = (ds['p'] - 12.5*ds['zq']) / (Rgas * ds['thetav']) # density with height correction
    # Sign is correct because we define hsc as -w'T'
    denom = kappa * g * (ds['hsc']/ds['cpa']/ds['thetav'] + 0.61*ds['hlc']/ds['Le'] + 1e-8) # Monin-Obukhov length
    ds['MOL'] = ds['taucx'] * (ds['taucx']/ds['rhoair']) ** 0.5 / denom # Monin-Obukhov length

    # dt = ts - t - 0.0098*zt
    # coefficients
    ds['ann_CE'] = ds['LH'] / ds['U'] / (ds['qair'] - ds['qsea']) / ds['Le'] / ds['rhoair']
    ds['bulk_CE'] = ds['hlb'] / ds['U'] / (ds['qair'] - ds['qsea']) / ds['Le'] / ds['rhoair']
    ds['CE'] = ds['hlc'] / ds['U'] / (ds['qair'] - ds['qsea']) / ds['Le'] / ds['rhoair']
    ds['ann_CT'] = ds['SH'] / ds['U'] / (ds['tair'] + 0.0098*ds['zt'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    ds['bulk_CT'] = ds['hsb'] / ds['U'] / (ds['tair']+ 0.0098*ds['zt'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    ds['CT'] = ds['hsc'] / ds['U'] / (ds['tair']+ 0.0098*ds['zt'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    # ds['ann_CT'] = ds['SH'] / ds['U'] / (ds['tair'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    # ds['bulk_CT'] = ds['hsb'] / ds['U'] / (ds['tair'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    # ds['CT'] = ds['hsc'] / ds['U'] / (ds['tair'] - ds['tsea']) / ds['cpa'] / ds['rhoair']
    # ds['ann_CD'] = abs(ds['M']) / ds['U']**2 / ds['rhoair']
    ds['ann_CD'] = (ds['M']**2 + ds['M_cross']**2)**0.5 / ds['U']**2 / ds['rhoair']
    ds['bulk_CD'] = ds['taubx'] / ds['U']**2 / ds['rhoair']
    ds['CD'] = ds['taucx'] / ds['U']**2 / ds['rhoair']
    return ds

def compute_ds(ds):
    global model_LH, model_SH, model_M
    required_fields = ['U', 'tsea', 'tair', 'rh', 'p']
    optional_fields = ['zq', 'zt', 'zu']
    if not all(field in ds for field in required_fields):
        missing = [field for field in required_fields if field not in ds]
        raise ValueError(f"Missing required fields in dataset: {missing}")
    if ds['p'].mean() not in range(60000, 150000):
        print(f"Warning: Unusual pressure values detected, mean p = {ds['p'].mean().item()}, check units!")
    if ds['rh'].mean() not in range(0, 120):
        print(f"Warning: Unusual relative humidity values detected, mean rh = {ds['rh'].mean().item()}, check units!")
        
    vd = RealFluxDataset(ds, input_keys=model_M.config['ikeys'], 
                         output_keys=model_M.config['okeys'], bulk_keys=model_M.config['bkeys'])
    LH = model_LH.pred_mean(vd.X)
    LH_var = model_LH.pred_var(vd.X)   
    SH = model_SH.pred_mean(vd.X)
    SH_var = model_SH.pred_var(vd.X)
    M = model_M.pred_mean(vd.X)
    M_var = model_M.pred_var(vd.X)
    M_cross = model_Mcross.pred_mean(vd.X)
    M_cross_var = model_Mcross.pred_var(vd.X)

    ds["LH"] = xr.DataArray(LH.detach().numpy().squeeze(), dims=("time"))
    ds["LH_var"] = xr.DataArray(LH_var.detach().numpy().squeeze(), dims=("time"))
    ds["SH"] = xr.DataArray(SH.detach().numpy().squeeze(), dims=("time"))
    ds["SH_var"] = xr.DataArray(SH_var.detach().numpy().squeeze(), dims=("time"))
    ds["M"] = xr.DataArray(M.detach().numpy().squeeze(), dims=("time"))
    ds["M_var"] = xr.DataArray(M_var.detach().numpy().squeeze(), dims=("time"))
    ds["M_cross"] = xr.DataArray(M_cross.detach().numpy().squeeze(), dims=("time"))
    ds["M_cross_var"] = xr.DataArray(M_cross_var.detach().numpy().squeeze(), dims=("time"))