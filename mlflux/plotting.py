''' Helper functions for plotting '''

from matplotlib import pyplot as plt
import numpy as np
from utils import mse_r2

def comparison(ds, ax, xplot='U', yplot='tau'):
    if xplot == 'Tdiff':
        x = ds.tair - ds.tsea
    else:
        x = ds[xplot]
    
    if yplot == 'tau':
        ax.plot(x, ds.taucx, '.', markersize=1, alpha=0.5, label='Measured')
        ax.plot(x, ds.taubx, '.', markersize=1, alpha=0.5, 
                label='COARE, mse=%.3f, r2=%.3f' %mse_r2(ds.taubx.values,ds.taucx.values))
        ax.set_ylim([-0.1,0.6]); ax.set_ylabel('Momentum flux ($N/m^2$)')
              
    if yplot == 'hs':
       ax.plot(x, ds.hsc, '.', markersize=1, alpha=0.5, label='Measured')
       ax.plot(x, ds.hsb, '.', markersize=1, alpha=0.5, 
               label='COARE, mse=%.3f, r2=%.3f' %mse_r2(ds.hsb.values,ds.hsc.values))
       ax.set_ylim([-40,20]); ax.set_ylabel('Sensible heat flux ($W/m^2$)')
       
    if yplot == 'hl':
       ax.plot(x, ds.hlc, '.', markersize=1, alpha=0.5, label='Measured')
       ax.plot(x, ds.hlb, '.', markersize=1, alpha=0.5, 
               label='COARE, mse=%.3f, r2=%.3f' %mse_r2(ds.hlb.values,ds.hlc.values))
       ax.set_ylim([-300,50]); ax.set_ylabel('Latent heat flux ($W/m^2$)')
    
    if xplot == 'U': ax.set_xlabel('Wind speed ($m/s$)'); ax.set_xlim([0,20])
    elif xplot == 'Tdiff': ax.set_xlabel('Temp. diff. ($\degree C$)');  ax.set_xlim([-5,2])
    elif xplot == 'rh': ax.set_xlabel('Relative humidity (%)');  ax.set_xlim([40,100])

    ax.legend(fancybox=False, loc='upper left')
    return ax

    
# def comparison_witherror(X, Y, predictor):
#     ''' Scatter plot that include the uncertainty '''
    
#     fig, axes = plt.subplots(1,2,figsize=[10,4])
    
#     ax = axes[0]
#     idx = np.argsort(X[:,0][:100])
#     ax.plot(X[:,0][idx], Y[:,1][idx],'.') # Plotting training data
#     # ax.plot(X[:,0][:100], y2_mean[:100], '.') # Plotting conditional mean
#     mean = predictor.predict(X)
#     dist = predictor.pred_dist(X)
#     std = dist.std()
#     # ax.plot(X[:,0][idx], mean[idx], '.', c='k')
#     ax.errorbar(X[:,0][idx], mean[idx], yerr=std[idx], fmt=".", c='k')

#     ax.set_xlabel('Input $x_1$')
#     ax.set_ylabel('Output $y_2$')

#     ax = axes[1]
#     idx = np.argsort(X[:,1][:100])
#     ax.plot(X[:,1][idx], Y[:,1][idx],'.')
#     ax.plot(X[:,1][idx], mean[idx], '.', c='k')
#     ax.errorbar(X[:,1][idx], mean[idx], yerr=std[idx], fmt=".", c='k')
#     # plt.legend()

#     ax.set_xlabel('Input $x_2$')
#     ax.set_ylabel('Output $y_2$')
#     pass