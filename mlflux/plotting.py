''' Helper functions for plotting '''

from matplotlib import pyplot as plt
import numpy as np
from mlflux.utils import mse_r2
import torch

''' Inputs to this plotting functions are torch tensors. '''
def plot_feature(ax, X, Y_truth, Y_pred, LEGEND=True):
    mse = torch.mean((Y_truth-Y_pred)**2)
    r2 = 1 - mse/torch.var(Y_truth)
    ax.plot(X,Y_truth, '.', markersize=0.5, label='Measurements')
    ax.plot(X,Y_pred, '.', markersize=0.5, label='Prediction (of mean) \nr2=%.4f, mse=%.4f' %(r2, mse))
    if LEGEND:
        ax.legend(fancybox=False)

''' Inputs to these plotting functions is a dataset instance. 
    It also calls the model to evaluate. '''    
def plotting_4features (model, data):
    fig, axes = plt.subplots(4, 2, figsize=[8,16], dpi=200)
    
    feature = 0; Y_pred = model.pred_mean(data.X)[:,feature]
    ax = axes[0,0]; plot_feature(ax,data.X[:,0],data.Y[:,feature],Y_pred.detach())
    ax.set_ylim([0,1]); ax.set_xlim([0,20])
    ax.set_xlabel('$U$'); ax.set_ylabel('Momentum flux (along)')
    ax = axes[0,1]; plot_feature(ax,data.X[:,1]-data.X[:,2],data.Y[:,feature],Y_pred.detach(), LEGEND=False)
    ax.set_ylim([0,1]); ax.set_xlim([-2,5])
    ax.set_xlabel('$T_o-T_a$')
    
    feature = 1; Y_pred = model.pred_mean(data.X)[:,feature]
    ax = axes[1,0]; plot_feature(ax,data.X[:,0],data.Y[:,feature],Y_pred.detach())
    ax.set_ylim([-0.1,0.1]); ax.set_xlim([0,20])
    ax.set_xlabel('$U$'); ax.set_ylabel('Momentum flux (cross)')
    ax = axes[1,1]; plot_feature(ax,data.X[:,1]-data.X[:,2],data.Y[:,feature],Y_pred.detach(), LEGEND=False)
    ax.set_ylim([-0.1,0.1]); ax.set_xlim([-2,5])
    ax.set_xlabel('$T_o-T_a$')
    
    feature = 2; Y_pred = model.pred_mean(data.X)[:,feature]
    ax = axes[2,0]; plot_feature(ax,data.X[:,0],data.Y[:,feature],Y_pred.detach())
    ax.set_ylim([-60,20]); ax.set_xlim([0,20])
    ax.set_xlabel('$U$'); ax.set_ylabel('Sensible heat flux')
    ax = axes[2,1]; plot_feature(ax,data.X[:,1]-data.X[:,2],data.Y[:,feature],Y_pred.detach(), LEGEND=False)
    ax.set_ylim([-60,20]); ax.set_xlim([-2,5])
    ax.set_xlabel('$T_o-T_a$')
    
    feature = 3; Y_pred = model.pred_mean(data.X)[:,feature]
    ax = axes[3,0]; plot_feature(ax,data.X[:,0],data.Y[:,feature],Y_pred.detach())
    ax.set_ylim([-200,0]); ax.set_xlim([0,20])
    ax.set_xlabel('$U$'); ax.set_ylabel('Latent heat flux')
    ax = axes[3,1]; plot_feature(ax,data.X[:,3],data.Y[:,feature],Y_pred.detach(), LEGEND=False)
    ax.set_ylim([-200,0]); ax.set_xlim([40,100])
    ax.set_xlabel(r'$RH(\%)$'); 
    
    return fig

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

def vis_training (log):
    fig, axes = plt.subplots(4,1,sharex=True,figsize=[4,8],dpi=200)
          
    log['training_mse'] = np.array(log['training_mse'])
    log['training_r2'] = np.array(log['training_r2'])
    log['validating_mse'] = np.array(log['validating_mse'])
    log['validating_r2'] = np.array(log['validating_r2'])

    axes[0].plot(log['LLLoss'])
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim([-100,500])
    
    axes[1].plot(log['lr'])
    axes[1].set_ylabel('Learning rate')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[1].set_ylim([0,0.002])
    
    for i in range(len(log['training_r2'][0])):
        line, = axes[2].plot(log['training_r2'][:,i], label='Training data') 
        axes[2].plot(log['validating_r2'][:,i], label='Validating data')
    axes[2].set_ylabel(r'$R^2$')
    axes[2].legend(fontsize=6)
    # axes[2].set_ylim([0,1])
    
    for i in range(len(log['training_r2'][0])):
        line, = axes[3].plot(log['training_mse'][:,i], label='Training data') 
        axes[3].plot(log['validating_mse'][:,i], label='Validating data')
        # axes[3].plot(log['validating_mse'][:,i], '--', c=line.get_color(), label='Validating data, feature %g' %(i+1))
    axes[3].set_ylabel(r'$MSE$')
    axes[3].set_xlabel('Epoch')
    
    return fig, axes
    
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