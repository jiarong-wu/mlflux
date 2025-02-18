''' Assuming ypred, ytruth, and weight to have dimension N_sample * N_outputs. '''
from scipy.stats import wasserstein_distance
import numpy as np
import xarray as xr
from mlflux.ann import RealFluxDataset
from matplotlib import pyplot as plt
import json
import pickle 
from mlflux.datafunc import data_split_psd

####### Plot prediction (thinned points) ########
def plot_pred (ax, model, ds, subsample=None):
    ''' First evaluate on the whole data set. '''
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    # [ann_mse, ann_r2, res_mean, res_var, wd, bulk_mse, bulk_r2]
    scores = evaluate (model, ds, WEIGHT=model.config['WEIGHT'])
    
    ''' Then subsample for visualization. '''    
    if subsample != None:        
        if model.config['WEIGHT']:
            w_index = np.random.choice(len(vd.X), subsample, p=vd.W.squeeze()/vd.W.sum())
        else:
            w_index = np.random.choice(len(vd.X), subsample)
        vd.X = vd.X[w_index,:]; vd.Y = vd.Y[w_index,:]; vd.Bulk = vd.Bulk[w_index,:]
        
    Ypred_mean = model.pred_mean(vd.X)
    if model.config['okeys'] == ['hlc']:
        X = vd.X[:,3]
        ax.set_ylim([-250,50]); ax.set_xlim([50,110])
        ax.set_xlabel(r'$RH(\%)$')
        ax.set_ylabel('Latent heat flux $Q_L \; [W/m^2]$')
    elif model.config['okeys'] == ['hsc']:
        X = vd.X[:,1] - vd.X[:,2]
        ax.set_ylim([-100,50]); ax.set_xlim([-4,6])
        ax.set_xlabel('$T_o-T_a \; [\degree C]$'); ax.set_ylabel('Sensible heat flux $Q_S \; [W/m^2]$') 
    elif model.config['okeys'] == ['taucx']:
        X = vd.X[:,0]
        ax.set_ylim([0,1]); ax.set_xlim([0,20])
        ax.set_xlabel('$U_{10} \; [m/s]$'); ax.set_ylabel(r'Momentum flux $\tau_x \; [N/m^2]$')  
    elif model.config['okeys'] == ['taucy']:
        X = vd.X[:,0]
        ax.set_ylim([0,0.1]); ax.set_xlim([0,20])
        ax.set_xlabel('$U_{10} \; [m/s]$'); ax.set_ylabel(r'Cross-wind momentum flux $\tau_y \; [N/m^2]$')  
        
    ax.plot(X, vd.Y, 'o', mfc="None", markeredgewidth=0.5, markersize=4, label='Measurements')
    ax.plot(X, vd.Bulk, 'o', mfc="None", markeredgewidth=0.5, markersize=4, label='Bulk (R2=%.2f)' %scores[-1])
    ax.plot(X, Ypred_mean.detach().numpy(), 'o', mfc="None", markeredgewidth=0.5, markersize=4, label='ANN (R2=%.2f)' %scores[1])
    ax.legend(fancybox=False)

''' Instead of visualizing wrt one input, visualizing wrt truth. '''
def plot_corr (ax, model, ds, subsample=None):
    ''' First evaluate on the whole data set. '''
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    # [ann_mse, ann_r2, res_mean, res_var, wd, bulk_mse, bulk_r2]
    scores = evaluate (model, ds, WEIGHT=model.config['WEIGHT'])
    
    ''' Then subsample for visualization. '''    
    if subsample != None:        
        if model.config['WEIGHT']:
            w_index = np.random.choice(len(vd.X), subsample, p=vd.W.squeeze()/vd.W.sum())
        else:
            w_index = np.random.choice(len(vd.X), subsample)
        vd.X = vd.X[w_index,:]; vd.Y = vd.Y[w_index,:]; vd.Bulk = vd.Bulk[w_index,:]
        
    Ypred_mean = model.pred_mean(vd.X)

    cbulk = 'k'
    cann = 'C1'
    if model.config['okeys'] == ['hlc']:
        ax.set_ylim([-250,50]); ax.set_xlim([-250,50])
        ax.set_yticks([-250,-150,-50,0,50])
        ax.set_xticks([-250,-150,-50,0,50])
        ax.set_xlabel(r'Measurement $Q_{L,c} \; [W/m^2]$')
        # ax.set_ylabel(r'Prediction $[W/m^2]$')
        ax.plot(vd.Y, vd.Bulk, 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$Q_{L,b}$ ($R^2$=%.2f)' %scores[-1], c=cbulk)
        ax.plot(vd.Y, Ypred_mean.detach().numpy(), 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\mu_{Q_L}$ ($R^2$=%.2f)' %scores[1], c=cann)
    elif model.config['okeys'] == ['hsc']:
        X = vd.X[:,1] - vd.X[:,2]
        ax.set_ylim([-100,50]); ax.set_xlim([-100,50])
        ax.set_yticks([-100,-50,0,50]); ax.set_xticks([-100,-50,0,50])
        ax.set_xlabel('Measurement  $Q_{S,c} \; [W/m^2]$')
        # ax.set_ylabel('Prediction $[W/m^2]$') 
        ax.plot(vd.Y, vd.Bulk, 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$Q_{S,b}$ ($R^2$=%.2f)' %scores[-1], c=cbulk)
        ax.plot(vd.Y, Ypred_mean.detach().numpy(), 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\mu_{Q_S}$ ($R^2$=%.2f)' %scores[1], c=cann)
    elif model.config['okeys'] == ['taucx']:
        ax.set_ylim([0,0.6]); ax.set_xlim([0,0.6])
        ax.set_xticks([0,0.2,0.4,0.6]); ax.set_yticks([0,0.2,0.4,0.6])
        ax.set_xlabel(r'Measurement $\tau_{x,c} \; [N/m^2]$')
        # ax.set_ylabel(r'Prediction $[N/m^2]$')
        ax.plot(vd.Y, vd.Bulk, 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\tau_{x,b}$ ($R^2$=%.2f)' %scores[-1], c=cbulk)
        ax.plot(vd.Y, Ypred_mean.detach().numpy(), 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\mu_{\tau_x}$ ($R^2$=%.2f)' %scores[1], c=cann)
    elif model.config['okeys'] == ['taucy']:
        ax.set_ylim([-0.1,0.1]); ax.set_xlim([-0.1,0.1])
        ax.set_xlabel(r'Measurement $\tau_{y,c} \; [N/m^2]$')
        # ax.set_ylabel(r'Prediction $[N/m^2]$')    
        ax.plot(vd.Y, vd.Bulk, 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\tau_{y,b}$ (R2=%.2f)' %scores[-1], c=cbulk)
        ax.plot(vd.Y, Ypred_mean.detach().numpy(), 'o', mfc="None", markeredgewidth=0.5, markersize=4, 
                label=r'$\mu_{\tau_y}$ (R2=%.2f)' %scores[1], c=cann)
           
    ax.plot(np.arange(-1000,1000), np.arange(-1000,1000), lw=0.5, ls='--', c='gray')
    ax.legend(fancybox=False, bbox_to_anchor=(0, 1.22), loc='upper left', handletextpad=0.)

####### Plot residual ########
def plot_res (ax, model, ds):
    ''' First evaluate on the whole data set '''
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    # [ann_mse, ann_r2, res_mean, res_var, wd, bulk_mse, bulk_r2]
    scores = evaluate (model, ds, WEIGHT=model.config['WEIGHT'])
    
    Ypred_mean = model.pred_mean(vd.X)
    Ypred_var = model.pred_var(vd.X)
    error = Ypred_mean.detach().numpy() - vd.Y.detach().numpy()
    error_norm = error/Ypred_var.detach().numpy()**0.5
    # label='$(\mu_{ANN}(\mathbf{X}) - Obs)/\sigma_{ANN}(\mathbf{X})$'
    if model.config['WEIGHT']:
        ax.hist(error_norm, bins=np.linspace(-4, 4, 80), density=True, weights=vd.W[:,0], label='W res.')
    else:
        ax.hist(error_norm, bins=np.linspace(-4, 4, 80), density=True, 
                label='$\hat{\epsilon}$')

    mu = 0      # mean
    sigma = 1   # standard deviation    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    
    ax.plot(x, y, color='k', label='Gaussian')
    ax.legend(loc='upper left')
    ax.text(2, 0.5, '$\mu_{\hat{\epsilon}} = %.2f$ \n $\sigma_{\hat{\epsilon}} = %.2f$ \n $wd = %.2f$' %(scores[2],scores[3], scores[4]), ha='center', va='center')
    
    ax.set_xlim([-4,4]); ax.set_ylim([0.0001,1])

####### Load model ########
def open_case (model_dir, model_name):
    with open(model_dir + 'config.json', 'r') as f:
        config = json.load(f)
    ds =  xr.load_dataset(config['datapath'] + config['datafile'])    
    filename = model_dir + model_name
    with open(filename, "rb") as input_file:
        model = pickle.load(input_file)   
    model.config = config 
    return model
    
####### Scores related to the mean prediction ##########
def mse_r2_weighted(ypred, ytruth, weight):
    mse = np.average((ypred-ytruth)**2, weights=weight, axis=0)
    ytruth_mean = np.average(ytruth, weights=weight, axis=0)
    ytruth_var = np.average((ytruth-ytruth_mean)**2, weights=weight, axis=0)
    r2 = 1 - np.average((ypred-ytruth)**2, weights=weight, axis=0)/ytruth_var
    return mse.flatten(), r2.flatten()

####### Scores related to the normalized distribution shape ##########
def distribution_weighted(ypred, ytruth, yvar, weight):
    res_norm = (ypred - ytruth) / yvar**0.5
    res_mean = np.average(res_norm, weights=weight, axis=0)
    res_var = np.average(res_norm**2, weights=weight, axis=0)
    wd = [] # In case that y has more than one output
    np.random.seed(0) # for reproducibility
    r = np.random.normal(0, 1, size=1000000) # Reference Gaussian. Pick a big numble of samples from normal distribution
    for yi in range(res_norm.shape[-1]): 
        d = wasserstein_distance(res_norm[:,yi], r, u_weights=weight)
        wd.append(d)  
    wd = np.array(wd).reshape(1,-1)
    return res_mean.flatten(), res_var.flatten(), wd.flatten()

###### Evalute score of model on ds, either with or without weights. #######
def evaluate (model, ds, WEIGHT=True):
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    Ypred_mean = model.pred_mean(vd.X)
    Ypred_var = model.pred_var(vd.X)
    if WEIGHT:
        ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W)
        bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W)
        res_mean, res_var, wd = distribution_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), 
                                                      Ypred_var.detach().numpy(), vd.W)
    else:
        ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)
        bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)            
        res_mean, res_var, wd = distribution_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), 
                                                      Ypred_var.detach().numpy(), vd.W/vd.W)
    return np.vstack([ann_mse, ann_r2, res_mean, res_var, wd, bulk_mse, bulk_r2])

def eval_bias (model, ds, WEIGHT=True):
    vd = RealFluxDataset(ds, input_keys=model.config['ikeys'], 
                         output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
    Ypred_mean = model.pred_mean(vd.X)
    Ypred_var = model.pred_var(vd.X)
    if WEIGHT:
        ann_bias = np.average(Ypred_mean.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W, axis=0).flatten()
        bulk_bias = np.average(vd.Bulk.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W, axis=0).flatten()
    else:
        ann_bias = np.average(Ypred_mean.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W/vd.W, axis=0).flatten()
        bulk_bias = np.average(vd.Bulk.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W/vd.W, axis=0).flatten()
    return np.vstack([ann_bias, bulk_bias])
        
####### Different regimes #######
# Here ds is full psd
def evaluate_over_splits (model, ds, WEIGHT=True):
    split1 = [[69, 83, 78, 87, 72, 71, 68, 67, 73], [77], [77]] # metz
    split2 = [[77, 69, 83, 87, 68], [67, 72, 73, 78, 71], [67, 72, 73, 78, 71]] # calwater, hiwings, capricorn, neaqs, gasex
    split3 = [[77, 67, 72, 73, 78, 71], [68, 83, 69, 87], [68, 83, 69, 87]] # dynamo, stratus, epic, whots
    split_ensem = [split1, split2, split3]
    nn_mse_splits = []; bulk_mse_splits = []
    nn_r2_splits = []; bulk_r2_splits = []

    for i in range(len(split_ensem)):
        training_ds, validating_ds, testing_ds = data_split_psd(ds, split=split_ensem[i], 
                                                                PLOT=False, XVIS='samples', VERBOSE=False)
        vd = RealFluxDataset(validating_ds, input_keys=model.config['ikeys'], 
                             output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
        Ypred_mean = model.pred_mean(vd.X)
        if WEIGHT:
            ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W)
            bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W)
        else:
            ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)
            bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)            
        nn_mse_splits.append(ann_mse); bulk_mse_splits.append(bulk_mse)
        nn_r2_splits.append(ann_r2); bulk_r2_splits.append(bulk_r2) 
        
    return (np.array(nn_r2_splits).squeeze(), np.array(bulk_r2_splits).squeeze(), 
            np.array(nn_mse_splits).squeeze(), np.array(bulk_mse_splits).squeeze())

def evaluate_over_splits_alter (model, ds, WEIGHT=True):
    split1 = [[69, 83, 78, 87, 72, 71, 68, 67, 73], [77], [77]] # metz
    split2 = [[77, 69, 87, 83, 68, 73, 71], [67, 72, 78], [67, 72, 78]] # calwater, hiwings, neaqs
    split3 = [[77, 69, 83, 87, 68, 67, 72, 78], [73, 71], [73, 71]] # capricorn, gasex
    split4 = [[77, 67, 72, 73, 78, 71], [68, 83, 69, 87], [68, 83, 69, 87]] # dynamo, stratus, epic, whots
    split_ensem = [split1, split2, split3, split4]
    nn_mse_splits = []; bulk_mse_splits = []
    nn_r2_splits = []; bulk_r2_splits = []

    for i in range(len(split_ensem)):
        training_ds, validating_ds, testing_ds = data_split_psd(ds, split=split_ensem[i], 
                                                                PLOT=False, XVIS='samples', VERBOSE=False)
        vd = RealFluxDataset(validating_ds, input_keys=model.config['ikeys'], 
                             output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
        Ypred_mean = model.pred_mean(vd.X)
        if WEIGHT:
            ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W)
            bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W)
        else:
            ann_mse, ann_r2 = mse_r2_weighted(Ypred_mean.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)
            bulk_mse, bulk_r2 = mse_r2_weighted(vd.Bulk.detach().numpy(), vd.Y.detach().numpy(), vd.W/vd.W)            
        nn_mse_splits.append(ann_mse); bulk_mse_splits.append(bulk_mse)
        nn_r2_splits.append(ann_r2); bulk_r2_splits.append(bulk_r2) 
        
    return (np.array(nn_r2_splits).squeeze(), np.array(bulk_r2_splits).squeeze(), 
            np.array(nn_mse_splits).squeeze(), np.array(bulk_mse_splits).squeeze())

def eval_bias_over_splits_alter (model, ds, WEIGHT=True):
    split1 = [[69, 83, 78, 87, 72, 71, 68, 67, 73], [77], [77]] # metz
    split2 = [[77, 69, 87, 83, 68, 73, 71], [67, 72, 78], [67, 72, 78]] # calwater, hiwings, neaqs
    split3 = [[77, 69, 83, 87, 68, 67, 72, 78], [73, 71], [73, 71]] # capricorn, gasex
    split4 = [[77, 67, 72, 73, 78, 71], [68, 83, 69, 87], [68, 83, 69, 87]] # dynamo, stratus, epic, whots
    split_ensem = [split1, split2, split3, split4]
    nn_bias_splits = []; bulk_bias_splits = []

    for i in range(len(split_ensem)):
        training_ds, validating_ds, testing_ds = data_split_psd(ds, split=split_ensem[i], 
                                                                PLOT=False, XVIS='samples', VERBOSE=False)
        vd = RealFluxDataset(validating_ds, input_keys=model.config['ikeys'], 
                             output_keys=model.config['okeys'], bulk_keys=model.config['bkeys'])
        Ypred_mean = model.pred_mean(vd.X)
        if WEIGHT:
            ann_bias = np.average(Ypred_mean.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W, axis=0).flatten()
            bulk_bias = np.average(vd.Bulk.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W, axis=0).flatten()
        else:
            ann_bias = np.average(Ypred_mean.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W/vd.W, axis=0).flatten()
            bulk_bias = np.average(vd.Bulk.detach().numpy()-vd.Y.detach().numpy(), weights=vd.W/vd.W, axis=0).flatten()         
        nn_bias_splits.append(ann_bias); bulk_bias_splits.append(bulk_bias)
        
    return (np.array(nn_bias_splits).squeeze(), np.array(bulk_bias_splits).squeeze())

####### Plot heat map #######
import matplotlib as mpl
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts