import numpy as np

datapath = '/home/jw8736/mlflux/data/Processed/'
datafile = 'psd_coare3p0_weight1_wave0.nc'

ann_size = [64,32,16]
mean_activation = 'no'
var_activation = 'exponential'

ikeys = ['U','tsea','tair','rh']
okeys = ['hsc']
bkeys = ['hsb']

NRAND = 5 # number of random training
RATIO = 0.2 # ratio of validation/total, default 20%

WEIGHT = True # use importance weighting or not
RESIDUAL = False # optional training on residual, TODO: implement

training_paras = {'batchsize':1000, 'num_epochs':500, 'lr':5e-4, 'gamma':0.2,
                 'EARLYSTOPPING':False, 'patience':20, 'factor':0.5, 'max_epochs_without_improvement':100}

compute_norm = True
norms = {'imean':np.array([0,0,0,0]), 'iscale':np.array([20,20,20,100]), 
        'omean':np.array([0]), 'oscale':np.array([0.1])} # only used if compute_norm is False, need to match ikeys and okeys

VERBOSE = False
two_steps = True
