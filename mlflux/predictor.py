import pickle 
import numpy as np
import torch 
import torch.nn as nn
from mlflux.ann import ANN, ANNdiff, train
from time import time
import copy

from scipy.stats import wasserstein_distance, norm

# def channelwise_function(X: np.array, fun) -> np.array:
#     '''
#     Help function for normalizing. For array X of size 
#     Nbatch x Nfeatures 
#     applies function "fun" for each channel and returns array of size
#     1 x Nfeatures 
#     '''

#     N_features = X.shape[1]
#     if len(X.shape) == 4:
#         out = np.zeros((1,N_features,1,1))
#     elif len(X.shape) == 2:
#         out = np.zeros((1,N_features))
#     else:
#         raise ValueError('Wrong dimensions of input array')
    
#     for n_f in range(N_features):
#         out[0,n_f] = fun(X[:,n_f])

#     return out.astype('float32')


class predictor:
    """
    Attributes inherited from all classes
    __init__ fills the class based on a dictionary
    save pickles itself, this is for convience and one does not need to convert
    self.fname to a filename etc.
    """

    def __init__(self, params={}):
        for key in params.keys():
            setattr(self, key, params[key])

    def save(self, fname="model_1"):
        ## Use fname if the fname is defined.
        pickle.dump(self, open(getattr(self, "fname", fname) + ".p", "wb")) 


class FluxANNs(predictor):
    ''' Define a predictor that has both mean and variance with ANN.
        Required parameters:
            mean_ann_para and var_ann_para: dictionaries containing the following parameters for ANN 
                {'n_in': input dim, 'n_out': output dim, 'hidden_channels': hidden layer nodes, layer numbers are inferred, e.g., [16,16].}
    '''
    def __init__(self, params={}):
        super().__init__(params)
        # Check that it has all the parameters
        if not hasattr(self, "mean_ann_para"):
            raise ValueError('Need to define ANN parameters for mean!')
        if not hasattr(self, "var_ann_para"):
            raise ValueError('Need to define ANN parameters for var!')
        self.mean_func = ANN(**self.mean_ann_para)
        self.var_func = ANN(**self.var_ann_para)

    def summary(self):
        for item in sorted(self.__dict__):
            print(str(item) + ' = ' + str(self.__dict__[item]))
            
    def pred_mean(self, X):
        # X is a torch tensor of dim Nsamples * Nfeatures
        X_ = (X - self.Xscale['mean']) / self.Xscale['scale']
        Ypred_mean = self.mean_func(X_) * self.Yscale['scale'] + self.Yscale['mean']
        return Ypred_mean 
      
    def pred_var(self, X):
        # X is a torch tensor of dim Nsamples * Nfeatures
        # NOTICE: We call it var_func but it's actually std, thus the square
        X_ = (X - self.Xscale['mean']) / self.Xscale['scale']
        # Ypred_var = (self.var_func(X_) * self.Yscale['scale'])**2
        # Actually we have the nonlinear activation function to prevent negative values! So it should be like below:
        Ypred_var = self.var_func(X_) * self.Yscale['scale']**2 
        return Ypred_var  
    
    def metrics(self, X, Ytruth):
        # These operations are performed on torch tensor
        # Assuming X is of dimension Nsample * Nfeatures
        # return torch tensors as well
        # Compute all three metrics together because why not
        
        Ypred_mean = self.pred_mean(X)
        mse = torch.mean((Ypred_mean - Ytruth)**2, dim=0)
        r2 = 1 - mse / torch.var(Ytruth, dim=0) # over sample axis
        Ypred_var = self.pred_var(X)
        residual_norm = (Ytruth - Ypred_mean) / Ypred_var**0.5
        wd = []
        for yi in range(residual_norm.shape[-1]):
            r = norm.rvs(size=1000000)  # Pick a big numble of samples from normal distribution  
            l1 = wasserstein_distance(residual_norm[:,yi].detach(),r)
            wd.append(l1)  
        self.scores = {'mse':mse.detach(), 'r2':r2.detach(), 'wd':np.array(wd)}  
        
        return self.scores
    
    def mse_r2_likelihood_scaled(self, dataset):
        # NOTICE: Here X and Y are assumed already SCALED 
        # Used for evaluation during training
        # There is also NO weights applied!!
        X_ = dataset.X
        Ypred_mean = self.mean_func(X_) 
        Ypred_var = self.var_func(X_)
        
        mse = torch.mean((Ypred_mean - dataset.Y)**2, dim=0)
        r2 = 1 - mse / torch.var(dataset.Y, dim=0) # over sample axis
        loss = nn.GaussianNLLLoss(reduction='none')
        LLLoss = torch.sum(loss(dataset.Y, Ypred_mean, Ypred_var)) 
        # TODO: should we add weights?
        
        return (mse, r2, LLLoss)
   
    # ''' These two needs to be defined after knowing how many variables we are using.
    #     It is a dictionary containing mean and variance, each should be of dimension 1 * Nfeatures
    # '''
    # @property
    # def Xscale(self):
    #     # It depends on variable feature length and need to be implemented later
    #     raise NotImplementedError
    
    # @property
    # def Yscale(self):
    #     # It depends on output vector length and need to be implemented later
    #     raise NotImplementedError            
    
    def fit(self, training_data, validating_data, training_paras, VERBOSE=True):
        ''' training_paras shoud be a dictionary containing:
            {'batchsize':100, 'num_epochs':100, 'lr':5e-3}
        '''
        
        self.training_paras = training_paras
        training_data_cp = copy.deepcopy(training_data) # so that we don't modify training_data itself
        training_data_cp.X = (training_data.X - self.Xscale['mean']) / self.Xscale['scale']
        training_data_cp.Y = (training_data.Y - self.Yscale['mean']) / self.Yscale['scale']

        validating_data_cp = copy.deepcopy(validating_data) # so that we don't modify training_data itself
        validating_data_cp.X = (validating_data.X - self.Xscale['mean']) / self.Xscale['scale']
        validating_data_cp.Y = (validating_data.Y - self.Yscale['mean']) / self.Yscale['scale']
        
        t_start = time()
        log = train (self.mean_func, self.var_func, training_data_cp, validating_data_cp, 
                     self.mse_r2_likelihood_scaled, **training_paras, FIXMEAN=False, VERBOSE=VERBOSE)
        print(f'training took {time() - t_start:.2f} seconds, loss at last epoch %.4f' %log['LLLoss'][-1])
        self.log = log 
        return log
    
    def evaluate_uniform (self):
        # A uniform grid flattened to make prediction maps
        # Need to be implemented depending on how many input features
        raise NotImplementedError

class Fluxdiff(FluxANNs):
    ''' Similar as FluxANNs but with a fixed channel for variable difference. '''
    def __init__(self,params={}):
        super().__init__(params)
        self.mean_func = ANNdiff(**self.mean_ann_para)
        self.var_func = ANNdiff(**self.var_ann_para) 
  
