import pickle 
import torch 
import torch.nn as nn
import json
import numpy as np

class ANN(nn.Module):
    def __init__(self, n_in, n_out, hidden_channels=[24, 24], degree=None, ACTIVATION='no', dropout_rate=-1.):

        super().__init__()
        
        self.degree = degree # But not necessary for this application
        self.dropout_rate = dropout_rate
    
        layers = []
        layers.append(nn.Linear(n_in, hidden_channels[0]))
        layers.append(nn.Sigmoid())
        if dropout_rate > 0.:
            layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_channels)-1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(nn.Sigmoid())
            if dropout_rate > 0.:
                layers.append(nn.Dropout(dropout_rate))
            
        layers.append(nn.Linear(hidden_channels[-1], n_out))
        
        self.layers = nn.Sequential(*layers)
        self.activation = ACTIVATION
    
    def forward(self, x):
        if self.degree is not None:
            norm = torch.norm(x, dim=-1, p=2, keepdim=True)  # Norm computed across features
            return norm**self.degree * self.layers(x / norm)   # Normalizing by vector norm, for every sample
        else:
            if self.activation == 'no':
                return self.layers(x)
            elif self.activation == 'square':
                return self.layers(x)**2
            elif self.activation == 'exponential':
                return torch.exp(self.layers(x))
            elif self.activation == 'softplus':
                m = nn.Softplus()
                return m(self.layers(x))
    
    def loss_mse(self, x, ytrue):
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}
        
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
    
    def score_scaled(self, dataset):
        # NOTICE: Here X and Y are assumed already SCALED 
        # Used for evaluation during training
        # Weights are applied!!
        X_ = dataset.X
        Ypred_mean = self.mean_func(X_) 
        Ypred_var = self.var_func(X_)

        # mse and r2
        mse = torch.mean((Ypred_mean - dataset.Y)**2*dataset.W, dim=0)
        r2 = 1 - mse / torch.mean(dataset.Y**2*dataset.W, dim=0) # over sample axis
        # log likelihood loss
        loss = nn.GaussianNLLLoss(reduction='none')
        LLLoss = torch.mean(loss(dataset.Y, Ypred_mean, Ypred_var)*dataset.W, dim=0)
        # residual gaussianity
        error = Ypred_mean - dataset.Y
        error_norm = error / Ypred_var**0.5
        res_mean = torch.mean(error_norm*dataset.W, dim=0)
        res_var = torch.mean(error_norm**2*dataset.W, dim=0)
        
        return np.array([mse.detach(), r2.detach(), LLLoss.detach(), res_mean.detach(), res_var.detach()]).squeeze()
        
    def score_scaled_mean_only(self, dataset):
        # For if one wants to train a mean net first
        # NOTICE: Here X and Y are assumed already SCALED 
        # Used for evaluation during training
        # Weights are applied!!
        X_ = dataset.X
        Ypred_mean = self.mean_func(X_) 

        # mse and r2
        mse = torch.mean((Ypred_mean - dataset.Y)**2*dataset.W, dim=0)
        r2 = 1 - mse / torch.mean(dataset.Y**2*dataset.W, dim=0) # over sample axis
        
        return np.array([mse.detach(), r2.detach()]).squeeze()
   
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
    
    def fit(self, training_data, validating_data, training_paras, VERBOSE=True, TWOSTEPS=False):
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
        
        if TWOSTEPS:
            log1 = train_mse (self.mean_func, training_data_cp, validating_data_cp, 
                              self.score_scaled_mean_only, **training_paras, VERBOSE=VERBOSE)
            print(f'Mean training took {time() - t_start:.2f} seconds.')
            log2 = train (self.mean_func, self.var_func, training_data_cp, validating_data_cp, 
                         self.score_scaled, **training_paras, FIXMEAN=True, VERBOSE=VERBOSE)  
            print(f'Variance training took {time() - t_start:.2f} seconds.')
            self.log = [log1, log2]
        else:
            log = train (self.mean_func, self.var_func, training_data_cp, validating_data_cp, 
                         self.score_scaled, **training_paras, FIXMEAN=False, VERBOSE=VERBOSE)
            print(f'Training took {time() - t_start:.2f} seconds. Loss at last epoch %.4f' %log['LLLoss'][-1])
            self.log = log       
            
        return self.log
    
    def evaluate_uniform (self):
        # A uniform grid flattened to make prediction maps
        # Need to be implemented depending on how many input features
        raise NotImplementedError


def open_case (model_dir, model_name):
    with open(model_dir + 'config.json', 'r') as f:
        config = json.load(f)   
    filename = model_dir + model_name
    with open(filename, "rb") as input_file:
        model = pickle.load(input_file)   
    model.config = config 
    return model