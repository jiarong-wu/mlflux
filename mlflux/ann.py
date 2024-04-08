''' Define the ANN class.
    Not sure if training, weights assignment, and other data-related stuff should be here. '''

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
 

''' This class of ANN is inherited from Pavel's code 
    https://github.com/m2lines/pyqg_generative/blob/master/pyqg_generative/tools/cnn_tools.py
    with the change of activation function from ReLu to Sigmoid.
'''  
class ANN(nn.Module):
    def __init__(self, n_in, n_out, hidden_channels=[24, 24], degree=None):
        super().__init__()
        
        self.degree = degree # But not necessary for this application
    
        layers = []
        layers.append(nn.Linear(n_in, hidden_channels[0]))
        layers.append(nn.Sigmoid())
        
        for i in range(len(hidden_channels)-1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(nn.Sigmoid())
            
        layers.append(nn.Linear(hidden_channels[-1], n_out))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.degree is not None:
            norm = torch.norm(x, dim=-1, p=2, keepdim=True)  # Norm computed across features
            return norm**self.degree * self.layers(x / norm)   # Normalizing by vector norm, for every sample
        else:
            return self.layers(x)
    
    def loss_mse(self, x, ytrue):
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}
    
    
''' The function that specifies sample weights. There can be different candidates.
    Need to be a ufunc that applies to numpy array. 
    Weight should be of order 1 for most samples.
'''

def sample_weights(x):
    return np.where(x > 18, 100.0, 1.0)

# def sample_weights(x, breakpoints, values):
#     return np.piecewise(x, [x < breakpoints[0], 
#                             (x >= breakpoints[0]) & (x < breakpoints[1]), 
#                             (x >= breakpoints[1]) & (x < breakpoints[2]), 
#                              x >= breakpoints[2]], 
#                            [values[0], values[1], values[2], values[0]])

''' Construct a dataset from real measurements '''
class RealFluxDataset(Dataset):
    def __init__(self, ds, input_keys=['U','tsea','tair','rh'], output_keys=['taucx','hsc','hlc'], 
                 weightfunc=sample_weights):
        
        # Assemble input and output features
        self.X = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in input_keys]).astype('float32'))
        self.Y = torch.tensor(np.hstack([ds[key].values.reshape(-1,1) for key in output_keys]).astype('float32'))
            
        # Weights according to weightfunc of choice
        self.W = torch.tensor(weightfunc(self.X).astype('float32'))
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]

from mlflux.synthetic import generate_synthetic_data

''' TODO: Clean these up (change hard-coded sample weight function etc.)'''
class SynFluxDataset2D(Dataset):
    def __init__(self, N=10000, choice='momentum'):
        
        # Generate data
        self.x1, self.x2, self.y1, self.y2 = generate_synthetic_data(N=N)
        self.X = torch.tensor(np.hstack([np.reshape(self.x1,(-1,1)),np.reshape(self.x2,(-1,1))]).astype('float32'))
        
        # A uniform grid flattened to make prediction maps
        # After making the prediction, Y_pred.reshape([100,100])
        x1_vis = np.linspace(0,20,100); x2_vis = np.linspace(-2,2,100)
        x1_mesh, x2_mesh = np.meshgrid(x1_vis,x2_vis,indexing='ij')
        X_uniform = np.hstack([np.reshape(x1_mesh,(-1,1)), np.reshape(x2_mesh,(-1,1))])
        self.X_uniform = torch.tensor(X_uniform.astype('float32'))
        
        # Choice y
        if choice == 'momentum':
            self.Y = torch.tensor(np.reshape(self.y1['sample'],(-1,1)).astype('float32'))
            self.min_MSE = np.average((self.y1['sample']-self.y1['mean'])**2)**0.5
        if choice == 'heat':
            self.Y = torch.tensor(np.reshape(self.y2['sample'],(-1,1)).astype('float32'))
            self.min_MSE = np.average((self.y2['sample']-self.y2['mean'])**2)**0.5
            
        # Weights
        self.W = torch.tensor(sample_weights(self.X).astype('float32'))
        
    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]
    
class SynFluxDataset1D(Dataset):
    def __init__(self, N=10000, choice='momentum'):
        
        # Generate data
        self.x1, self.x2, self.y1, self.y2 = generate_synthetic_data(N=N)
        self.X = np.reshape(self.x1,(-1,1)).astype('float32')
        
        # A uniform grid flattened to make prediction maps
        # After making the prediction, Y_pred.reshape([100,100])
        x1_vis = np.linspace(0,20,100)
        X_uniform = np.reshape(x1_vis,(-1,1))
        self.X_uniform = torch.tensor(X_uniform.astype('float32'))
        
        # Choice y
        if choice == 'momentum':
            self.Y = torch.tensor(np.reshape(self.y1['sample'],(-1,1)).astype('float32'))
            self.min_MSE = np.average((self.y1['sample']-self.y1['mean'])**2)
        if choice == 'heat':
            self.Y = torch.tensor(np.reshape(self.y2['sample'],(-1,1)).astype('float32'))
            self.min_MSE = np.average((self.y2['sample']-self.y2['mean'])**2)
            
        # Weights?
        self.W = torch.tensor(sample_weights(self.X).astype('float32'))
        
    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]
    
    
''' Train a predictor based on log likelihood. 
    predictor has a mean_func and a var_func, which can be both NNs or fixed parameterizations.
    For now the mean and var learning rates are the same. 
    TODO: what do we do with training/testing split? 
    mean_func: function to predict mean (can be fixed or learnable, switch with FIXMEAN) 
    var_func: function to predict variance 
    training_data and validating data: should be instances of torch's Dataset class, already normalized
                                       has dimension N_sample * N_feature
    evaluate_func: a function that computes metrics during training
    batchsize, num_epochs, lr: some training hyperparameters
    
    We don't worry about moving the net to device like https://github.com/m2lines/pyqg_generative/blob/master/pyqg_generative/tools/cnn_tools.py
    because the nets are expected to be small.
'''


def train (mean_func, var_func, training_data, validating_data, evaluate_func,
           batchsize=100, num_epochs=100, lr=5e-3, gamma=0.2, FIXMEAN=True):
    
    # Put the training data into dataloader
    dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    
    # Whether we have fixed deterministic model or not
    if FIXMEAN:
        optimizer = optim.Adam(var_func.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(list(var_func.parameters()) \
            +list(mean_func.parameters()), lr=lr)
    
    # Can adjust the scheduler later
    milestones = [int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)] 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    log = {'LLLoss': [], 'lr': [], 'training_mse': [], 'validating_mse': [], 'training_r2': [], 'validating_r2': []}
    loss = nn.GaussianNLLLoss(reduction='none')

    # Training
    for epoch in range(num_epochs):
        LLLoss = 0.
        for i, (inputs, targets, w) in enumerate(dataloader):
            optimizer.zero_grad()
            mean = mean_func(inputs)
            # TODO: maybe change where this activation is 
            var = (var_func(inputs))**2 # squared to predict positive variance 
            likelihood = torch.sum(loss(targets, mean, var)*w)   
            likelihood.backward() 
            optimizer.step()
            LLLoss += likelihood.item() * len(inputs)  # Returns the value of this tensor as a standard Python number          
        scheduler.step()     

        LLLoss = LLLoss / len(training_data)
        log['LLLoss'].append(LLLoss)
        log['lr'].append(scheduler.get_last_lr())
        
        # TODO: write a function that evaluate some metrics and then log
        # It could be some metrics function of predictor
        # It could be performed on both training and validating dataset
        # These can be added to the log dictionary
        mse, r2 = evaluate_func(training_data)
        log['training_mse'].append(mse.detach()); log['training_r2'].append(r2.detach())
        mse, r2 = evaluate_func(validating_data)
        log['validating_mse'].append(mse.detach()); log['validating_r2'].append(r2.detach())
        
        # TODO: write a function for visualizing of the model behavior across epochs, if so desire
        # var = (predictor.var_func(training_data.X_uniform))**2
        # mean = predictor.mean_func(training_data.X_uniform)
        # log['var'].append(var.detach())
        # log['mean'].append(mean.detach())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {LLLoss:.8f}")
        
    return log

