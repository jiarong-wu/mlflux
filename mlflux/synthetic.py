import numpy as np

def generate_synthetic_data (N=10000,):
    ''' Generate synthetic data according to prescribed destribution for state variables. 
        So far the scale and the distribution are fixed but we can relax that late.
        Arguments:
            N: sample size
        Returns:
            x1: wind speed drawn from Reyleigh distribution
            x2: temperature difference drawn from normal distribution
            y1: dictionary containing mean, std, and samples of momentum flux
            y2: dictionary containing mean, std, and samples of sensible heat flux
    '''
            
    # Wind speed
    scale = 5.
    x1 = np.random.rayleigh(scale, N)
    # Temperature diff
    mean, var = -0.8, 0.8
    x2 = np.random.normal(mean, var, N)
    # Bulk coefficients
    Cd = 0.0015 # For omentum, between 1e-3 to 2e-3
    Ch = 1 # For sensible heat, this is after multiplied by heat capacity

    # Fluxes
    y1_mean = Cd*x1**2; y1_std = Cd*x1**2*0.2 + 0.01
    y1_sample = y1_mean + np.array([np.random.normal(0,std,1).squeeze() for std in y1_std]) # This is kind of inefficient to generate random samples
    y2_mean = Ch*x1*x2; y2_std = abs(Ch*x1*x2)*0.15 + 1
    y2_sample = y2_mean + np.array([np.random.normal(0,var,1).squeeze() for var in y2_std])
    y1 = {'mean':y1_mean,'std':y1_std,'sample':y1_sample}
    y2 = {'mean':y2_mean,'std':y2_std,'sample':y2_sample}
    return (x1,x2,y1,y2)