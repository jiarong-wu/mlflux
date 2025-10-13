import torch
import torch.nn as nn

# def rhcalc(t,p,q):
#     ''' TAKEN FROM COARE PACKAGE. usage: rh = rhcalc(t,p,q)
#         Returns RH(%) for given t(C), p(mb) and specific humidity, q(kg/kg)
#         Returns ndarray float for any numeric object input.
#     '''
    
#     q2 = copy(asarray(q, dtype=float))    # conversion to ndarray float
#     p2 = copy(asarray(p, dtype=float))
#     t2 = copy(asarray(t, dtype=float))
#     es = qsat(t2,p2)
#     em = p2 * q2 / (0.622 + 0.378 * q2)
#     rh = 100.0 * em / es
#     return rh

model_path = '/scratch/jw8736/mlflux/example_torch_script/'

####### Class of ANN with hardcoded sizes and some other options
class ANN_online(nn.Module):
    def __init__(self, n_in, n_out, hidden_channels, ACTIVATION, 
                 Xmean, Xscale, Ymean, Yscale):

        super().__init__()
    
        # Build the layers
        layers = []
        layers.append(nn.Linear(n_in, hidden_channels[0]))
        layers.append(nn.Sigmoid()) 
        for i in range(len(hidden_channels)-1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(nn.Sigmoid())   
        layers.append(nn.Linear(hidden_channels[-1], n_out))
        self.layers = nn.Sequential(*layers)
        self.activation = ACTIVATION
        
        # For input and output normalization
        self.register_buffer('Xmean', Xmean)
        self.register_buffer('Xscale', Xscale)
        self.register_buffer('Ymean', Ymean)
        self.register_buffer('Yscale', Yscale)
    
    def forward(self, x):
        # Input normalization
        x_ = (x - self.Xmean) / self.Xscale 
        # Forward pass with final activation
        if self.activation == 'no':
            y_ = self.layers(x_)
        elif self.activation == 'exponential':
            y_ = torch.exp(self.layers(x_))
        else:
            raise ValueError('Unknown activation')
        # Output denormalization
        y = y_ * self.Yscale + self.Ymean
        return y
    
if __name__ == "__main__":
    # Hard coded config of the model (mean)
    dir_name = model_path + 'LH/'
    n_in = 5
    n_out = 1
    hidden_channels = [32, 16] 
    activation = 'no' # 'exponential' for var
    # Load the trained weights and scales
    state_dict = torch.load(dir_name + "mean_weights.pt", map_location="cpu")
    # Instantiate and load weights
    mean_ann = ANN_online(n_in=n_in, n_out=n_out, hidden_channels=hidden_channels, ACTIVATION=activation, 
                          Xmean=state_dict['Xmean'], Xscale=state_dict['Xscale'], Ymean=state_dict['Ymean'], Yscale=state_dict['Yscale'])
    mean_ann.load_state_dict(state_dict)
    # Set the model to evaluation mode
    mean_ann.eval()
    # Example inputs
    # x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # x = x.reshape(1, -1)
    # with torch.no_grad():
    #     y = mean_ann.forward(x)
    # print(y)

    # Example dummy inputs (same shape as your test arrays)
    # ux = torch.tensor([4.0, 10.0, -5.0])
    # uy = torch.tensor([8.0, 2.0, 10.0])
    # To = torch.tensor([12.0, 10.0, 12.0])
    # Ta = torch.tensor([10.0, 12.0, 14.0])
    # p = torch.tensor([1.01, 1.01, 1.0]) * 1e5
    # q = torch.tensor([0.005, 0.007, 0.006])
    # out = model(ux, uy, To, Ta, p, q)
    # print(out)

    # Script and save
    # traced_model = torch.jit.trace(mean_ann, x)
    # traced_model.save("LH_mean_model.pt")

    scripted = torch.jit.script(mean_ann)
    scripted.save("LH_mean_model_script.pt")