import torch
import torch.nn as nn

# model_path = '/scratch/jw8736/mlflux/example_torch_script/'
model_path = '/glade/work/jiarongw/mlflux/example_torch_script/'

####### Class of ANN with hardcoded sizes and some other options
class ManualSigmoid(nn.Module):
    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-x))
    
class ANN_online(nn.Module):
    def __init__(self, n_in, n_out, hidden_channels, ACTIVATION, 
                 Xmean, Xscale, Ymean, Yscale):

        super().__init__()
    
        # Build the layers
        layers = []
        layers.append(nn.Linear(n_in, hidden_channels[0]))
        layers.append(ManualSigmoid()) 
        for i in range(len(hidden_channels)-1):
            layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            layers.append(ManualSigmoid())   
        layers.append(nn.Linear(hidden_channels[-1], n_out))
        self.layers = nn.Sequential(*layers)
        self.activation = ACTIVATION
        
        # For input and output normalization
        self.register_buffer('Xmean', Xmean)
        self.register_buffer('Xscale', Xscale)
        self.register_buffer('Ymean', Ymean)
        self.register_buffer('Yscale', Yscale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = x.to(torch.float32)
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

    
# class ANN_test(nn.Module):
#     def __init__(self, n_in, n_out, hidden_channels, ACTIVATION, 
#                  Xmean, Xscale, Ymean, Yscale):

#         super().__init__()
    
#         # Build the layers
#         layers = []
#         layers.append(nn.Linear(n_in, hidden_channels[0]))
#         layers.append(nn.Sigmoid()) 
#         # layers.append(nn.Tanh())
#         for i in range(len(hidden_channels)-1):
#             layers.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
#             layers.append(ManualSigmoid())   
#             # layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_channels[-1], n_out))
#         self.layers = nn.Sequential(*layers)
#         self.activation = ACTIVATION
        
#         # For input and output normalization
#         self.register_buffer('Xmean', Xmean)
#         self.register_buffer('Xscale', Xscale)
#         self.register_buffer('Ymean', Ymean)
#         self.register_buffer('Yscale', Yscale)

#     # def forward(self, batch: torch.Tensor) -> torch.Tensor:
#     #     batch_ = (batch - self.Xmean) / self.Xscale 
#     #     return self._fwd_seq(batch_)
#     def forward(self, batch: torch.Tensor) -> torch.Tensor:
#         batch_ = (batch - self.Xmean) / self.Xscale 
#         return self.layers(batch_)
    
    # def forward(self, x):
    #     # Input normalization
    #     x = x.to(torch.float32)
    #     x_ = (x - self.Xmean) / self.Xscale 
    #     x_ = torch.clamp(x_, -1e3, 1e3)
    #     # Forward pass with final activation
    #     if self.activation == 'no':
    #         y_ = self.layers(x_)
    #     elif self.activation == 'exponential':
    #         y_ = torch.exp(self.layers(x_))
    #     else:
    #         raise ValueError('Unknown activation')
    #     # Output denormalization
    #     # y = y_ * self.Yscale + self.Ymean
    #     y = torch.clamp(y_ * self.Yscale + self.Ymean, -1e3, 1e3)
    #     return y
    
if __name__ == "__main__":
    # Hard coded config of the model (mean)
    dir_name = model_path + 'SH/'
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
    mean_ann = mean_ann.float()
    # Set the model to evaluation mode
    mean_ann.eval()
    
    print("Xmean dtype/shape:", mean_ann.Xmean, mean_ann.Xmean.shape)
    print("Xscale dtype/shape:", mean_ann.Xscale, mean_ann.Xscale.shape)
    print("Ymean dtype/shape:", mean_ann.Ymean, mean_ann.Ymean.shape)
    print("Yscale dtype/shape:", mean_ann.Yscale, mean_ann.Yscale.shape)

    # Example inputs
    x = torch.tensor([8.637314, 22.73069, 21.83207, 77.19225, 101078.4]).reshape(1, -1)
    with torch.no_grad():
        y = mean_ann.forward(x)
    print(y)

    # Script and save
    traced_model = torch.jit.trace(mean_ann, x)
    traced_model.save("SH_mean_model_trace.pt")

    # scripted = torch.jit.script(mean_ann)
    # scripted.save("LH_mean_model_script.pt")
