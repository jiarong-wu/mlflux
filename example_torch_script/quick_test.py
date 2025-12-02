import torch
model = torch.jit.load("SH_mean_model_script.pt", map_location="cpu")
model.eval()

# create a placeholder input shaped like what Fortran sends
x = torch.ones(10, 5)  # or load a saved sample
with torch.no_grad():
    out = model(x)
    print("input:", x.shape)
    print("output:", out.shape)
