import torch
model = torch.jit.load("LH_mean_model.pt", map_location="cpu")
model.eval()

# create a placeholder input shaped like what Fortran sends
x = torch.tensor([1,2,3,4,5]).reshape(1,-1)  # or load a saved sample
with torch.no_grad():
    out = model(x)
    print("input:", x)
    print("output:", out)