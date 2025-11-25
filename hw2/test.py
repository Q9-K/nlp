import torch

data = torch.randn(2, 3, 4)
print(data.view(-1))