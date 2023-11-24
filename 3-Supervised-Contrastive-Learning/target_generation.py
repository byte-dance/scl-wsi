import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()


N = 5   # number of class
M = 128 # feature dimension
print("N =", N)
print("M =", M)
criterion = uniform_loss()
criterion = criterion.cuda()
x = torch.randn(N,M)
x = x.cuda()
x.requires_grad = True
optimizer = optim.Adam([x], lr=1e-3)
min_loss = 100
optimal_target = None

N_iter = 10000
for i in range(N_iter):
    x_norm = F.normalize(x, dim=1)
    loss = criterion(x_norm)
    if i % 100 == 0:
        print(i, loss.item())
    if loss.item() < min_loss:
        min_loss = loss.item()
        optimal_target = x_norm

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('min loss : {}'.format(min_loss))

np.save('optimal_{}_{}.npy'.format(N, M), optimal_target.detach().cpu().numpy())

