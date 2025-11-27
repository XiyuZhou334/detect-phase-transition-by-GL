import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
device=torch.device('cuda:1')
epsilon = 1e-15

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, exclusive=False):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (0 if exclusive else 1):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
class ResMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, exclusive=False):
        super(ResMaskedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias)
        self.register_buffer('resmask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.resmask.fill_(1)
        self.resmask[:, :, kW // 2 + (0 if exclusive else 1):] = 0
        self.resmask[:, :, :, :kH // 2 ] = 0
    def forward(self, x):
        self.weight.data *= self.resmask
        return super(ResMaskedConv2d, self).forward(x)
class PixelCNN(nn.Module):
    def __init__(self,L,kernel):
        super(PixelCNN, self).__init__()
        self.L=L
        self.kernel=kernel
        self.width1=64
        self.width2=32
        self.net1 = nn.Sequential(
            MaskedConv2d(1, self.width1, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2, self.kernel//2), bias=False, exclusive=True),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width1),
                MaskedConv2d(self.width1, self.width1, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=False)
            ),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width1),
                MaskedConv2d(self.width1, self.width1, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=False)
            ),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width1),
                MaskedConv2d(self.width1, 1, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=False)
            ),
        )
        self.Sigmoid = nn.Sigmoid()
        self.net2 = nn.Sequential(
            ResMaskedConv2d(1, self.width2, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2, self.kernel//2), bias=False, exclusive=False),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width2),
                ResMaskedConv2d(self.width2, self.width2, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=False)
            ),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width2),
                ResMaskedConv2d(self.width2, self.width2, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=False)
            ),
            nn.Sequential(
                nn.PReLU(num_parameters=self.width2),
                ResMaskedConv2d(self.width2, 1, kernel_size=(self.kernel,self.kernel), stride=(1, 1), padding=(self.kernel//2,self.kernel//2), bias=False, exclusive=True)
            ),
        )
    def forward(self, x):
        out1=self.net1(x)
        out2=self.net2(x)
        out=out1+out2
        final_out = self.Sigmoid(out)
        return final_out
    def sample(self,batch_size):
        tensor = torch.zeros([batch_size, 1, self.L,self.L],device=device)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(tensor)

                tensor[:, :, i, j] = torch.bernoulli(x_hat[:, :, i, j]) * 2 - 1
        return tensor, x_hat
    def log_p(self,tensors):
        x_hat = self.forward(tensors)
        mask = (tensors + 1) / 2
        log_p = (torch.log(x_hat + epsilon) * mask + torch.log(1 - x_hat + epsilon) * (1 - mask))
        log_p = log_p.view(log_p.shape[0], -1).sum(dim=1)
        return log_p
T=5.0
L = 8          
kernel=5 
J=2.0
boundary = 'tri'
epochs = 3000
lr = 0.0001
batch_size=1000
seed=69
hyperparams = {"L": L, "kernel": kernel}

plt.ion()
fig, ax = plt.subplots()
x_data, y1_data=[],[]
line1, = ax.plot([], [], 'b-')
ax.set_xlabel('Epochs')
ax.set_ylabel('loss')

#load samples
c=torch.load(f"L={L} {boundary} J={J:.1f} ten.pth")
for key in c:
    c[key] = c[key].to(device=device)
spl=c[f"T={T:.1f}"]#samples of T

torch.manual_seed(seed)
net = PixelCNN(**hyperparams)
net=net.to(device=device)
params = net.parameters()
optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
train_time = 0

#train
for epoch in range(epochs):
    net.train()
    trainstart_time = time.time()
    optimizer.zero_grad()
    with torch.no_grad():
        indices = torch.randperm(spl.size(0))[:batch_size]
        tensors=spl[indices]
    assert not tensors.requires_grad
    log_q=net.log_p(tensors)
    loss_reinforce= (-log_q).mean()
    loss_reinforce.backward()
    optimizer.step()  
    train_time +=  time.time()-trainstart_time
    x_data.append(epoch)
    y1_data.append(loss_reinforce.item())
    line1.set_xdata(x_data)
    line1.set_ydata(y1_data)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
# plt.savefig(f'L=16loss_beta={beta:.2f}.svg', format='svg')
plt.ioff()
plt.show()
print(f'train_time:{train_time:.2f}s')