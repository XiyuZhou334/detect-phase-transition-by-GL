import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
device=torch.device('cuda:1')
'''hyperparameters
J
T
L
a   #Order parameter interval
num_T #numbers of Order parameters
boundary
'''
y=[]
for i in range (1,num_T-1):

    lnp1=torch.load(f"L={L} {boundary} J={J:.1f} sample_logp.pth")[f"T={T+i*a-a:.1f}"][i]
    lnp2=torch.load(f"L={L} {boundary} J={J:.1f} sample_logp.pth")[f"T={T+i*a+a:.1f}"][i]
    I=torch.abs(lnp2-lnp1)
    I=I.cpu().numpy()
    y.append(I.mean().item())
x=np.linspace(T,T+num_T*a-a,num_T)+a
plt.axvline(x=(4*J/np.log(3)),color="green", linestyle='--')
# plt.axvline(x=1/(2*J/np.log(np.sqrt(2)+1)), color='gray', linestyle='--')
plt.scatter(x[:-2],y[:], color="green", marker='s')
plt.plot(x[:-2],y[:],color="blue")
plt.xlabel('T')    
plt.ylabel('I')
# plt.savefig(f'I_16x16.svg', format='svg')
plt.show()

