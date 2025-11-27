import torch
import numpy as np
import matplotlib.pyplot as plt
# torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float64)
device=torch.device('cuda:1')

J=1.0
# samp
data = torch.load(f"L={16} J={1.0:.1f}.pth", weights_only=False)
# x1= torch.load(f"L={16} J={1.0:.1f}.pth")["beta"][1:-1]
# y1 = torch.load(f"L={16} J={1.0:.1f}.pth")["I"]
# x2 = torch.load(f"L={16} J={1.0:.1f}.pth")["beta2"][1:-1]
# y2= torch.load(f"L={16} J={1.0:.1f}.pth")["I2"]
x1 = data["beta"][1:-1]
y1 = data["I"]
x2 = data["beta2"][1:-1]
y2 = data["I2"]

# plt.xlabel(r"inverse temperature,\beta")   

fig, ax1 = plt.subplots(figsize=(6, 4))
color="black"
color1 ='#3498db'#'#2E86AB'
color2 = '#2ecc71'#'#A23B72'#, '#F18F01', '#C73E1D']
ax1.set_xlabel(r"inverse temperature ($\beta$)")
ax1.set_ylabel('$I$', color=color)
line1 = ax1.plot(x1, y1, color=color1, marker='s',ms=7,label='data1',linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
line2 = ax2.plot(x2, y2, color=color2, marker='o',ms=9, label='data2',linewidth=1.5)
ax2.tick_params(axis='y', labelcolor=color2)
lines = line1+line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels,fontsize=12,loc='upper left', bbox_to_anchor=(0.02, 0.98),frameon=False)


plt.axvline(x=1/(2*J/np.log(np.sqrt(2)+1)), color='black', linestyle='--',alpha=0.2,linewidth=1.5)
x_pos=1/(2*J/np.log(np.sqrt(2)+1))
y_min, y_max = plt.ylim()
y_pos = y_max * 0.8
plt.text(x_pos, y_pos, f'$\\beta_c= {x_pos:.2f}$', 
         horizontalalignment='center', 
         verticalalignment='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.0),
         fontsize=10, color='black')

plt.tight_layout()
# plt.savefig('I_16x16.pdf', dpi=300, format='pdf')
# plt.savefig(f'I_16x16.svg', format='svg')
plt.show()