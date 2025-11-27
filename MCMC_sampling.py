import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
device=torch.device('cuda:1')
epsilon = 1e-15

def initialize_spins(n_spins, size, device=device):
    a=torch.randint(0, 2, (int(n_spins/2), size, size), device=device) * 2 - 1
    a=torch.cat([a, -a], dim=0)
    return a

def total_energy(spins, J):
    L = spins.shape[1]
    energy = 0
    for i in range(L):
        for j in range(L):
            energy -= J * spins[:,i,j] * (spins[:,i,(j+1)%L] + spins[:,(i+1)%L,j]+spins[:,(i+1)%L,(j+1)%L])
    return energy

def metropolis_step(spins, beta, J):
    """Metropolis-Hastings"""
    n_spins, L, _ = spins.shape
    
    i = torch.randint(0, L, (n_spins,), device=device)
    j = torch.randint(0, L, (n_spins,), device=device)
    batch_indices = torch.arange(n_spins, device=device)
  
    delta_E = 2 * J * spins[batch_indices, i, j] * (
        spins[batch_indices, (i-1) % L, j] + 
        spins[batch_indices, (i+1) % L, j] + 
        spins[batch_indices, i, (j-1) % L] + 
        spins[batch_indices, i, (j+1) % L] +
        spins[batch_indices, (i-1) % L, (j-1) % L]+
        spins[batch_indices, (i+1) % L, (j+1) % L]
    )

    accept_prob = torch.exp(-beta * delta_E)
    accept = torch.rand(n_spins, device=spins.device) < accept_prob
    # updata spins
    spins[batch_indices[accept], i[accept], j[accept]] *= -1
    return spins

def MCMC(spins,beta,burn_in,beta_anneal,J):
    n_spins, _, _ = spins.shape   
    for step in range(burn_in):
        # burn-in
        b_eta = beta * (1 - beta_anneal**step)
        spins= metropolis_step(spins, b_eta,J)
    return spins
def MCsample(spins,n_steps,avg_step,beta,J):
    n_spins, L, _ = spins.shape
    results = torch.zeros((n_spins, L,L), device=device)
    tensor = torch.zeros((0, L,L), device=device)
    for step in range(n_steps):    
        spins = metropolis_step(spins, beta,J)
        if (step) % avg_step == 0:
            results.copy_(spins)
            tensor=torch.cat([tensor, results], dim=0)
    return tensor.unsqueeze(1)