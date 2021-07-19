#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:40:12 2021

@author: vittorio
"""

import copy
import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN_PI_LO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN_PI_LO, self).__init__()
        
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, 256)
        nn.init.uniform_(self.l1.weight, -0.5, 0.5)
        self.l2 = nn.Linear(256, 256)
        nn.init.uniform_(self.l2.weight, -0.5, 0.5)
        self.l3 = nn.Linear(256, 2*action_dim)
        nn.init.uniform_(self.l2.weight, -0.5, 0.5)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        mean = a[:,0:self.action_dim]
        std = torch.exp(a[:, self.action_dim:])
        output = torch.tanh(torch.normal(mean,std))
        return output, mean, std
    
state_dim = 24
action_dim = 4
option_dim = 2

pi_lo = NN_PI_LO(state_dim, action_dim).to(device)
NN_low = [[None]*1 for _ in range(option_dim)] 
for option in range(option_dim):
        NN_low[option] = copy.deepcopy(pi_lo)
        
with open('TD3_expert/DataFromExpert/TrainingSet_continuous.npy', 'rb') as f:
    TrainingSet_tot = np.load(f, allow_pickle=True)

with open('TD3_expert/DataFromExpert/Labels_continuous.npy', 'rb') as f:
    Labels_tot = np.load(f, allow_pickle=True)
    
with open('TD3_expert/DataFromExpert/Reward_continuous.npy', 'rb') as f:
    Reward = np.load(f, allow_pickle=True)

state_samples = TrainingSet_tot
action_samples = Labels_tot

state_samples = torch.FloatTensor(state_samples.reshape(len(state_samples), state_dim)).to(device)
action_samples = torch.FloatTensor(action_samples.reshape(len(action_samples), action_dim)).to(device)
output, mean, std = NN_low[0](state_samples) #.cpu().clamp(-1,1) .data.numpy()
output = output.cpu().data.flatten().numpy()
# %%

def get_prob(mean, std, actions):
    action_dim = mean.shape[1]
    a_prob = torch.ones(len(actions),1)
    denominator = torch.ones(len(actions),1)
    for a in range(action_dim):
        a_prob *= (torch.exp(-(actions[:,a]-mean[:,a])**2/(2*std[:,a]**2))/(torch.sqrt(2*torch.FloatTensor([np.pi])*std[:,a]**2))).reshape(len(actions),1)
        denominator *= (1-(torch.tanh(actions[:,a]))**2).reshape(len(actions),1)
        
    return a_prob/torch.abs(denominator)

a_prob = get_prob(mean, std, action_samples).cpu()

# %%

def get_log_likelihood_pi_lo(mean, std, actions):
    action_dim = mean.shape[1]
    Log_likelihood_a_prob = torch.zeros(len(actions),1)
    Log_denominator = torch.zeros(len(actions),1)
    for a in range(action_dim):
         Log_likelihood_a_prob += (-(actions[:,a]-mean[:,a])**2/(2*(std[:,a]).clamp(1e-10,1e10)**2)  - torch.log((std[:,a]).clamp(1e-10,1e10)) - torch.log(torch.sqrt(2*torch.FloatTensor([np.pi])))).reshape(len(actions),1) 
         Log_denominator += (torch.log(1-(torch.tanh(actions[:,a]))**2)).reshape(len(actions),1)
    return Log_likelihood_a_prob - Log_denominator

Log_likelihood_a_prob = get_log_likelihood_pi_lo(mean, std, action_samples)


