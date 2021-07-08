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
        return torch.normal(mean,std)
    
state_dim = 2
action_dim = 4
option_dim = 2

pi_lo = NN_PI_LO(state_dim, action_dim).to(device)
NN_low = [[None]*1 for _ in range(option_dim)] 
for option in range(option_dim):
        NN_low[option] = copy.deepcopy(pi_lo)

state = np.array([[0, 1], [1, 1], [2, 2]])
state = torch.FloatTensor(state.reshape(len(state), state_dim)).to(device)
albi = NN_low[0](state).cpu().clamp(-1,1) #.data.numpy()

