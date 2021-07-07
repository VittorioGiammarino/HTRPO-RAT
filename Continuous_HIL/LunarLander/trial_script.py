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
        
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128, action_dim)
        self.lS = nn.Softmax(dim=1)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
                
        return self.lS(self.l3(a))
    
state_dim = 2
action_dim = 4
option_dim = 2

pi_lo = NN_PI_LO(state_dim, action_dim).to(device)
NN_low = [[None]*1 for _ in range(option_dim)] 
for option in range(option_dim):
        NN_low[option] = copy.deepcopy(pi_lo)

state = np.array([[0, 1], [1, 1], [2, 2]])
state = torch.FloatTensor(state.reshape(len(state), state_dim)).to(device)
albi = NN_low[0](state).cpu() #.data.numpy()

