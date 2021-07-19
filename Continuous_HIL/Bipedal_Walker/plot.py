#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:31:04 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt


with open('TD3_expert/results/TD3_BipedalWalker-v3_0.npy', 'rb') as f:
    TD3_eval = np.load(f, allow_pickle=True)


steps = np.linspace(0,int(1e6),201)


plt.plot(steps, TD3_eval, label = 'TD3 from scratch')
plt.ylabel('Reward')
plt.xlabel('Steps')
plt.show()