#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:31:04 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%

with open('FlatRL_expert/results/mean_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    mean = np.load(f, allow_pickle=True)
    
with open('FlatRL_expert/results/std_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    std = np.load(f, allow_pickle=True)
    
with open('FlatRL_expert/results/steps_TD3_BipedalWalker-v3_Nseed_20.npy', 'rb') as f:
    steps = np.load(f, allow_pickle=True)
    
with open('results/HIL+HTD3_BipedalWalker-v3_0.npy', 'rb') as f:
    HTD3_HIL_0 = np.load(f, allow_pickle=True)
 
with open('FlatRL_expert/results/evaluation_SAC_BipedalWalker-v3_Nseed_1.npy', 'rb') as f:
    SAC = np.load(f, allow_pickle=True)
    
with open('results/HSAC_HIL_True_BipedalWalker-v3_1.npy', 'rb') as f:
    HSAC_HIL = np.load(f, allow_pickle=True)

with open('results/HTD3_HIL_True_BipedalWalker-v3_1.npy', 'rb') as f:
    HTD3_HIL_1 = np.load(f, allow_pickle=True)


with open('results/HSAC_HIL_True_HTD0_True_BipedalWalker-v3_2.npy', 'rb') as f:
    HSAC_HIL = np.load(f, allow_pickle=True)    

    
# %%

if not os.path.exists("./Figures"):
    os.makedirs("./Figures")

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 5)
ax.plot(steps, mean, label='Vanilla TD3', c=clrs[0])
ax.fill_between(steps, mean-std, mean+std, alpha=0.1, facecolor=clrs[0])
ax.plot(steps, HTD3_HIL_0, label='HIL + HTD3 seed 0', c=clrs[1])
ax.plot(steps, HTD3_HIL_1, label='HIL + HTD3 seed 1', c=clrs[4])
ax.set_ylim([-200,400])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Bipedal Walker')
plt.savefig('Figures/TD3_vs_HTD3.pdf', format='pdf')

fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 5)
ax.plot(steps, SAC, label='SAC', c=clrs[2])
ax.plot(steps, HSAC_HIL, label='HIL + HSAC', c=clrs[3])
ax.legend(loc=3, facecolor = '#d8dcd6')
ax.set_ylim([-200,400])
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
ax.set_xlabel('Steps')
ax.set_ylabel('Reward')
ax.set_title('Bipedal Walker')
plt.savefig('Figures/SAC_vs_HSAC.pdf', format='pdf')

