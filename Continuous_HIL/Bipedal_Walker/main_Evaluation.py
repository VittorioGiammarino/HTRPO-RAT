#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:59 2020

@author: vittorio
"""
import World
import numpy as np
import OnlineBW_HIL
import BatchBW_HIL 
import matplotlib.pyplot as plt

with open('Models/likelihood_batch.npy', 'rb') as f:
    likelihood_batch = np.load(f, allow_pickle=True)

with open('Models/likelihood_online.npy', 'rb') as f:
    likelihood_online = np.load(f, allow_pickle=True)
        
pi_hi_batch = BatchBW_HIL.NN_PI_HI.load('Models/Saved_Model_Batch/pi_hi_NN')
pi_lo_batch = []
pi_b_batch = []
option_space = 2
for i in range(option_space):
    pi_lo_batch.append(BatchBW_HIL.NN_PI_LO.load('Models/Saved_Model_Batch/pi_lo_NN_{}'.format(i)))
    pi_b_batch.append(BatchBW_HIL.NN_PI_B.load('Models/Saved_Model_Batch/pi_b_NN_{}'.format(i)))

pi_hi_online = OnlineBW_HIL.NN_PI_HI.load('Models/Saved_Model_Online/pi_hi_NN')     
pi_lo_online = []
pi_b_online = []
for i in range(option_space):
    pi_lo_online.append(OnlineBW_HIL.NN_PI_LO.load('Models/Saved_Model_Online/pi_lo_NN_{}'.format(i)))
    pi_b_online.append(OnlineBW_HIL.NN_PI_B.load('Models/Saved_Model_Online/pi_b_NN_{}'.format(i)))
        

with open('Models/Saved_Model_Expert/Q_trained.npy', 'rb') as f:
    Q_trained = np.load(f, allow_pickle=True)

# %% Expert
n_bins = (3, 3, 6, 6)
Q_table = np.zeros(n_bins + (2,))
expert = World.CartPole.Expert(n_bins, 1, 1, Q_table)
lower_bounds = [expert.env.observation_space.low[0], -0.3, expert.env.observation_space.low[2], -np.radians(50) ]
upper_bounds = [expert.env.observation_space.high[0], 0.3, expert.env.observation_space.high[2], np.radians(50) ]
expert = World.CartPole.Expert(n_bins, lower_bounds, upper_bounds, Q_table)

max_epoch = 500
nTraj = 300
TrainingSet, Labels, Reward = expert.Evaluation(Q_trained, nTraj, max_epoch)

averageExpert = np.sum(Reward)/nTraj

# %% Batch Agent
BatchSim = World.CartPole.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, RewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

averageBatch = np.sum(RewardBatch)/nTraj

# %% Online Agent
OnlineSim = World.CartPole.Simulation(pi_hi_online, pi_lo_online, pi_b_online)
[trajOnline, controlOnline, OptionsOnline, 
 TerminationOnline, RewardOnline] = OnlineSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj)

averageOnline = np.sum(RewardOnline)/nTraj


# %% Plot Likelihood 


x_likelihood_batch = np.linspace(0, len(likelihood_batch)-1, len(likelihood_batch)) 
x_likelihood_online = np.linspace(0,len(likelihood_online)-1,len(likelihood_online))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Batch iterations' , color=color)
ax1.set_ylabel('likelihood')
ax1.plot(x_likelihood_batch, likelihood_batch, '-d', color=color)
ax1.tick_params(axis='x', labelcolor=color)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_xlabel('Online iterations' , color=color)  # we already handled the x-label with ax1
ax2.plot(x_likelihood_online, likelihood_online, color=color)
ax2.tick_params(axis='x', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Figures/likelihood.eps', format='eps')
plt.show()

   