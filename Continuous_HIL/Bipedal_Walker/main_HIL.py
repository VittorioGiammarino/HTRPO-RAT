#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:15:50 2021

@author: vittorio
"""
import numpy as np
import torch
import gym
import argparse
import os

import BatchBW_HIL_pytorch
import H_TD3
import H_SAC
import H_TD0
import time
import matplotlib.pyplot as plt

from evaluation import HierarchicalStochasticSampleTrajMDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# %% Trial with Gaussian actor
with open('FlatRL_expert/DataFromExpert/TrainingSet_continuous.npy', 'rb') as f:
    TrainingSet_tot = np.load(f, allow_pickle=True)

with open('FlatRL_expert/DataFromExpert/Labels_continuous.npy', 'rb') as f:
    Labels_tot = np.load(f, allow_pickle=True)
    
with open('FlatRL_expert/DataFromExpert/Reward_continuous.npy', 'rb') as f:
    Reward = np.load(f, allow_pickle=True)
    
Options = np.zeros((len(Labels_tot,)), dtype = int)
for i in range(len(Labels_tot)):
    if Labels_tot[i,3] >= 0.9:
        Options[i] = 1
        
# %% 
time = np.linspace(0,len(Labels_tot[100:420,0]),len(Labels_tot[100:420,0])) 
plt.subplot(411)
plot_data = plt.plot(time, Labels_tot[100:420,0], 'b')
plt.subplot(412)
plot_data = plt.plot(time, Labels_tot[100:420,1], 'b')
plt.subplot(413)
plot_data = plt.plot(time, Labels_tot[100:420,2], 'b')
plt.subplot(414)
plot_data = plt.plot(time, Labels_tot[100:420,3], 'b')
plt.xlabel('time')
plt.ylabel('actions')
# plt.title('Options for supervised learning')
# plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
plt.show()     
    
time = np.linspace(0,len(Labels_tot[100:420,0]),len(Labels_tot[100:420,0])) 
plt.subplot(411)
plot_data = plt.plot(time, Labels_tot[100:420,0], 'b', time, Options[100:420], '--r')
plt.subplot(412)
plot_data = plt.plot(time, Labels_tot[100:420,1], 'b', time, Options[100:420], '--r')
plt.subplot(413)
plot_data = plt.plot(time, Labels_tot[100:420,2], 'b', time, Options[100:420], '--r')
plt.subplot(414)
plot_data = plt.plot(time, Labels_tot[100:420,3], 'b', time, Options[100:420], '--r')
plt.xlabel('time')
plt.ylabel('actions')
# plt.title('Options for supervised learning')
# plt.savefig('Figures/FiguresExpert/Expert_Traj_VS_View.eps', format='eps')
plt.show()  

# %% Unsupervised

parser = argparse.ArgumentParser()
#General
parser.add_argument("--number_options", default=2, type=int)     # number of options
parser.add_argument("--policy", default="HSAC")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
parser.add_argument("--seed", default=2, type=int)               # Sets Gym, PyTorch and Numpy seeds
#HIL
parser.add_argument("--HIL", default=True, type=bool)         # Batch size for HIL
parser.add_argument("--size_data_set", default=2000, type=int)         # Batch size for HIL
parser.add_argument("--batch_size_HIL", default=256, type=int)         # Batch size for HIL
parser.add_argument("--maximization_epochs_HIL", default=10, type=int) # Optimization epochs HIL
parser.add_argument("--l_rate_HIL", default=1e-3, type=float)         # Optimization epochs HIL
parser.add_argument("--N_iterations", default=10, type=int)            # Number of EM iterations
parser.add_argument("--pi_hi_supervised", default=True, type=bool)     # Supervised pi_hi
parser.add_argument("--pi_hi_supervised_epochs", default=200, type=int)  
#HTD0
parser.add_argument("--init_critic", default=True, type=bool)   
parser.add_argument("--HTD0_timesteps", default=3e5, type=int)    # Max time steps to run environment
# HRL
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--alpha", default=0.2, type=int)            # SAC entropy regularizer term
parser.add_argument("--critic_freq", default=2, type=int)        # Frequency of delayed critic updates
parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
parser.add_argument("--load_model", default=True, type=bool)                  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--load_model_path", default="") 
# Evaluation
parser.add_argument("--evaluation_episodes", default=10, type=int)
parser.add_argument("--evaluation_max_n_steps", default=2000, type=int)
args = parser.parse_args()

file_name = f"{args.policy}_HIL_{args.HIL}_HTD0_{args.init_critic}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}_HIL_{args.HIL}_HTD0_{args.init_critic}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")
   
if not os.path.exists("./results"):
    os.makedirs("./results")
   
if not os.path.exists("./models"):
    os.makedirs("./models")
    
if not os.path.exists(f"./models/{file_name}"):
    os.makedirs(f"./models/{file_name}")
    
if not os.path.exists("./models/HIL"):
    os.makedirs("./models/HIL")
    
if not os.path.exists("./models/H_TD0"):
    os.makedirs("./models/H_TD0")

env = gym.make(args.env)

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = np.zeros((action_dim,))
for a in range(action_dim):
    max_action[a] = env.action_space.high[a]
option_dim = args.number_options
termination_dim = 2
state_samples = TrainingSet_tot[0:args.size_data_set,:]
action_samples = Labels_tot[0:args.size_data_set,:]
batch_size = args.batch_size_HIL
M_step_epochs = args.maximization_epochs_HIL
l_rate = args.l_rate_HIL

kwargs = {
    "max_action": max_action,
	"state_dim": state_dim,
    "action_dim": action_dim,
    "option_dim": option_dim,
    "termination_dim": termination_dim,
    "state_samples": state_samples,
    "action_samples": action_samples,
    "M_step_epoch": M_step_epochs,
    "batch_size": batch_size,
    "l_rate_pi_lo": l_rate,
    "l_rate_pi_hi": 1e-3,
    "l_rate_pi_b": 1e-3,
    }

Agent_continuous_BatchHIL_pytorch = BatchBW_HIL_pytorch.BatchBW(**kwargs)
N = args.N_iterations
eval_episodes = args.evaluation_episodes
max_epoch = args.evaluation_max_n_steps

# %% train Policy with HIL

if args.HIL:
    if args.pi_hi_supervised:
        epochs = args.pi_hi_supervised_epochs
        Agent_continuous_BatchHIL_pytorch.pretrain_pi_hi(epochs, Options[0:args.size_data_set])
        Labels_b = Agent_continuous_BatchHIL_pytorch.prepare_labels_pretrain_pi_b(Options[0:args.size_data_set])
        for option in range(args.number_options):
            Agent_continuous_BatchHIL_pytorch.pretrain_pi_b(epochs, Labels_b[option], option)
    
    Loss = 100000
    evaluation_HIL = []
    for i in range(N):
        print(f"Iteration {i+1}/{N}")
        loss = Agent_continuous_BatchHIL_pytorch.Baum_Welch()
        if loss > Loss:
            Agent_continuous_BatchHIL_pytorch.reset_learning_rates(l_rate/2, l_rate/2, l_rate/2)
        Loss = loss
        [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
         TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_continuous_BatchHIL_pytorch, env, max_epoch, eval_episodes)
        avg_reward = np.sum(RewardBatch_torch)/eval_episodes
        evaluation_HIL.append(avg_reward)
        
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
    
    # Save
    np.save(f"./results/HIL_{args.env}_{args.seed}", evaluation_HIL)
    Agent_continuous_BatchHIL_pytorch.save(f"./models/HIL/HIL_{args.env}_{args.seed}")
    
# %% Train Critics with HTD0

if args.init_critic:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "max_action": max_action,
        "l_rate_critic": 3e-4, 
        "discount": 0.99,
        "tau": 0.005,
        "eta": 1e-7, 
        }
   
    Train_Critic = H_TD0.H_TD0(**kwargs)
    Train_Critic.load(f"./models/HIL/HIL_{args.env}_{args.seed}")
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    initial_option = 0
    initial_b = 1
    option = Train_Critic.select_option(state, initial_b, initial_option)
    for t in range(int(args.HTD0_timesteps)):
    		
        episode_timesteps += 1
        state = torch.FloatTensor(state.reshape(1,-1)).to(device) 
        action = Train_Critic.select_action(state,option).flatten()
        
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        termination = Train_Critic.select_termination(next_state, option)
        
        if termination == 1:
            cost = Train_Critic.eta
        else:
            cost = 0
    
        # Store data in replay buffer
        Train_Critic.Buffer[option].add(state, action, next_state, reward, cost, done_bool)
        
        next_option = Train_Critic.select_option(next_state, termination, option)
    
        state = next_state
        option = next_option
        episode_reward += reward
    
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            Train_Critic.train(option, args.batch_size)
    
        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            initial_option = 0
            initial_b = 1
            option = Train_Critic.select_option(state, initial_b, initial_option)   
            
    Train_Critic.save(f"./models/H_TD0/H_TD0_{args.env}_{args.seed}")
            
# %% 

if args.policy == "HTD3":
    kwargs = {
    	"state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "max_action": max_action,
        "l_rate_pi_lo": 3e-4,
        "l_rate_pi_hi": 1e-6,
        "l_rate_pi_b": 1e-6,
        "l_rate_critic": 3e-4, 
        "discount": 0.99,
        "tau": 0.005, 
        "eta": 1e-7, 
        "policy_noise": 0.2, 
        "noise_clip": 0.5, 
        "pi_lo_freq": 2, 
        "pi_b_freq": 500,
        "pi_hi_freq": 2e5
        }
    Agent_HRL = H_TD3.H_TD3(**kwargs)
    if args.load_model and args.HIL:
    	Agent_HRL.load_actor(f"./models/HIL/HIL_{args.env}_{args.seed}")
    if args.load_model and args.critic_init:
    	Agent_HRL.load_critic(f"./models/H_TD0/H_TD0_{args.env}_{args.seed}")
        
if args.policy == "HSAC":
    kwargs = {
    	"state_dim": state_dim,
        "action_dim": action_dim,
        "option_dim": option_dim,
        "termination_dim": termination_dim,
        "max_action": max_action,
        "l_rate_pi_lo": 3e-4,
        "l_rate_pi_hi": 3e-4,
        "l_rate_pi_b": 1e-6,
        "l_rate_critic": 3e-4, 
        "discount": 0.99,
        "tau": 0.005, 
        "eta": 1e-7, 
        "pi_b_freq": 500,
        "pi_hi_freq": 1e5,
        "alpha": 0.2,
        "critic_freq": 2
        }
    Agent_HRL = H_SAC.H_SAC(**kwargs)
    if args.load_model and args.HIL:
    	Agent_HRL.load_actor(f"./models/HIL/HIL_{args.env}_{args.seed}")
    if args.load_model and args.init_critic:
    	Agent_HRL.load_critic(f"./models/H_TD0/H_TD0_{args.env}_{args.seed}")

# Evaluate untrained policy
evaluation_HRL = []
[trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
  TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_HRL, env, max_epoch, eval_episodes)
avg_reward = np.sum(RewardBatch_torch)/eval_episodes
evaluation_HRL.append(avg_reward)

print("---------------------------------------")
print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
print("---------------------------------------")

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

initial_option = 0
initial_b = 1

option = Agent_HRL.select_option(state, initial_b, initial_option)

for t in range(int(args.max_timesteps)):
		
    episode_timesteps += 1
    state = torch.FloatTensor(state.reshape(1,-1)).to(device) 

    if t < args.start_timesteps:
        action = env.action_space.sample()    
    elif args.policy == "HTD3":
        output = Agent_HRL.select_action(state,option)
    	# Perform action
        action = (output.flatten() + np.random.normal(0, max_action * args.expl_noise, size=Agent_HRL.action_dim)).clip(-Agent_HRL.max_action, Agent_HRL.max_action)
    elif args.policy == "HSAC":
        action = Agent_HRL.select_action(state,option).flatten()
    
    next_state, reward, done, _ = env.step(action) 
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
    
    termination = Agent_HRL.select_termination(next_state, option)
    
    if termination == 1:
        cost = Agent_HRL.eta
    else:
        cost = 0

    # Store data in replay buffer
    Agent_HRL.Buffer[option].add(state, action, next_state, reward, cost, done_bool)
    
    next_option = Agent_HRL.select_option(next_state, termination, option)

    state = next_state
    option = next_option
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= args.start_timesteps:
        Agent_HRL.train(option, args.batch_size)

    if done: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 
        initial_option = 0
        initial_b = 1
        option = Agent_HRL.select_option(state, initial_b, initial_option)

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
        [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
         TerminationBatch_torch, RewardBatch_torch] = HierarchicalStochasticSampleTrajMDP(Agent_HRL, env, max_epoch, eval_episodes)
        avg_reward = np.sum(RewardBatch_torch)/eval_episodes
        evaluation_HRL.append(avg_reward)
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        
        np.save(f"./results/{file_name}", evaluation_HRL)
        Agent_HRL.save_actor(f"./models/{file_name}/{file_name}")
        Agent_HRL.save_critic(f"./models/{file_name}/{file_name}")

