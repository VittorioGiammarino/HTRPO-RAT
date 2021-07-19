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

import World 
import BatchBW_HIL_tensorflow
import BatchBW_HIL_pytorch
from tensorflow import keras
import time
import matplotlib.pyplot as plt

# %%

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="BipedalWalker-v3")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
            
# %%

with open('TD3_expert/DataFromExpert/TrainingSet_discrete.npy', 'rb') as f:
    TrainingSet_tot = np.load(f, allow_pickle=True)

with open('TD3_expert/DataFromExpert/Labels_discrete.npy', 'rb') as f:
    Labels_tot = np.load(f, allow_pickle=True)
    
with open('TD3_expert/DataFromExpert/Reward_discrete.npy', 'rb') as f:
    Reward = np.load(f, allow_pickle=True)

# %% Expert Policy Generation and simulation tensorflow
TrainingSet = TrainingSet_tot[0:2000,:]
Labels = Labels_tot[0:2000]
option_space = 2
M_step_epoch = 1
size_batch = 33
optimizer = keras.optimizers.Adamax(learning_rate=0.1)
Agent_BatchHIL = BatchBW_HIL_tensorflow.BatchHIL(TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer) 
N=10 #number of iterations for the BW algorithm
start_batch_time = time.time()
pi_hi_batch, pi_lo_batch, pi_b_batch, likelihood_batch, time_per_iteration = Agent_BatchHIL.Baum_Welch(N,1)
end_batch_time = time.time()
Batch_time = end_batch_time-start_batch_time
#evaluation
# max_epoch = 20000
# nTraj = 20
# # BatchSim = World.Walker.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch, Labels)
# BatchSim = World.Walker.Simulation(Agent_BatchHIL.NN_options, Agent_BatchHIL.NN_actions, Agent_BatchHIL.NN_termination, Labels)
# [trajBatch, controlBatch, OptionsBatch, 
#  TerminationBatch, RewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj, 1)

# %% Trial with softmax actor

parser = argparse.ArgumentParser()
parser.add_argument("--number_options", default=2, type=int)     # number of options
parser.add_argument("--policy", default="TD3")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

env = gym.make(args.env)

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
option_dim = args.number_options
termination_dim = 2
state_samples = TrainingSet_tot[0:5000,:]
action_samples = Labels_tot[0:5000]
batch_size = 32
l_rate = 0.001
Agent_BatchHIL_pytorch = BatchBW_HIL_pytorch.SoftmaxHierarchicalActor.BatchBW(state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, batch_size, l_rate)
N=20
eval_episodes = 10
max_epoch = 2000

# %%

for i in range(N):
    print(f"Iteration {i+1}/{N}")
    pi_hi_batch_torch, pi_lo_batch_torch, pi_b_batch_torch, likelihood_batch_torch, time_per_iteration_torch = Agent_BatchHIL_pytorch.Baum_Welch()
    BatchSim_torch = World.Walker.Simulation(pi_hi_batch_torch, pi_lo_batch_torch, pi_b_batch_torch, action_samples)
    [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
     TerminationBatch_torch, RewardBatch_torch] = BatchSim_torch.HierarchicalStochasticSampleTrajMDP_pytorch(max_epoch, eval_episodes)
    avg_reward = np.sum(RewardBatch_torch)/eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

# %% Trial with Gaussian actor
with open('TD3_expert/DataFromExpert/TrainingSet_continuous.npy', 'rb') as f:
    TrainingSet_tot = np.load(f, allow_pickle=True)

with open('TD3_expert/DataFromExpert/Labels_continuous.npy', 'rb') as f:
    Labels_tot = np.load(f, allow_pickle=True)
    
with open('TD3_expert/DataFromExpert/Reward_continuous.npy', 'rb') as f:
    Reward = np.load(f, allow_pickle=True)

parser = argparse.ArgumentParser()
parser.add_argument("--number_options", default=2, type=int)     # number of options
parser.add_argument("--policy", default="TD3")                   # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                  # Discount factor
parser.add_argument("--tau", default=0.005)                      # Target network update rate
parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

env = gym.make(args.env)

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = env.action_space.high[0]
option_dim = args.number_options
termination_dim = 2
state_samples = TrainingSet_tot[0:1000,:]
action_samples = Labels_tot[0:1000,:]
batch_size = 32
M_step_epochs = 50
l_rate = 0.001
Agent_continuous_BatchHIL_pytorch = BatchBW_HIL_pytorch.TanhGaussianHierarchicalActor.BatchBW(max_action, state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, M_step_epochs, batch_size, l_rate)
N=30
eval_episodes = 10
max_epoch = 2000

# %%

for i in range(N):
    print(f"Iteration {i+1}/{N}")
    pi_hi_batch_torch, pi_lo_batch_torch, pi_b_batch_torch, likelihood_batch_torch, time_per_iteration_torch = Agent_continuous_BatchHIL_pytorch.Baum_Welch()
    [trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
     TerminationBatch_torch, RewardBatch_torch] = Agent_continuous_BatchHIL_pytorch.HierarchicalStochasticSampleTrajMDP(env, max_epoch, eval_episodes)
    avg_reward = np.sum(RewardBatch_torch)/eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")



# %%

likelihood = Agent_continuous_BatchHIL_pytorch.likelihood_approximation()

# %%
[trajBatch_torch, controlBatch_torch, OptionsBatch_torch, 
 TerminationBatch_torch, RewardBatch_torch] = Agent_continuous_BatchHIL_pytorch.HierarchicalStochasticSampleTrajMDP(env, max_epoch, eval_episodes)








