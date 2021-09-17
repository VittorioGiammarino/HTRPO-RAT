#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:42:00 2021

@author: vittorio
"""


import numpy as np
import torch
import gym
import argparse
import os
import multiprocessing

import utils
import TD3
import SAC
import GAIL
import TRPO
import UATRPO
import PPO

# %%

with open('DataFromExpert/TrainingSet_continuous.npy', 'rb') as f:
    TrainingSet_tot = np.load(f, allow_pickle=True)

with open('DataFromExpert/Labels_continuous.npy', 'rb') as f:
    Labels_tot = np.load(f, allow_pickle=True)
    
with open('DataFromExpert/Reward_continuous.npy', 'rb') as f:
    Reward = np.load(f, allow_pickle=True)

# %%
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, training_iter, eval_episodes=10):
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
	print(f"Seed {seed}, Iter {training_iter}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def train(args, seed): 
    env = gym.make(args.env)
    
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    	
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = np.zeros((action_dim,))
    for a in range(action_dim):
        max_action[a] = env.action_space.high[a]
    
    # Initialize policy        
    if args.policy == "TRPO":
        kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        }
        # Target policy smoothing is scaled wrt the action scale
        policy = TRPO.TRPO(**kwargs)
        
    # Initialize policy        
    if args.policy == "UATRPO":
        kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        }
        # Target policy smoothing is scaled wrt the action scale
        policy = UATRPO.UATRPO(**kwargs)
        
    if args.policy == "PPO":
        kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        }
        # Target policy smoothing is scaled wrt the action scale
        policy = PPO.PPO(**kwargs)
        
    if args.GAIL:
        kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "expert_states": TrainingSet_tot,
        "expert_actions": Labels_tot,
        }
        IRL = GAIL.Gail(**kwargs)
            	
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, seed, 0)]

    for i in range(int(args.max_iter)):
		
        if args.GAIL:
            rollout_states, rollout_actions = policy.GAE(env, args.GAIL, IRL.discriminator)
            mean_expert_score, mean_learner_score = IRL.update(rollout_states, rollout_actions)
            
            print(f"Expert Score: {mean_expert_score}, Learner Score: {mean_learner_score}")
            
            policy.train()
        else:
            rollout_states, rollout_actions = policy.GAE(env)
            policy.train(Entropy = True)
            

        # Evaluate episode
        if (i + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, seed, i+1))
                
    return evaluations, policy


if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TRPO")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
    parser.add_argument("--Nseed", default=1, type=int)          # Sets Gym, PyTorch and Numpy seeds int(0.5*multiprocessing.cpu_count())
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--number_steps_per_iter", default=5000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)    # Max time steps to run environment
    parser.add_argument("--max_iter", default=3e6/5000, type=int)    # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
    parser.add_argument("--l_rate", default=3e-4)                    # Learning rate
    parser.add_argument("--discount", default=0.99)                  # Discount factor
    parser.add_argument("--tau", default=0.005)                      # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    parser.add_argument("--alpha", default=0.2, type=int)            # SAC entropy regularizer term
    parser.add_argument("--critic_freq", default=2, type=int)        # Frequency of delayed critic updates
    parser.add_argument("--GAIL", default=True)        # Frequency of delayed critic updates
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_Nseed_{args.Nseed}"
    
    if args.GAIL:
        file_name = f"{args.policy}+GAIL_{args.env}_Nseed_{args.Nseed}"
    
    print("---------------------------------------")
    print(f"Policy: {args.policy}, GAIL:{args.GAIL} Env: {args.env}, NSeed: {args.Nseed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
        
    # arguments = [(args, seed) for seed in range(args.Nseed)] 
    # with multiprocessing.Pool(args.Nseed) as pool:
    #     results = pool.starmap(train, arguments)
    #     pool.close()
    #     pool.join()
    
    evaluations, policy = train(args, 0)
    
    # evaluations = []
    # for i in range(args.Nseed):
    #     evaluations.append(results[i][0])
        
    # np.save(f"./results/mean_{file_name}", np.mean(evaluations,0))
    # np.save(f"./results/std_{file_name}", np.std(evaluations,0))
    # np.save(f"./results/steps_{file_name}", np.linspace(0,args.max_timesteps,len(evaluations)))
    if args.save_model: 
        # index = np.argmax(np.max(evaluations,1))
        # policy = results[index][1]
        np.save(f"./results/evaluation_{file_name}", evaluations)
        # policy.save(f"./models/{file_name}")