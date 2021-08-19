import argparse
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

import TD3
from ddpg_agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")        # OpenAI gym environment name
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
parser.add_argument("--load_model", default="default")          # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--evaluation_version", default = "discrete") #Evaluate the policy as it is, "continuous" or a discretized version of it "discrete"
args = parser.parse_args()
   
file_name = f"{args.policy}_{args.env}_{args.seed}"

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
    
if args.load_model != "":
	policy_file = file_name if args.load_model == "default" else args.load_model
	policy.load(f"./models/{policy_file}")
    

def Evaluate(episodes, step, version):

    reward_list = []
    obs = env.reset()
    size_input = len(obs)
    TrainingSet = np.empty((0,size_input))
            
    action = np.array([0.0, 0.0, 0.0, 0.0])
    size_action = len(action)
    Labels = np.empty((0,size_action))    

    for i in range(episodes):

        obs = env.reset()
        score = 0
        Reward = 0
        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
        # Discretize state into buckets
        done = False
                
        # policy action 
        Labels = np.append(Labels, action.reshape(1,size_action),0)

        for t in range(step):

            env.render()
            
            if version == "continuous":
                action = policy.select_action(np.array(obs))
            else:
                action = np.round(policy.select_action(np.array(obs)),0)
            Labels = np.append(Labels, action.reshape(1,size_action),0)
            obs, reward, done, info = env.step(action)
            TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    
            Reward = Reward + reward
            obs = obs.squeeze()
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(score, i, episodes))
                break

        reward_list.append(score)

    print('Training saved')
    env.close()
    
    return TrainingSet, Labels, reward_list


TrainingSet, Labels, scores = Evaluate(20, 2000, args.evaluation_version)

fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

with open(f'DataFromExpert/TrainingSet_{args.evaluation_version}.npy', 'wb') as f:
    np.save(f, TrainingSet)
    
with open(f'DataFromExpert/Labels_{args.evaluation_version}.npy', 'wb') as f:
    np.save(f, Labels)
    
with open(f'DataFromExpert/Reward_{args.evaluation_version}.npy', 'wb') as f:
    np.save(f, scores)
    
    