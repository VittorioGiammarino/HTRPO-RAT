#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Gaussian_Actor, self).__init__()
    
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action
        		
    def forward(self, states):        
        mean = self.net(states)
        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)
        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

        return distb


class Value_net(nn.Module):
    def __init__(self, state_dim):
        super(Value_net, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1

class PPO:
    def __init__(self, state_dim, action_dim, max_action, lr = 3e-4, num_steps_per_rollout=5000, gae_gamma = 0.99, gae_lambda = 0.99, 
                 epsilon = 0.2, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs=10):
        
        self.actor = Gaussian_Actor(state_dim, action_dim, max_action).to(device)
        self.value_function = Value_net(state_dim).to(device)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr)
        self.optimizer_value_function = torch.optim.Adam(self.value_function.parameters(), lr)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.Total_t = 0
        self.Total_iter = 0
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        distb = self.actor(state)
        action = distb.sample().detach().cpu().numpy().flatten()
        return action
        
    def GAE(self, env, GAIL = False, Discriminator = None):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        while step < self.num_steps_per_rollout: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout:            
                action = PPO.select_action(self, state)
            
                self.states.append(state)
                self.actions.append(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                
            episode_states = torch.FloatTensor(episode_states)
            episode_actions = torch.FloatTensor(episode_actions)
            episode_rewards = torch.FloatTensor(episode_rewards)
            episode_gammas = torch.FloatTensor(episode_gammas)
            episode_lambdas = torch.FloatTensor(episode_lambdas)        
            
            if GAIL:
                episode_rewards = - torch.log(Discriminator(episode_states, episode_actions)).squeeze().detach()
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.returns.append(episode_returns)
            self.value_function.eval()
            current_values = self.value_function(episode_states).detach()
            next_values = torch.cat((self.value_function(episode_states)[1:], torch.FloatTensor([[0.]]))).detach()
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            self.advantage.append(episode_advantage)
            self.gammas.append(episode_gammas)
            
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.FloatTensor(np.array(self.actions))

        return rollout_states, rollout_actions
    
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.FloatTensor(np.array(self.actions))
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)        
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/rollout_advantage.std()
        
        self.actor.eval()
        old_log_pi = self.actor(rollout_states).log_prob(rollout_actions).detach()
        
        self.value_function.train()
        self.actor.train()
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_states=rollout_states[minibatch_indices]
            batch_actions = rollout_actions[minibatch_indices]
            batch_returns = rollout_returns[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]
            batch_gammas = rollout_gammas[minibatch_indices]       
            
        
            distb = self.actor(batch_states)
            log_pi = distb.log_prob(batch_actions)
            batch_old_log_pi = old_log_pi[minibatch_indices]
            
            r = torch.exp(log_pi - batch_old_log_pi)
            L_clip = torch.minimum(r*batch_advantage, torch.clip(r, 1-self.epsilon, 1+self.epsilon)*batch_advantage)
            L_vf = (self.value_function(batch_states).squeeze() - batch_returns)**2
            
            if Entropy:
                S = distb.entropy()
            else:
                S = torch.zeros_like(distb.entropy())
                
            self.optimizer_value_function.zero_grad()
            self.optimizer_actor.zero_grad()
            loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S).mean()
            loss.backward()
            self.optimizer_value_function.step()
            self.optimizer_actor.step()        
        
        
        
        
        

        
        
        
        
        
        

            
            
        
            
            
            

        