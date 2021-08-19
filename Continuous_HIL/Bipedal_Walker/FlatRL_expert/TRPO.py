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

class TRPO:
    def __init__(self, state_dim, action_dim, max_action, num_steps_per_rollout=5000, gae_gamma = 0.99, gae_lambda = 0.99, epsilon = 0.01, conj_grad_damping=0.1, lambda_ = 1e-3):
        
        self.actor = Gaussian_Actor(state_dim, action_dim, max_action).to(device)
        self.value_function = Value_net(state_dim).to(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.conj_grad_damping = conj_grad_damping
        self.lambda_ = lambda_
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
                action = TRPO.select_action(self, state)
            
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
    
    def get_flat_grads(f, net):
        flat_grads = torch.cat([grad.view(-1) for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)])
        return flat_grads
    
    def get_flat_params(net):
        return torch.cat([param.view(-1) for param in net.parameters()])
    
    def set_params(net, new_flat_params):
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + np.prod(list(param.shape))
            param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)
            start_idx = end_idx
      
    def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b - Av_func(x)
        p = r
        rsold = r.norm() ** 2
    
        for _ in range(max_iter):
            Ap = Av_func(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.norm() ** 2
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew   
        return x
    
    def rescale_and_linesearch(self, g, s, Hs, L, kld, old_params, max_iter=10, success_ratio=0.1):
        TRPO.set_params(self.actor, old_params)
        L_old = L().detach()
        max_kl = self.epsilon
        
        eta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))
    
        for _ in range(max_iter):
            new_params = old_params + eta * s
    
            TRPO.set_params(self.actor, new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta * s)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params
    
            eta *= 0.7
    
        print("The line search was failed!")
        return old_params
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(self.states)
        rollout_actions = torch.FloatTensor(np.array(self.actions))
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)        
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/rollout_advantage.std()
        
        self.value_function.train()
        old_params = TRPO.get_flat_params(self.value_function).detach()
        old_v = self.value_function(rollout_states).detach()
        
        def constraint():
            return ((old_v - self.value_function(rollout_states))**2).mean()
        
        gradient_constraint = TRPO.get_flat_grads(constraint(), self.value_function)
        
        def Hv(v):
            hessian_v = TRPO.get_flat_grads(torch.dot(gradient_constraint, v), self.value_function).detach()
            return hessian_v
        
        gradient = TRPO.get_flat_grads(((-1)*(self.value_function(rollout_states).squeeze() - rollout_returns)**2).mean(), self.value_function).detach()
        s = TRPO.conjugate_gradient(Hv, gradient).detach()
        Hessian_s = Hv(s).detach()
        alpha = torch.sqrt(2*self.epsilon/torch.dot(s,Hessian_s))
        new_params = old_params + alpha*s
        TRPO.set_params(self.value_function, new_params)
        
        self.actor.train()
        old_params = TRPO.get_flat_params(self.actor).detach()
        old_distb = self.actor(rollout_states)
        
        def L():
            distb = self.actor(rollout_states)
            return (rollout_advantage*torch.exp(distb.log_prob(rollout_actions) - old_distb.log_prob(rollout_actions).detach())).mean()
        
        def kld():
            distb = self.actor(rollout_states)
            old_mean = old_distb.mean.detach()
            old_cov = old_distb.covariance_matrix.sum(-1).detach()
            mean = distb.mean
            cov = distb.covariance_matrix.sum(-1)
            return (0.5)*((old_cov/cov).sum(-1)+(((old_mean - mean) ** 2)/cov).sum(-1)-self.action_dim + torch.log(cov).sum(-1) - torch.log(old_cov).sum(-1)).mean()
        
        grad_kld_old_param = TRPO.get_flat_grads(kld(), self.actor)
        
        def Hv(v):
            hessian_v = TRPO.get_flat_grads(torch.dot(grad_kld_old_param, v), self.actor).detach()
            return hessian_v + self.conj_grad_damping*v
        
        gradient = TRPO.get_flat_grads(L(), self.actor).detach()
        s = TRPO.conjugate_gradient(Hv, gradient).detach()
        Hs = Hv(s).detach()
        new_params = TRPO.rescale_and_linesearch(self, gradient, s, Hs, L, kld, old_params)
        
        if Entropy:
            discounted_casual_entropy = ((-1)*rollout_gammas*self.actor(rollout_states).log_prob(rollout_actions)).mean()
            gradient_discounted_casual_entropy = TRPO.get_flat_grads(discounted_casual_entropy, self.actor)
            new_params += self.lambda_*gradient_discounted_casual_entropy
            
        TRPO.set_params(self.actor, new_params)
        
        
        
        
        
        

            
            
        
            
            
            

        