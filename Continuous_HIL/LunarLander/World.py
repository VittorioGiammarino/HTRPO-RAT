#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import gym
from sklearn.preprocessing import KBinsDiscretizer
import time, math, random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                      
class LunarLander:
    class Expert:
        def heuristic(s):
            env = gym.make("LunarLander-v2")
            """
            The heuristic for
            1. Testing
            2. Demonstration rollout.

            Args:
                env: The environment
                s (list): The state. Attributes:
                      s[0] is the horizontal coordinate
                      s[1] is the vertical coordinate
                      s[2] is the horizontal speed
                      s[3] is the vertical speed
                      s[4] is the angle
                      s[5] is the angular speed
                      s[6] 1 if first leg has contact, else 0
                      s[7] 1 if second leg has contact, else 0
                returns:
                    a: The heuristic to be fed into the step function defined above to determine the next step and reward.
                """

            angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center
            if angle_targ > 0.4: angle_targ = 0.4    # more than 0.4 radians (22 degrees) is bad
            if angle_targ < -0.4: angle_targ = -0.4
            hover_targ = 0.55*np.abs(s[0])           # target y should be proportional to horizontal offset

            angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
            hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

            if s[6] or s[7]:  # legs have contact
                angle_todo = 0
                hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

            if env.continuous:
                a = np.array([hover_todo*20 - 1, -angle_todo*20])
                a = np.clip(a, -1, +1)
            else:
                a = 0
                if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
                elif angle_todo < -0.05: a = 3
                elif angle_todo > +0.05: a = 1
            return a

        def demo_heuristic_lander(seed=None, render=False):
            env = gym.make("LunarLander-v2")
            env.seed(seed)
            total_reward = 0
            steps = 0
            s = env.reset()
            while True:
                a = LunarLander.Expert.heuristic(s)
                s, r, done, info = env.step(a)
                total_reward += r

                if render:
                    still_open = env.render()
                    if still_open == False: 
                        break

                if steps % 20 == 0 or done:
                    print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1
                if done: break
            return total_reward
        
        def Evaluation(n_episodes, max_epoch_per_traj, seed = None):
            env = gym.make("LunarLander-v2")
            env._max_episode_steps = max_epoch_per_traj
            Reward_array = np.empty((0))
            obs = env.reset()
            size_input = len(obs)
            TrainingSet = np.empty((0,size_input))
            Labels = np.empty((0))
            
            for e in range(n_episodes):
                
                print(e, '/', n_episodes)
                Reward = 0
                obs = env.reset()
                TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
                # Discretize state into buckets
                done = False
                
                # policy action 
                action = LunarLander.Expert.heuristic(obs)
                Labels = np.append(Labels, action)
                
    
                while done==False:
                    
                    # increment enviroment
                    obs, reward, done, _ = env.step(action)
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                    Reward = Reward + reward
        
                    
                    # policy action 
                    action = LunarLander.Expert.heuristic(obs)
                    Labels = np.append(Labels, action)
        
                    # Render the cartpole environment
                    env.render()
                    
                Reward_array = np.append(Reward_array, Reward) 
                env.close()
                    
            return TrainingSet, Labels, Reward_array            

    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b, Labels):
            self.env = gym.make("LunarLander-v2").env
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            self.action_dictionary = np.unique(Labels)
            
        def HierarchicalStochasticSampleTrajMDP_pytorch(self, max_epoch_per_traj, number_of_trajectories):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
                Reward = 0
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
                prob_b = self.pi_b[o](state).cpu().data.numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).cpu().data.numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
                    # draw action
                    prob_u = self.pi_lo[o](state).cpu().data.numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = int(self.action_dictionary[u])
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,3)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
                    prob_b = self.pi_b[o](state_plus1).cpu().data.numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).cpu().data.numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
                Reward = 0
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = int(self.action_dictionary[u])
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,3)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state_plus1).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    

        def HILVideoSimulation(self, directory, max_epoch_per_traj):
            self.env._max_episode_steps = max_epoch_per_traj
    
            # Record the environment
            self.env = gym.wrappers.Monitor(self.env, directory, resume=True)

            for t in range(1):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                while not done: # Start with while True
                    self.env.render()
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state = obs.reshape((1,size_input))
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                        u_tot = np.append(u_tot,u)
            
                        # given action, draw next state
                        action = int(self.action_dictionary[u])
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,3)
                        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                        if done == True:
                            u_tot = np.append(u_tot,0.5)
                            break
            
                        # Select Termination
                        # Termination
                        state_plus1 = obs.reshape((1,size_input))
                        prob_b = self.pi_b[o](state_plus1).numpy()
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi(state_plus1).numpy()
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
                    
            self.env.close()
            return x, u_tot, o_tot, b_tot           


class Acrobot:
    class Expert:
        class CEMOptimizer:
            def __init__(self, weights_dim, batch_size=1000, deviation=10, deviation_lim=100, rho=0.1, eta=0.1, mean=None):
                self.rho = rho
                self.eta = eta
                self.weights_dim = weights_dim
                self.mean = mean if mean!=None else np.zeros(weights_dim)
                self.deviation = np.full(weights_dim, deviation)
                self.batch_size = batch_size
                self.select_num = int(batch_size * rho)
                self.deviation_lim = deviation_lim

                assert(self.select_num > 0)

            def update_weights(self, weights, rewards):
                rewards = np.array(rewards).flatten()
                weights = np.array(weights)
                sorted_idx = (-rewards).argsort()[:self.select_num]
                top_weights = weights[sorted_idx]
                top_weights = np.reshape(top_weights, (self.select_num, self.weights_dim))
                self.mean = np.sum(top_weights, axis=0) / self.select_num
                self.deviation = np.std(top_weights, axis=0)
                self.deviation[self.deviation > self.deviation_lim] = self.deviation_lim
                if(len(self.deviation)!=self.weights_dim):
                    print("dim error")
                    print(len(self.deviation))
                    print(self.weights_dim)
                    exit()


            def sample_batch_weights(self):
                return [np.random.normal(self.mean, self.deviation * (1 + self.eta)) \
                        for _ in range(self.batch_size)]

            def get_weights(self):
                return self.mean

        def train():
            def select_action(ob, weights):
                b1 = np.reshape(weights[0], (1, 1))
                w1 = np.reshape(weights[1:4], (1, 3))
                b2 = np.reshape(weights[4:7], (3, 1))
                w2 = np.reshape(weights[7:16], (3, 3))
                w3 = np.reshape(weights[16:25], (3, 3))
                b3 = np.reshape(weights[25:], (3, 1))
                ob = np.reshape(ob, (3, 1))
                action = np.dot(w1, np.tanh(np.dot(w2, np.tanh(np.dot(w3, ob) - b3)) - b2)) - b1
                return np.tanh(action) * 2

            opt = Pendulum.Expert.CEMOptimizer(3*3+3*3+3*1+3*1+3*1+1, 500, rho=0.01, eta=0.3, deviation=10, deviation_lim=20)
            env = gym.make("Acrobot-v1")
            #env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-3', force=True)
            epoch = 80
            run_times = 10

            def test():
                W = opt.get_weights()
                observation = env.reset()
                accreward = 0
                while True:
                    #env.render()
                    action = select_action(observation, W)
                    observation, reward, done, info = env.step(action)
                    accreward += reward
                    if done:
                        print("test end with reward: {}".format(accreward))
                        break

            for ep in range(epoch):
                print("start epoch {}".format(ep))
                weights = opt.sample_batch_weights()
                rewards = []
                opt.eta *= 0.99
                print("deviation mean = {}".format(np.mean(opt.deviation)))
                for b in range(opt.batch_size):
                    accreward = 0
                    for _ in range(run_times):  
                        observation = env.reset()  
                        while True:
                            action = select_action(observation, weights[b])
                            observation, reward, done, info = env.step(action)
                            accreward += reward
                            if done:
                                break
                    rewards.append(accreward)
                opt.update_weights(weights, rewards)
                test()
                
            return opt.get_weights()
        
        def Evaluation(W, n_episodes, max_epoch_per_traj):
            def select_action(ob, weights):
                b1 = np.reshape(weights[0], (1, 1))
                w1 = np.reshape(weights[1:4], (1, 3))
                b2 = np.reshape(weights[4:7], (3, 1))
                w2 = np.reshape(weights[7:16], (3, 3))
                w3 = np.reshape(weights[16:25], (3, 3))
                b3 = np.reshape(weights[25:], (3, 1))
                ob = np.reshape(ob, (3, 1))
                action = np.dot(w1, np.tanh(np.dot(w2, np.tanh(np.dot(w3, ob) - b3)) - b2)) - b1
                return np.tanh(action) * 2
            
            env = gym.make("Acrobot-v1")
            env._max_episode_steps = max_epoch_per_traj
            obs = env.reset()
            size_input = len(obs)
            Reward_array = np.empty((0))
            TrainingSet = np.empty((0,size_input))
            Labels = np.empty((0))
            
            for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
                    accreward = 0
                    obs = env.reset()
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                    done = False
                
                
                    # policy action 
                    action = select_action(obs, W)
                    Labels = np.append(Labels, action)
                
    
                    while done==False:
                    
                        # increment enviroment
                        obs, reward, done, _ = env.step(action)
                        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                        accreward += reward
        
                    
                        # policy action 
                        action = select_action(obs, W)
                        Labels = np.append(Labels, action)
        
                        # Render the cartpole environment
                        #self.env.render()
                    
                    Reward_array = np.append(Reward_array, accreward) 
                    
            return TrainingSet, Labels, Reward_array              
            
                
        class Expert_Q_learning:
            # =============================================================================
            #         Credit: Richard Brooker https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1/blob/master/cartpole.ipynb
            # =============================================================================
            def __init__(self, n_bins, Q_table):
                self.env = gym.make('Acrobot-v1')
                self.n_bins = n_bins
                self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1], self.env.observation_space.low[2], 
                                     self.env.observation_space.low[3], self.env.observation_space.low[4], self.env.observation_space.low[5]]
                self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1], self.env.observation_space.high[2],
                                     self.env.observation_space.high[3], self.env.observation_space.high[4], self.env.observation_space.high[5]]
                self.Q_table = Q_table
                self.action_dictionary = np.array([-1, 1])
        
            def discretizer(self, cos_theta1, sin_theta1, cos_theta2, sin_theta2, thetadot1, thetadot2) -> Tuple[int,...]:
                """Convert continues state intro a discrete state"""
                est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
                est.fit([self.lower_bounds, self.upper_bounds ])
                return tuple(map(int,est.transform([[cos_theta1, sin_theta1, cos_theta2, sin_theta2, thetadot1, thetadot2]])[0]))
        
            def policy(self, state : tuple ):
                """Choosing action based on epsilon-greedy policy"""
                return np.argmax(self.Q_table[state])
        
            def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
                """Temperal diffrence for updating Q-value of state-action pair"""
                future_optimal_value = np.max(self.Q_table[new_state])
                learned_value = reward + discount_factor * future_optimal_value
                return learned_value
        
            # Adaptive learning of Learning Rate
            def learning_rate(self, n : int , min_rate=0.1 ) -> float  :
                """Decaying learning rate"""
                return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
        
            def exploration_rate(n : int, min_rate= 0.1 ) -> float :
                """Decaying exploration rate"""
                return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
        
            def Training(self, n_episodes):
                for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
    
                # Discretize state into buckets
                    current_state, done = Acrobot.Expert.Expert_Q_learning.discretizer(self,*self.env.reset()), False
    
                    while done==False:
        
                    # policy action 
                        action_index = Acrobot.Expert.Expert_Q_learning.policy(self, current_state) # exploit
                        action = self.action_dictionary[action_index]
        
                    # insert random action
                        if np.random.random() < Acrobot.Expert.Expert_Q_learning.exploration_rate(e) : 
                            action = self.env.action_space.sample() # explore 
                            if action == 0:
                                action = -1
         
                        # increment enviroment
                        obs, reward, done, _ = self.env.step(action)
                        new_state = Acrobot.Expert.Expert_Q_learning.discretizer(self, *obs)
                        
                        #get action index
                        action_index = np.where(action == self.action_dictionary)[0]
        
                        # Update Q-Table
                        lr = Acrobot.Expert.learning_rate(self, e)
                        learnt_value = Acrobot.Expert.Expert_Q_learning.new_Q_value(self, reward , new_state)
                        old_value = self.Q_table[current_state][action_index]
                        self.Q_table[current_state][action_index] = (1-lr)*old_value + lr*learnt_value
        
                        current_state = new_state
        
                        # Render the cartpole environment
                        #self.env.render()
                    
                return self.Q_table
            
            def Evaluation(self, Q_trained, n_episodes, max_epoch_per_traj):
                self.env._max_episode_steps = max_epoch_per_traj
                Reward_array = np.empty((0))
                obs = self.env.reset()
                size_input = len(obs)
                TrainingSet = np.empty((0,size_input))
                Labels = np.empty((0))
            
                for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
                    Reward = 0
                    obs = self.env.reset()
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
                    # Discretize state into buckets
                    current_state, done = Acrobot.Expert.Expert_Q_learning.discretizer(self,*obs), False
                
                    # policy action 
                    action_index = np.argmax(Q_trained[current_state]) # exploit
                    action = self.action_dictionary[action_index]
                    Labels = np.append(Labels, action)
                
    
                    while done==False:
                    
                        # increment enviroment
                        obs, reward, done, _ = self.env.step(action)
                        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                        new_state = Acrobot.Expert.Expert_Q_learning.discretizer(self, *obs)
                        Reward = Reward + reward
        
                        current_state = new_state
                    
                        # policy action 
                        action_index = np.argmax(Q_trained[current_state]) # exploit
                        action = self.action_dictionary[action_index]
                        Labels = np.append(Labels, action)
        
                        # Render the cartpole environment
                        self.env.render()
                    
                    Reward_array = np.append(Reward_array, Reward) 
                    self.env.close()
                    
                return TrainingSet, Labels, Reward_array         

    def Plot(x, u, o, b, name_file):
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plot_action = plt.scatter(x[:,1], x[:,2], c=o, marker='x', cmap='cool');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Option1', 'Option2'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(312, sharex=ax1)
        plot_action = plt.scatter(x[:,1], x[:,2], c=u, marker='x', cmap='winter');
        cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        plot_action = plt.scatter(x[0:-1,1], x[0:-1,2], c=b, marker='x', cmap='copper');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
        plt.xlabel('Pole Angle')
        plt.ylabel('Pole Velocity')
        plt.savefig(name_file, format='eps')
        plt.show()
            
    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b, Labels):
            self.env = gym.make("Acrobot-v1").env
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            self.action_dictionary = np.unique(Labels)
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
                Reward = 0
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = int(self.action_dictionary[u])
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,3)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state_plus1).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    

        def HILVideoSimulation(self, directory, max_epoch_per_traj):
            self.env._max_episode_steps = max_epoch_per_traj
    
            # Record the environment
            self.env = gym.wrappers.Monitor(self.env, directory, resume=True)

            for t in range(1):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                while not done: # Start with while True
                    self.env.render()
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state = obs.reshape((1,size_input))
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                        u_tot = np.append(u_tot,u)
            
                        # given action, draw next state
                        action = int(self.action_dictionary[u])
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,3)
                        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                        if done == True:
                            u_tot = np.append(u_tot,0.5)
                            break
            
                        # Select Termination
                        # Termination
                        state_plus1 = obs.reshape((1,size_input))
                        prob_b = self.pi_b[o](state_plus1).numpy()
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi(state_plus1).numpy()
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
                    
            self.env.close()
            return x, u_tot, o_tot, b_tot           




class Pendulum:
    class Expert:
        class CEMOptimizer:
            def __init__(self, weights_dim, batch_size=1000, deviation=10, deviation_lim=100, rho=0.1, eta=0.1, mean=None):
                self.rho = rho
                self.eta = eta
                self.weights_dim = weights_dim
                self.mean = mean if mean!=None else np.zeros(weights_dim)
                self.deviation = np.full(weights_dim, deviation)
                self.batch_size = batch_size
                self.select_num = int(batch_size * rho)
                self.deviation_lim = deviation_lim

                assert(self.select_num > 0)

            def update_weights(self, weights, rewards):
                rewards = np.array(rewards).flatten()
                weights = np.array(weights)
                sorted_idx = (-rewards).argsort()[:self.select_num]
                top_weights = weights[sorted_idx]
                top_weights = np.reshape(top_weights, (self.select_num, self.weights_dim))
                self.mean = np.sum(top_weights, axis=0) / self.select_num
                self.deviation = np.std(top_weights, axis=0)
                self.deviation[self.deviation > self.deviation_lim] = self.deviation_lim
                if(len(self.deviation)!=self.weights_dim):
                    print("dim error")
                    print(len(self.deviation))
                    print(self.weights_dim)
                    exit()


            def sample_batch_weights(self):
                return [np.random.normal(self.mean, self.deviation * (1 + self.eta)) \
                        for _ in range(self.batch_size)]

            def get_weights(self):
                return self.mean

        def train():
            def select_action(ob, weights):
                b1 = np.reshape(weights[0], (1, 1))
                w1 = np.reshape(weights[1:4], (1, 3))
                b2 = np.reshape(weights[4:7], (3, 1))
                w2 = np.reshape(weights[7:16], (3, 3))
                w3 = np.reshape(weights[16:25], (3, 3))
                b3 = np.reshape(weights[25:], (3, 1))
                ob = np.reshape(ob, (3, 1))
                action = np.dot(w1, np.tanh(np.dot(w2, np.tanh(np.dot(w3, ob) - b3)) - b2)) - b1
                return np.tanh(action) * 2

            opt = Pendulum.Expert.CEMOptimizer(3*3+3*3+3*1+3*1+3*1+1, 500, rho=0.01, eta=0.3, deviation=10, deviation_lim=20)
            env = gym.make("Pendulum-v0")
            #env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-3', force=True)
            epoch = 80
            run_times = 10

            def test():
                W = opt.get_weights()
                observation = env.reset()
                accreward = 0
                while True:
                    #env.render()
                    action = select_action(observation, W)
                    observation, reward, done, info = env.step(action)
                    accreward += reward
                    if done:
                        print("test end with reward: {}".format(accreward))
                        break

            for ep in range(epoch):
                print("start epoch {}".format(ep))
                weights = opt.sample_batch_weights()
                rewards = []
                opt.eta *= 0.99
                print("deviation mean = {}".format(np.mean(opt.deviation)))
                for b in range(opt.batch_size):
                    accreward = 0
                    for _ in range(run_times):  
                        observation = env.reset()  
                        while True:
                            action = select_action(observation, weights[b])
                            observation, reward, done, info = env.step(action)
                            accreward += reward
                            if done:
                                break
                    rewards.append(accreward)
                opt.update_weights(weights, rewards)
                test()
                
            return opt.get_weights()
        
        def Evaluation(W, n_episodes, max_epoch_per_traj):
            def select_action(ob, weights):
                b1 = np.reshape(weights[0], (1, 1))
                w1 = np.reshape(weights[1:4], (1, 3))
                b2 = np.reshape(weights[4:7], (3, 1))
                w2 = np.reshape(weights[7:16], (3, 3))
                w3 = np.reshape(weights[16:25], (3, 3))
                b3 = np.reshape(weights[25:], (3, 1))
                ob = np.reshape(ob, (3, 1))
                action = np.dot(w1, np.tanh(np.dot(w2, np.tanh(np.dot(w3, ob) - b3)) - b2)) - b1
                return np.tanh(action) * 2
            
            env = gym.make("Pendulum-v0")
            env._max_episode_steps = max_epoch_per_traj
            obs = env.reset()
            size_input = len(obs)
            Reward_array = np.empty((0))
            TrainingSet = np.empty((0,size_input))
            Labels = np.empty((0))
            
            for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
                    accreward = 0
                    obs = env.reset()
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                    done = False
                
                
                    # policy action 
                    action = select_action(obs, W)
                    Labels = np.append(Labels, action)
                
    
                    while done==False:
                    
                        # increment enviroment
                        obs, reward, done, _ = env.step(action)
                        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                        accreward += reward
        
                    
                        # policy action 
                        action = select_action(obs, W)
                        Labels = np.append(Labels, action)
        
                        # Render the cartpole environment
                        #self.env.render()
                    
                    Reward_array = np.append(Reward_array, accreward) 
                    
            return TrainingSet, Labels, Reward_array              
            
                

  
        class Expert_Q_learning:
            # =============================================================================
            #         Credit: Richard Brooker https://github.com/RJBrooker/Q-learning-demo-Cartpole-V1/blob/master/cartpole.ipynb
            # =============================================================================
            def __init__(self, n_bins, n_bins_action, Q_table):
                self.env = gym.make('Pendulum-v0')
                self.n_bins = n_bins
                self.n_bins_action = n_bins_action
                self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1], self.env.observation_space.low[2]]
                self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1], self.env.observation_space.high[2]]
                self.lower_bound_torque = [self.env.action_space.low[0]]
                self.upper_bound_torque = [self.env.action_space.high[0]]
                self.Q_table = Q_table
        
            def discretizer(self, cos_theta, sin_theta, theta_dot) -> Tuple[int,...]:
                """Convert continues state intro a discrete state"""
                est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
                est.fit([self.lower_bounds, self.upper_bounds])
                return tuple(map(int,est.transform([[cos_theta, sin_theta, theta_dot]])[0]))
        
            def action_discretizer(self, action) -> Tuple[int,...]:
                est = KBinsDiscretizer(n_bins=self.n_bins_action, encode='ordinal', strategy='uniform')
                est.fit([self.lower_bound_torque, self.upper_bound_torque])
                return tuple(map(int,est.transform([[action]])[0]))          
        
            def policy(self, state : tuple ):
                """Choosing action based on epsilon-greedy policy"""
                action_index = np.argmax(self.Q_table[state])
                return action_index
        
            def new_Q_value(self, reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
                """Temperal diffrence for updating Q-value of state-action pair"""
                future_optimal_value = np.max(self.Q_table[new_state])
                learned_value = reward + discount_factor * future_optimal_value
                return learned_value
        
            # Adaptive learning of Learning Rate
            def learning_rate(self, n : int , min_rate=0.1 ) -> float  :
                """Decaying learning rate"""
                return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
        
            def exploration_rate(n : int, min_rate=0.1 ) -> float :
                """Decaying exploration rate"""
                return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))
        
            def Training(self, n_episodes):
                action_array = np.linspace(self.lower_bound_torque,self.upper_bound_torque,self.n_bins_action)
                for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
    
                    # Discretize state into buckets
                    current_state, done = Pendulum.Expert.discretizer(self,*self.env.reset()), False
    
                    while done==False:
        
                        # policy action 
                        action_index = Pendulum.Expert.policy(self, current_state) # exploit
        
                        # insert random action
                        if np.random.random() < Pendulum.Expert.exploration_rate(e) : 
                            action_continuous = self.env.action_space.sample()[0] # explore 
                            action_index = Pendulum.Expert.action_discretizer(self,action_continuous)
         
                        # increment enviroment
                        action = action_array[action_index]
                        obs, reward, done, _ = self.env.step(action)
                        new_state = Pendulum.Expert.discretizer(self, *obs)
        
                        # Update Q-Table
                        lr = Pendulum.Expert.learning_rate(self, e)
                        learnt_value = Pendulum.Expert.new_Q_value(self, reward , new_state)
                        old_value = self.Q_table[current_state][action_index]
                        self.Q_table[current_state][action_index] = (1-lr)*old_value + lr*learnt_value
        
                        current_state = new_state
        
                        # Render the cartpole environment
                        #self.env.render()
                    
                return self.Q_table
            
            def Evaluation(self, Q_trained, n_episodes, max_epoch_per_traj):
                self.env._max_episode_steps = max_epoch_per_traj
                Reward_array = np.empty((0))
                obs = self.env.reset()
                size_input = len(obs)
                TrainingSet = np.empty((0,size_input))
                Labels = np.empty((0))
            
                action_array = np.linspace(self.lower_bound_torque,self.upper_bound_torque,self.n_bins_action)
                for e in range(n_episodes):
                
                    print(e, '/', n_episodes)
                    Reward = 0
                    obs = self.env.reset()
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
                    # Discretize state into buckets
                    current_state, done = Pendulum.Expert.discretizer(self,*obs), False
                
                    # policy action 
                    action_index = np.argmax(Q_trained[current_state]) # exploit
                    Labels = np.append(Labels, action_index)
                
    
                    while done==False:
                    
                        # increment enviroment
                        action = action_array[action_index]
                        obs, reward, done, _ = self.env.step(action)
                        TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    

                        new_state = Pendulum.Expert.discretizer(self, *obs)
                        Reward = Reward + reward
        
                        current_state = new_state
                    
                        # policy action 
                        action_index = np.argmax(Q_trained[current_state]) # exploit
                        Labels = np.append(Labels, action_index)
        
                        # Render the cartpole environment
                        #self.env.render()
                    
                    Reward_array = np.append(Reward_array, Reward) 
                    
                return TrainingSet, Labels, Reward_array  

    def Plot(x, u, o, b, name_file):
        fig = plt.figure()
        ax1 = plt.subplot(311)
        plot_action = plt.scatter(x[:,1], x[:,2], c=o, marker='x', cmap='cool');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Option1', 'Option2'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(312, sharex=ax1)
        plot_action = plt.scatter(x[:,1], x[:,2], c=u, marker='x', cmap='winter');
        cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
        #plt.xlabel('Position')
        plt.ylabel('Pole Velocity')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        plot_action = plt.scatter(x[0:-1,1], x[0:-1,2], c=b, marker='x', cmap='copper');
        cbar = fig.colorbar(plot_action, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
        plt.xlabel('Pole Angle')
        plt.ylabel('Pole Velocity')
        plt.savefig(name_file, format='eps')
        plt.show()
            
    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b, Labels):
            self.env = gym.make("Pendulum-v0")
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            self.action_dictionary = np.unique(Labels)
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories):
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
                Reward = 0
        
                # Initial Option
                prob_o = self.mu
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[0]):
                    prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                b_tot = np.append(b_tot,b)
                if b == 1:
                    b_bool = True
                else:
                    b_bool = False
        
                o_prob_tilde = np.empty((1,self.option_space))
                if b_bool == True:
                    o_prob_tilde = self.pi_hi(state).numpy()
                else:
                    o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                    o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                prob_o = o_prob_tilde
                prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                for i in range(1,prob_o_rescaled.shape[1]):
                    prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action, = [[self.action_dictionary[u]]]
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,3)
                    Reward = Reward + reward
                    x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                    if done == True:
                        u_tot = np.append(u_tot,0.5)
                        break
            
                    # Select Termination
                    # Termination
                    state_plus1 = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state_plus1).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state_plus1).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    

        def HILVideoSimulation(self, directory, max_epoch_per_traj):
            self.env._max_episode_steps = max_epoch_per_traj
    
            # Record the environment
            self.env = gym.wrappers.Monitor(self.env, directory, resume=True)

            for t in range(1):
                done = False
                obs = np.round(self.env.reset(),3)
                size_input = len(obs)
                x = np.empty((0,size_input),int)
                x = np.append(x, obs.reshape((1,size_input)), axis=0)
                u_tot = np.empty((0,0))
                o_tot = np.empty((0,0),int)
                b_tot = np.empty((0,0),int)
        
                while not done: # Start with while True
                    self.env.render()
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state = obs.reshape((1,size_input))
                    prob_b = self.pi_b[o](state).numpy()
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi(state).numpy()
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state = obs.reshape((1,size_input))
                        # draw action
                        prob_u = self.pi_lo[o](state).numpy()
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
                        u_tot = np.append(u_tot,u)
            
                        # given action, draw next state
                        action = [[self.action_dictionary[u]]]
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,3)
                        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
                        if done == True:
                            u_tot = np.append(u_tot,0.5)
                            break
            
                        # Select Termination
                        # Termination
                        state_plus1 = obs.reshape((1,size_input))
                        prob_b = self.pi_b[o](state_plus1).numpy()
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi(state_plus1).numpy()
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
                    
            self.env.close()
            return x, u_tot, o_tot, b_tot           


        
    
    

            
            