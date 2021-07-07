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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Walker:
    class Expert:
        def heuristic():
                # Heurisic: suboptimal, have no notion of balance.
                env = gym.make("BipedalWalker-v3")
                env.reset()
                steps = 0
                total_reward = 0
                a = np.array([0.0, 0.0, 0.0, 0.0])
                STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
                SPEED = 0.29  # Will fall forward on higher speed
                state = STAY_ON_ONE_LEG
                moving_leg = 0
                supporting_leg = 1 - moving_leg
                SUPPORT_KNEE_ANGLE = +0.1
                supporting_knee_angle = SUPPORT_KNEE_ANGLE
                while True:
                    s, r, done, info = env.step(a)
                    total_reward += r
                    if steps % 20 == 0 or done:
                        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                        print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
                        print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
                        print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
                    steps += 1

                    contact0 = s[8]
                    contact1 = s[13]
                    moving_s_base = 4 + 5*moving_leg
                    supporting_s_base = 4 + 5*supporting_leg

                    hip_targ  = [None,None]   # -0.8 .. +1.1
                    knee_targ = [None,None]   # -0.6 .. +0.9
                    hip_todo  = [0.0, 0.0]
                    knee_todo = [0.0, 0.0]

                    if state==STAY_ON_ONE_LEG:
                        hip_targ[moving_leg]  = 1.1
                        knee_targ[moving_leg] = -0.6
                        supporting_knee_angle += 0.03
                        if s[2] > SPEED: supporting_knee_angle += 0.03
                        supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
                        knee_targ[supporting_leg] = supporting_knee_angle
                        if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                            state = PUT_OTHER_DOWN
                    if state==PUT_OTHER_DOWN:
                        hip_targ[moving_leg]  = +0.1
                        knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                        knee_targ[supporting_leg] = supporting_knee_angle
                        if s[moving_s_base+4]:
                            state = PUSH_OFF
                            supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
                    if state==PUSH_OFF:
                        knee_targ[moving_leg] = supporting_knee_angle
                        knee_targ[supporting_leg] = +1.0
                        if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                            state = STAY_ON_ONE_LEG
                            moving_leg = 1 - moving_leg
                            supporting_leg = 1 - moving_leg

                    if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
                    if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
                    if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
                    if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

                    hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
                    hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
                    knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
                    knee_todo[1] -= 15.0*s[3]

                    a[0] = hip_todo[0]
                    a[1] = knee_todo[0]
                    a[2] = hip_todo[1]
                    a[3] = knee_todo[1]
                    a = np.clip(0.5*a, -1.0, 1.0)

                    env.render()
                    if done: break
                
        def Evaluation(n_episodes, max_epoch_per_traj):
            env = gym.make("BipedalWalker-v3")
            env._max_episode_steps = max_epoch_per_traj
            Reward_array = np.empty((0))
            obs = env.reset()
            size_input = len(obs)
            TrainingSet = np.empty((0,size_input))
            
            action = np.array([0.0, 0.0, 0.0, 0.0])
            size_action = len(action)
            Labels = np.empty((0,size_action))
            
            for e in range(n_episodes):
                
                print(e, '/', n_episodes)
                STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
                SPEED = 0.10  # Will fall forward on higher speed
                state = STAY_ON_ONE_LEG
                moving_leg = 0
                supporting_leg = 1 - moving_leg
                SUPPORT_KNEE_ANGLE = +0.1
                supporting_knee_angle = SUPPORT_KNEE_ANGLE
                
                Reward = 0
                obs = env.reset()
                TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)
                
                # Discretize state into buckets
                done = False
                
                # policy action 
                Labels = np.append(Labels, action.reshape(1,size_action),0)
                
    
                while done==False:
                    
                    obs, reward, done, _ = env.step(action)
                    TrainingSet = np.append(TrainingSet, obs.reshape(1,len(obs)), 0)                    
                    Reward = Reward + reward

                    contact0 = obs[8]
                    contact1 = obs[13]
                    moving_s_base = 4 + 5*moving_leg
                    supporting_s_base = 4 + 5*supporting_leg

                    hip_targ  = [None,None]   # -0.8 .. +1.1
                    knee_targ = [None,None]   # -0.6 .. +0.9
                    hip_todo  = [0.0, 0.0]
                    knee_todo = [0.0, 0.0]

                    if state==STAY_ON_ONE_LEG:
                        hip_targ[moving_leg]  = 1.1
                        knee_targ[moving_leg] = -0.6
                        supporting_knee_angle += 0.03
                        if obs[2] > SPEED: supporting_knee_angle += 0.03
                        supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
                        knee_targ[supporting_leg] = supporting_knee_angle
                        if obs[supporting_s_base+0] < 0.10: # supporting leg is behind
                            state = PUT_OTHER_DOWN
                    if state==PUT_OTHER_DOWN:
                        hip_targ[moving_leg]  = +0.1
                        knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
                        knee_targ[supporting_leg] = supporting_knee_angle
                        if obs[moving_s_base+4]:
                            state = PUSH_OFF
                            supporting_knee_angle = min( obs[moving_s_base+2], SUPPORT_KNEE_ANGLE )
                    if state==PUSH_OFF:
                        knee_targ[moving_leg] = supporting_knee_angle
                        knee_targ[supporting_leg] = +1.0
                        if obs[supporting_s_base+2] > 0.88 or obs[2] > 1.2*SPEED:
                            state = STAY_ON_ONE_LEG
                            moving_leg = 1 - moving_leg
                            supporting_leg = 1 - moving_leg

                    if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - obs[4]) - 0.25*obs[5]
                    if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - obs[9]) - 0.25*obs[10]
                    if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - obs[6])  - 0.25*obs[7]
                    if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - obs[11]) - 0.25*obs[12]

                    hip_todo[0] -= 0.9*(0-obs[0]) - 1.5*obs[1] # PID to keep head strait
                    hip_todo[1] -= 0.9*(0-obs[0]) - 1.5*obs[1]
                    knee_todo[0] -= 15.0*obs[3]  # vertical speed, to damp oscillations
                    knee_todo[1] -= 15.0*obs[3]

                    action[0] = hip_todo[0]
                    action[1] = knee_todo[0]
                    action[2] = hip_todo[1]
                    action[3] = knee_todo[1]
                    action = np.clip(0.5*action, -1.0, 1.0)
                    
                    Labels = np.append(Labels, action.reshape(1,size_action),0)
        
                    # Render the cartpole environment
                    #env.render()
                    
                Reward_array = np.append(Reward_array, Reward) 
                env.close()
                    
            return TrainingSet, Labels, Reward_array  
        
    class Simulation:
        def __init__(self, pi_hi, pi_lo, pi_b, Labels):
            self.env = gym.make("BipedalWalker-v3").env
            option_space = len(pi_lo)
            self.option_space = option_space
            self.mu = np.ones(option_space)*np.divide(1,option_space)
            self.zeta = 0.0001
            self.pi_hi = pi_hi
            self.pi_lo = pi_lo
            self.pi_b = pi_b  
            self.action_dictionary = np.unique(Labels, axis = 0)
            
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
            
        def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories, seed):
            self.env.seed(seed)
            np.random.seed(seed)            
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),2)
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
                o = np.amin(np.where(draw_o<=prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
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
                o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    for i in range(1,prob_u_rescaled.shape[1]):
                        prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.amin(np.where(draw_u<=prob_u_rescaled)[1])
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = self.action_dictionary[u,:]
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,2)
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
                    b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
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
                    o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
            
        
                traj[t] = x
                control[t]=u_tot
                Option[t]=o_tot
                Termination[t]=b_tot
                Reward_array = np.append(Reward_array, Reward)
        
            return traj, control, Option, Termination, Reward_array    
     
        def HierarchicalStochasticSampleTrajMDP_Greedy(self, max_epoch_per_traj, number_of_trajectories, seed):
            self.env.seed(seed)
            np.random.seed(seed)            
            traj = [[None]*1 for _ in range(number_of_trajectories)]
            control = [[None]*1 for _ in range(number_of_trajectories)]
            Option = [[None]*1 for _ in range(number_of_trajectories)]
            Termination = [[None]*1 for _ in range(number_of_trajectories)]
            Reward_array = np.empty((0,0),int)
    
            for t in range(number_of_trajectories):
                done = False
                obs = np.round(self.env.reset(),2)
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
                o = np.amin(np.where(draw_o<=prob_o_rescaled))
                o_tot = np.append(o_tot,o)
        
                # Termination
                state = obs.reshape((1,size_input))
                prob_b = self.pi_b[o](state).numpy()
                prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                for i in range(1,prob_b_rescaled.shape[1]):
                    prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
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
                o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
                o_tot = np.append(o_tot,o)
        
                for k in range(0,max_epoch_per_traj):
                    state = obs.reshape((1,size_input))
                    # draw action
                    prob_u = self.pi_lo[o](state).numpy()
                    # prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                    # for i in range(1,prob_u_rescaled.shape[1]):
                    #     prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                    # draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                    u = np.argmax(prob_u)
                    u_tot = np.append(u_tot,u)
            
                    # given action, draw next state
                    action = self.action_dictionary[u,:]
                    obs, reward, done, _ = self.env.step(action)
                    obs = np.round(obs,2)
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
                    b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
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
                    o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
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
                obs = np.round(self.env.reset(),2)
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
                        action = self.action_dictionary[u,:]
                        obs, reward, done, info = self.env.step(action)
                        obs = np.round(obs,2)
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
        



        
    
    

            
            