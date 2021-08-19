#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:15:53 2021

@author: vittorio
"""
import copy
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def HierarchicalStochasticSampleTrajMDP(Hierarchical_policy, env, max_epoch_per_traj, number_of_trajectories):
    traj = [[None]*1 for _ in range(number_of_trajectories)]
    control = [[None]*1 for _ in range(number_of_trajectories)]
    Option = [[None]*1 for _ in range(number_of_trajectories)]
    Termination = [[None]*1 for _ in range(number_of_trajectories)]
    Reward_array = np.empty((0,0),int)
    
    for option in range(0,Hierarchical_policy.option_dim):
        Hierarchical_policy.pi_lo[option].eval()  
        Hierarchical_policy.pi_b[option].eval()
    Hierarchical_policy.pi_hi.eval()

    for t in range(number_of_trajectories):
        done = False
        obs = env.reset()
        size_input = len(obs)
        x = np.empty((0,size_input),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,Hierarchical_policy.action_dim))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        Reward = 0
        state = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
        
        # Initial Option
        prob_o = Hierarchical_policy.pi_hi(state).cpu().data.numpy()
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        temp = np.where(draw_o<=prob_o_rescaled)[1]
        if temp.size == 0:
             o = np.argmax(prob_o)
        else:
             o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)

        # Termination
        prob_b = Hierarchical_policy.pi_b[o](state).cpu().data.numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        temp = np.where(draw_b<=prob_b_rescaled)[1]
        if temp.size == 0:
            b = np.argmax(prob_b)
        else:
            b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
        b_tot = np.append(b_tot,b)
        if b == 1:
            b_bool = True
        else:
            b_bool = False

        o_prob_tilde = np.empty((1,Hierarchical_policy.option_dim))
        if b_bool == True:
            o_prob_tilde = Hierarchical_policy.pi_hi(state).cpu().data.numpy()
        else:
            o_prob_tilde[0,:] = 0
            o_prob_tilde[0,o] = 1 
    
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        temp = np.where(draw_o<=prob_o_rescaled)[1]
        if temp.size == 0:
             o = np.argmax(prob_o)
        else:
             o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)

        for k in range(0,max_epoch_per_traj):
            state = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
            # draw action
            output = Hierarchical_policy.select_action(state, o)
            u_tot = np.append(u_tot, output, axis=0)
            action = output.flatten()
            # env.render()
            # given action, draw next state
            obs, reward, done, _ = env.step(action)
            Reward = Reward + reward
            x = np.append(x, obs.reshape((1,size_input)), axis=0)

            if done == True:
                break
    
            # Select Termination
            # Termination
            state_plus1 = torch.FloatTensor(obs.reshape((1,size_input))).to(device)
            prob_b = Hierarchical_policy.pi_b[o](state_plus1).cpu().data.numpy()
            prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
            for i in range(1,prob_b_rescaled.shape[1]):
                prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
            draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
            temp = np.where(draw_b<=prob_b_rescaled)[1]
            if temp.size == 0:
                b = np.argmax(prob_b)
            else:
                b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
            b_tot = np.append(b_tot,b)
            if b == 1:
                b_bool = True
            else:
                b_bool = False

            o_prob_tilde = np.empty((1,Hierarchical_policy.option_dim))
            if b_bool == True:
                o_prob_tilde = Hierarchical_policy.pi_hi(state_plus1).cpu().data.numpy()
            else:
                o_prob_tilde[0,:] = 0
                o_prob_tilde[0,o] = 1
    
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            temp = np.where(draw_o<=prob_o_rescaled)[1]
            if temp.size == 0:
                 o = np.argmax(prob_o)
            else:
                 o = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
    

        traj[t] = x
        control[t]=u_tot
        Option[t]=o_tot
        Termination[t]=b_tot
        Reward_array = np.append(Reward_array, Reward)
        # env.close()

    return traj, control, Option, Termination, Reward_array    