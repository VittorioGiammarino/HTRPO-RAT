#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:58:25 2020

@author: vittorio
"""
import copy
import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftmaxHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = F.relu(self.l1(state))
            return self.lS(self.l2(a))
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(SoftmaxHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,termination_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = F.relu(self.l1(state))
            return self.lS(self.l2(b))            
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(SoftmaxHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,option_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = F.relu(self.l1(state))
            return self.lS(self.l2(o))

    class BatchBW(object):
        def __init__(self, state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, batch_size, l_rate):
            self.state_dim = state_dim
            self.action_dim = action_dim 
            self.option_dim = option_dim
            self.termination_dim = termination_dim
            self.TrainingSet = state_samples
            self.batch_size = batch_size
            self.mu = np.ones(option_dim)*np.divide(1,option_dim)
            self.action_space_discrete = len(np.unique(action_samples,axis=0))
            self.action_dictionary = np.unique(action_samples, axis = 0)
            labels = np.zeros((len(action_samples)))
            for i in range(len(action_samples)):
                for j in range(self.action_space_discrete):
                    if np.sum(action_samples[i,:] == self.action_dictionary[j,:]) == self.action_dim:
                        labels[i] = j     
            self.Labels = labels  
            # define hierarchical policy
            self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            self.pi_b = [[None]*1 for _ in range(option_dim)] 
            self.pi_lo = [[None]*1 for _ in range(option_dim)] 
            pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, self.action_space_discrete).to(device)
            pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            for option in range(self.option_dim):
                self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
                self.pi_b[option] = copy.deepcopy(pi_b_temp)
            # define optimizer 
            self.learning_rate = l_rate
            self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=self.learning_rate)
            self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
            self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)] 
            for option in range(self.option_dim):
                self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=self.learning_rate)
                self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=self.learning_rate)            

        def Pi_hi(ot, Pi_hi_parameterization, state):
            Pi_hi = Pi_hi_parameterization(state).cpu()
            o_prob = Pi_hi[0,ot]
            return o_prob

        def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space):
            if b == True:
                o_prob_tilde = SoftmaxHierarchicalActor.BatchBW.Pi_hi(ot, Pi_hi_parameterization, state)
            elif ot == ot_past:
                o_prob_tilde =  torch.FloatTensor([1])
            else:
                o_prob_tilde =  torch.FloatTensor([0])
            
            return o_prob_tilde

        def Pi_lo(a, Pi_lo_parameterization, state):
            Pi_lo = Pi_lo_parameterization(state).cpu()
            a_prob = Pi_lo[0,int(a)]
        
            return a_prob

        def Pi_b(b, Pi_b_parameterization, state):
            Pi_b = Pi_b_parameterization(state).cpu()
            if b == True:
                b_prob = Pi_b[0,1]
            else:
                b_prob = Pi_b[0,0]
            return b_prob
    
        def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, option_space):
            Pi_hi_eval = SoftmaxHierarchicalActor.BatchBW.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space).clamp(0.0001,1)
            Pi_lo_eval = SoftmaxHierarchicalActor.BatchBW.Pi_lo(a, Pi_lo_parameterization, state).clamp(0.0001,1)
            Pi_b_eval = SoftmaxHierarchicalActor.BatchBW.Pi_b(b, Pi_b_parameterization, state).clamp(0.0001,1)
            output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
        
            return output
    
        def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                             Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     alpha is the forward message: alpha.shape()= [option_space, termination_space]
            # =============================================================================
            alpha = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    if i2 == 1:
                        bt=True
                    else:
                        bt=False
                
                    Pi_comb = np.zeros(option_space)
                    for ot_past in range(option_space):
                        Pi_comb[ot_past] = SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                                                        Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                                                        state, option_space)
                    alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
            alpha = np.divide(alpha,np.sum(alpha))
                
            return alpha
    
        def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                                  Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     alpha is the forward message: alpha.shape()=[option_space, termination_space]
            #   mu is the initial distribution over options: mu.shape()=[1,option_space]
            # =============================================================================
            alpha = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    if i2 == 1:
                        bt=True
                    else:
                        bt=False
                
                    Pi_comb = np.zeros(option_space)
                    for ot_past in range(option_space):
                        Pi_comb[ot_past] = SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                                                        Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                                                        state, option_space)
                        alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
            alpha = np.divide(alpha, np.sum(alpha))
                
            return alpha

        def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     beta is the backward message: beta.shape()= [option_space, termination_space]
            # =============================================================================
            beta = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    for i1_next in range(option_space):
                        ot_next = i1_next
                        for i2_next in range(termination_space):
                            if i2_next == 1:
                                b_next=True
                            else:
                                b_next=False
                            beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot_next, ot, a, b_next, 
                                                                                                                                Pi_hi_parameterization, Pi_lo_parameterization[ot_next], 
                                                                                                                                Pi_b_parameterization[ot], state, option_space)
            beta = np.divide(beta,np.sum(beta))
        
            return beta

        def Alpha(self):
            alpha = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(len(self.TrainingSet)):
                print('alpha iter', t+1, '/', len(self.TrainingSet))
                if t ==0:
                    state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                    action = self.Labels[t]
                    alpha[:,:,t] = SoftmaxHierarchicalActor.BatchBW.ForwardFirstRecursion(self.mu, action, self.pi_hi, 
                                                                                          self.pi_lo, self.pi_b, 
                                                                                          state, self.option_dim, self.termination_dim)
                else:
                    state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                    action = self.Labels[t]
                    alpha[:,:,t] = SoftmaxHierarchicalActor.BatchBW.ForwardRecursion(alpha[:,:,t-1], action, self.pi_hi, 
                                                                                     self.pi_lo, self.pi_b, 
                                                                                     state, self.option_dim, self.termination_dim)
               
            return alpha

        def Beta(self):
            beta = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)+1))
            beta[:,:,len(self.TrainingSet)] = np.divide(np.ones((self.option_dim, self.termination_dim)),2*self.option_dim)
        
            for t_raw in range(len(self.TrainingSet)):
                t = len(self.TrainingSet) - (t_raw+1)
                print('beta iter', t_raw+1, '/', len(self.TrainingSet))
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                beta[:,:,t] = SoftmaxHierarchicalActor.BatchBW.BackwardRecursion(beta[:,:,t+1], action, self.pi_hi, 
                                                                                 self.pi_lo, self.pi_b, state,
                                                                                 self.option_dim, self.termination_dim)
            
            return beta

        def Smoothing(option_space, termination_space, alpha, beta):
            gamma = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot=i1
                for i2 in range(termination_space):
                    gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
                    
            gamma = np.divide(gamma,np.sum(gamma))
        
            return gamma

        def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                        Pi_b_parameterization, state, option_space, termination_space):
            gamma_tilde = np.empty((option_space, termination_space))
            for i1_past in range(option_space):
                ot_past = i1_past
                for i2 in range(termination_space):
                    if i2 == 1:
                        b=True
                    else:
                        b=False
                    for i1 in range(option_space):
                        ot = i1
                        gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, b, 
                                                                                                                                     Pi_hi_parameterization, Pi_lo_parameterization[ot], 
                                                                                                                                     Pi_b_parameterization[ot_past], state, option_space)
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
            gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
        
            return gamma_tilde

        def Gamma(self, alpha, beta):
            gamma = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(len(self.TrainingSet)):
                print('gamma iter', t+1, '/', len(self.TrainingSet))
                gamma[:,:,t]=SoftmaxHierarchicalActor.BatchBW.Smoothing(self.option_dim, self.termination_dim, alpha[:,:,t], beta[:,:,t])
            
            return gamma
    
        def GammaTilde(self, alpha, beta):
            gamma_tilde = np.zeros((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(1,len(self.TrainingSet)):
                print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                gamma_tilde[:,:,t]=SoftmaxHierarchicalActor.BatchBW.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                                                    self.pi_hi, self.pi_lo, self.pi_b, 
                                                                                    state, self.option_dim, self.termination_dim)
            return gamma_tilde
    
    # functions M-step
    
        def GammaTildeReshape(gamma_tilde, option_space):
    # =============================================================================
    #         Function to reshape Gamma_tilde with the same size of NN_pi_b output
    # =============================================================================
            T = gamma_tilde.shape[2]
            gamma_tilde_reshaped_array = np.empty((T-1,2,option_space))
            for i in range(option_space):
                gamma_tilde_reshaped = gamma_tilde[i,:,1:]
                gamma_tilde_reshaped_array[:,:,i] = gamma_tilde_reshaped.reshape(T-1,2)
                
            return gamma_tilde_reshaped_array
    
        def GammaReshapeActions(T, option_space, action_space, gamma, labels):
    # =============================================================================
    #         function to reshape gamma with the same size of the NN_pi_lo output
    # =============================================================================
            gamma_actions_array = np.empty((T, action_space, option_space))
            for k in range(option_space):
                gamma_reshaped_option = gamma[k,:,:]    
                gamma_reshaped_option = np.sum(gamma_reshaped_option,0)
                gamma_actions = np.empty((int(T),action_space))
                for i in range(T):
                    for j in range(action_space):
                        if int(labels[i])==j:
                            gamma_actions[i,j]=gamma_reshaped_option[i]
                        else:
                            gamma_actions[i,j] = 0
                gamma_actions_array[:,:,k] = gamma_actions
                
            return gamma_actions_array
        
        def GammaReshapeOptions(gamma):
    # =============================================================================
    #         function to reshape gamma with the same size of NN_pi_hi output
    # =============================================================================
            gamma_reshaped_options = gamma[:,1,:]
            gamma_reshaped_options = np.transpose(gamma_reshaped_options)
            
            return gamma_reshaped_options
    
    
        def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector,
                        NN_termination, NN_options, NN_actions, T, TrainingSet):
    # =============================================================================
    #         Compute batch loss function to minimize
    # =============================================================================
            loss = 0
            option_space = len(NN_actions)
            for i in range(option_space):
                pi_b = NN_termination[i](TrainingSet).cpu()
                loss = loss -torch.sum(torch.FloatTensor(gamma_tilde_reshaped[:,:,i])*torch.log(pi_b[:].clamp(1e-10,1.0)))/(T)
                pi_lo = NN_actions[i](TrainingSet).cpu()
                loss = loss -torch.sum(torch.FloatTensor(gamma_actions[:,:,i])*torch.log(pi_lo.clamp(1e-10,1.0)))/(T)
                
            pi_hi = NN_options(TrainingSet).cpu()
            loss_options = -torch.sum(torch.FloatTensor(gamma_reshaped_options)*torch.log(pi_hi.clamp(1e-10,1.0)))/(T)
            loss = loss + loss_options
        
            return loss    
    
    
        def OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector):
    # =============================================================================
    #         optimize loss in mini-batches
    # =============================================================================
            loss = 0
            n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
                    
            for n in range(n_batches):
                print("\n Batch %d" % (n+1,))
                TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                loss = SoftmaxHierarchicalActor.BatchBW.Loss(gamma_tilde_reshaped[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                                             gamma_reshaped_options[n*self.batch_size:(n+1)*self.batch_size,:], 
                                                             gamma_actions[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                                             auxiliary_vector[n*self.batch_size:(n+1)*self.batch_size,:],
                                                             self.pi_b, self.pi_hi, self.pi_lo, self.batch_size, TrainingSet)
        
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].zero_grad()
                    self.pi_b_optimizer[option].zero_grad()   
                self.pi_hi_optimizer.zero_grad()
                loss.backward()
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].step()
                    self.pi_b_optimizer[option].step()
                self.pi_hi_optimizer.step()
                print('loss:', float(loss))
        
            return loss   
        
        def likelihood_approximation(self):
            T = self.TrainingSet.shape[0]
            for t in range(T):
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]    
                partial = 0
                for o_past in range(self.option_dim):
                    for b in range(self.termination_dim):
                        for o in range(self.option_dim):
                            partial = partial + SoftmaxHierarchicalActor.BatchBW.Pi_hi(o_past, self.pi_hi, state)*SoftmaxHierarchicalActor.BatchBW.Pi_combined(o, o_past, action, b, self.pi_hi, self.pi_lo[o], 
                                                                                                                                                               self.pi_b[o_past], state, self.option_dim)
                if t == 0:
                    likelihood = partial
                else:
                    likelihood = likelihood+partial
    
    
            likelihood = (likelihood/T).data.numpy()
            
            return likelihood
                
        def Baum_Welch(self, N, likelihood_online=1):
    # =============================================================================
    #         batch BW for HIL
    # =============================================================================
            
            T = self.TrainingSet.shape[0]
            likelihood = []
            time_init = time.time()
            Time_list = [0]
                
            for n in range(N):
                print('iter Loss', n+1, '/', N)
            
                alpha = SoftmaxHierarchicalActor.BatchBW.Alpha(self)
                beta = SoftmaxHierarchicalActor.BatchBW.Beta(self)
                gamma = SoftmaxHierarchicalActor.BatchBW.Gamma(self, alpha, beta)
                gamma_tilde = SoftmaxHierarchicalActor.BatchBW.GammaTilde(self, alpha, beta)
            
                print('Expectation done')
                print('Starting maximization step')
                
                gamma_tilde_reshaped = SoftmaxHierarchicalActor.BatchBW.GammaTildeReshape(gamma_tilde, self.option_dim)
                gamma_actions = SoftmaxHierarchicalActor.BatchBW.GammaReshapeActions(T, self.option_dim, self.action_space_discrete, gamma, self.Labels)
                gamma_reshaped_options = SoftmaxHierarchicalActor.BatchBW.GammaReshapeOptions(gamma)
                m,n,o = gamma_actions.shape
                auxiliary_vector = np.zeros((m,n))
                for l in range(m):
                    for k in range(n):
                        if gamma_actions[l,k,0]!=0:
                            auxiliary_vector[l,k] = 1
        
    
                loss = SoftmaxHierarchicalActor.BatchBW.OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector)
                Time_list.append(time.time() - time_init)      
             
            likelihood = np.append(likelihood, SoftmaxHierarchicalActor.BatchBW.likelihood_approximation(self))  
            print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))
    
            
            return self.pi_hi, self.pi_lo, self.pi_b, likelihood, Time_list 
        
        def HierarchicalStochasticSampleTrajMDP(self, env, max_epoch_per_traj, number_of_trajectories):
           traj = [[None]*1 for _ in range(number_of_trajectories)]
           control = [[None]*1 for _ in range(number_of_trajectories)]
           Option = [[None]*1 for _ in range(number_of_trajectories)]
           Termination = [[None]*1 for _ in range(number_of_trajectories)]
           Reward_array = np.empty((0,0),int)
       
           for t in range(number_of_trajectories):
               done = False
               obs = np.round(env.reset(),3)
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
                   o_prob_tilde[0,:] = 0
                   o_prob_tilde[0,o] = 1 
           
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
                   obs, reward, done, _ = env.step(action)
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
                       o_prob_tilde[0,:] = 0
                       o_prob_tilde[0,o] = 1 
           
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

                    
class GaussianHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = F.relu(self.l1(state))
            return self.lS(self.l2(a))
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(SoftmaxHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,termination_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = F.relu(self.l1(state))
            return self.lS(self.l2(b))            
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(SoftmaxHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,option_dim)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = F.relu(self.l1(state))
            return self.lS(self.l2(o))

    class BatchBW(object):
        def __init__(self, state_dim, action_dim, option_dim, termination_dim, state_samples, action_samples, batch_size, l_rate):
            self.state_dim = state_dim
            self.action_dim = action_dim 
            self.option_dim = option_dim
            self.termination_dim = termination_dim
            self.TrainingSet = state_samples
            self.batch_size = batch_size
            self.mu = np.ones(option_dim)*np.divide(1,option_dim)
            self.action_space_discrete = len(np.unique(action_samples,axis=0))
            self.action_dictionary = np.unique(action_samples, axis = 0)
            labels = np.zeros((len(action_samples)))
            for i in range(len(action_samples)):
                for j in range(self.action_space_discrete):
                    if np.sum(action_samples[i,:] == self.action_dictionary[j,:]) == self.action_dim:
                        labels[i] = j     
            self.Labels = labels  
            # define hierarchical policy
            self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            self.pi_b = [[None]*1 for _ in range(option_dim)] 
            self.pi_lo = [[None]*1 for _ in range(option_dim)] 
            pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, self.action_space_discrete).to(device)
            pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            for option in range(self.option_dim):
                self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
                self.pi_b[option] = copy.deepcopy(pi_b_temp)
            # define optimizer 
            self.learning_rate = l_rate
            self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=self.learning_rate)
            self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
            self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)] 
            for option in range(self.option_dim):
                self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=self.learning_rate)
                self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=self.learning_rate)            

        def Pi_hi(ot, Pi_hi_parameterization, state):
            Pi_hi = Pi_hi_parameterization(state).cpu()
            o_prob = Pi_hi[0,ot]
            return o_prob

        def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space):
            if b == True:
                o_prob_tilde = SoftmaxHierarchicalActor.BatchBW.Pi_hi(ot, Pi_hi_parameterization, state)
            elif ot == ot_past:
                o_prob_tilde =  torch.FloatTensor([1])
            else:
                o_prob_tilde =  torch.FloatTensor([0])
            
            return o_prob_tilde

        def Pi_lo(a, Pi_lo_parameterization, state):
            Pi_lo = Pi_lo_parameterization(state).cpu()
            a_prob = Pi_lo[0,int(a)]
        
            return a_prob

        def Pi_b(b, Pi_b_parameterization, state):
            Pi_b = Pi_b_parameterization(state).cpu()
            if b == True:
                b_prob = Pi_b[0,1]
            else:
                b_prob = Pi_b[0,0]
            return b_prob
    
        def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, option_space):
            Pi_hi_eval = SoftmaxHierarchicalActor.BatchBW.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, option_space).clamp(0.0001,1)
            Pi_lo_eval = SoftmaxHierarchicalActor.BatchBW.Pi_lo(a, Pi_lo_parameterization, state).clamp(0.0001,1)
            Pi_b_eval = SoftmaxHierarchicalActor.BatchBW.Pi_b(b, Pi_b_parameterization, state).clamp(0.0001,1)
            output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
        
            return output
    
        def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                             Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     alpha is the forward message: alpha.shape()= [option_space, termination_space]
            # =============================================================================
            alpha = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    if i2 == 1:
                        bt=True
                    else:
                        bt=False
                
                    Pi_comb = np.zeros(option_space)
                    for ot_past in range(option_space):
                        Pi_comb[ot_past] = SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                                                        Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                                                        state, option_space)
                    alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
            alpha = np.divide(alpha,np.sum(alpha))
                
            return alpha
    
        def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                                  Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     alpha is the forward message: alpha.shape()=[option_space, termination_space]
            #   mu is the initial distribution over options: mu.shape()=[1,option_space]
            # =============================================================================
            alpha = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    if i2 == 1:
                        bt=True
                    else:
                        bt=False
                
                    Pi_comb = np.zeros(option_space)
                    for ot_past in range(option_space):
                        Pi_comb[ot_past] = SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, bt, 
                                                                                        Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                                                        state, option_space)
                        alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
            alpha = np.divide(alpha, np.sum(alpha))
                
            return alpha

        def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, option_space, termination_space):
            # =============================================================================
            #     beta is the backward message: beta.shape()= [option_space, termination_space]
            # =============================================================================
            beta = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot = i1
                for i2 in range(termination_space):
                    for i1_next in range(option_space):
                        ot_next = i1_next
                        for i2_next in range(termination_space):
                            if i2_next == 1:
                                b_next=True
                            else:
                                b_next=False
                            beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot_next, ot, a, b_next, 
                                                                                                                                Pi_hi_parameterization, Pi_lo_parameterization[ot_next], 
                                                                                                                                Pi_b_parameterization[ot], state, option_space)
            beta = np.divide(beta,np.sum(beta))
        
            return beta

        def Alpha(self):
            alpha = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(len(self.TrainingSet)):
                print('alpha iter', t+1, '/', len(self.TrainingSet))
                if t ==0:
                    state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                    action = self.Labels[t]
                    alpha[:,:,t] = SoftmaxHierarchicalActor.BatchBW.ForwardFirstRecursion(self.mu, action, self.pi_hi, 
                                                                                          self.pi_lo, self.pi_b, 
                                                                                          state, self.option_dim, self.termination_dim)
                else:
                    state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                    action = self.Labels[t]
                    alpha[:,:,t] = SoftmaxHierarchicalActor.BatchBW.ForwardRecursion(alpha[:,:,t-1], action, self.pi_hi, 
                                                                                     self.pi_lo, self.pi_b, 
                                                                                     state, self.option_dim, self.termination_dim)
               
            return alpha

        def Beta(self):
            beta = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)+1))
            beta[:,:,len(self.TrainingSet)] = np.divide(np.ones((self.option_dim, self.termination_dim)),2*self.option_dim)
        
            for t_raw in range(len(self.TrainingSet)):
                t = len(self.TrainingSet) - (t_raw+1)
                print('beta iter', t_raw+1, '/', len(self.TrainingSet))
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                beta[:,:,t] = SoftmaxHierarchicalActor.BatchBW.BackwardRecursion(beta[:,:,t+1], action, self.pi_hi, 
                                                                                 self.pi_lo, self.pi_b, state,
                                                                                 self.option_dim, self.termination_dim)
            
            return beta

        def Smoothing(option_space, termination_space, alpha, beta):
            gamma = np.empty((option_space, termination_space))
            for i1 in range(option_space):
                ot=i1
                for i2 in range(termination_space):
                    gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
                    
            gamma = np.divide(gamma,np.sum(gamma))
        
            return gamma

        def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                        Pi_b_parameterization, state, option_space, termination_space):
            gamma_tilde = np.empty((option_space, termination_space))
            for i1_past in range(option_space):
                ot_past = i1_past
                for i2 in range(termination_space):
                    if i2 == 1:
                        b=True
                    else:
                        b=False
                    for i1 in range(option_space):
                        ot = i1
                        gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*SoftmaxHierarchicalActor.BatchBW.Pi_combined(ot, ot_past, a, b, 
                                                                                                                                     Pi_hi_parameterization, Pi_lo_parameterization[ot], 
                                                                                                                                     Pi_b_parameterization[ot_past], state, option_space)
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
            gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
        
            return gamma_tilde

        def Gamma(self, alpha, beta):
            gamma = np.empty((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(len(self.TrainingSet)):
                print('gamma iter', t+1, '/', len(self.TrainingSet))
                gamma[:,:,t]=SoftmaxHierarchicalActor.BatchBW.Smoothing(self.option_dim, self.termination_dim, alpha[:,:,t], beta[:,:,t])
            
            return gamma
    
        def GammaTilde(self, alpha, beta):
            gamma_tilde = np.zeros((self.option_dim, self.termination_dim, len(self.TrainingSet)))
            for t in range(1,len(self.TrainingSet)):
                print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]
                gamma_tilde[:,:,t]=SoftmaxHierarchicalActor.BatchBW.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                                                    self.pi_hi, self.pi_lo, self.pi_b, 
                                                                                    state, self.option_dim, self.termination_dim)
            return gamma_tilde
    
    # functions M-step
    
        def GammaTildeReshape(gamma_tilde, option_space):
    # =============================================================================
    #         Function to reshape Gamma_tilde with the same size of NN_pi_b output
    # =============================================================================
            T = gamma_tilde.shape[2]
            gamma_tilde_reshaped_array = np.empty((T-1,2,option_space))
            for i in range(option_space):
                gamma_tilde_reshaped = gamma_tilde[i,:,1:]
                gamma_tilde_reshaped_array[:,:,i] = gamma_tilde_reshaped.reshape(T-1,2)
                
            return gamma_tilde_reshaped_array
    
        def GammaReshapeActions(T, option_space, action_space, gamma, labels):
    # =============================================================================
    #         function to reshape gamma with the same size of the NN_pi_lo output
    # =============================================================================
            gamma_actions_array = np.empty((T, action_space, option_space))
            for k in range(option_space):
                gamma_reshaped_option = gamma[k,:,:]    
                gamma_reshaped_option = np.sum(gamma_reshaped_option,0)
                gamma_actions = np.empty((int(T),action_space))
                for i in range(T):
                    for j in range(action_space):
                        if int(labels[i])==j:
                            gamma_actions[i,j]=gamma_reshaped_option[i]
                        else:
                            gamma_actions[i,j] = 0
                gamma_actions_array[:,:,k] = gamma_actions
                
            return gamma_actions_array
        
        def GammaReshapeOptions(gamma):
    # =============================================================================
    #         function to reshape gamma with the same size of NN_pi_hi output
    # =============================================================================
            gamma_reshaped_options = gamma[:,1,:]
            gamma_reshaped_options = np.transpose(gamma_reshaped_options)
            
            return gamma_reshaped_options
    
    
        def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector,
                        NN_termination, NN_options, NN_actions, T, TrainingSet):
    # =============================================================================
    #         Compute batch loss function to minimize
    # =============================================================================
            loss = 0
            option_space = len(NN_actions)
            for i in range(option_space):
                pi_b = NN_termination[i](TrainingSet).cpu()
                loss = loss -torch.sum(torch.FloatTensor(gamma_tilde_reshaped[:,:,i])*torch.log(pi_b[:].clamp(1e-10,1.0)))/(T)
                pi_lo = NN_actions[i](TrainingSet).cpu()
                loss = loss -torch.sum(torch.FloatTensor(gamma_actions[:,:,i])*torch.log(pi_lo.clamp(1e-10,1.0)))/(T)
                
            pi_hi = NN_options(TrainingSet).cpu()
            loss_options = -torch.sum(torch.FloatTensor(gamma_reshaped_options)*torch.log(pi_hi.clamp(1e-10,1.0)))/(T)
            loss = loss + loss_options
        
            return loss    
    
    
        def OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector):
    # =============================================================================
    #         optimize loss in mini-batches
    # =============================================================================
            loss = 0
            n_batches = np.int(self.TrainingSet.shape[0]/self.batch_size)
                    
            for n in range(n_batches):
                print("\n Batch %d" % (n+1,))
                TrainingSet = torch.FloatTensor(self.TrainingSet[n*self.batch_size:(n+1)*self.batch_size,:]).to(device)
                loss = SoftmaxHierarchicalActor.BatchBW.Loss(gamma_tilde_reshaped[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                                             gamma_reshaped_options[n*self.batch_size:(n+1)*self.batch_size,:], 
                                                             gamma_actions[n*self.batch_size:(n+1)*self.batch_size,:,:], 
                                                             auxiliary_vector[n*self.batch_size:(n+1)*self.batch_size,:],
                                                             self.pi_b, self.pi_hi, self.pi_lo, self.batch_size, TrainingSet)
        
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].zero_grad()
                    self.pi_b_optimizer[option].zero_grad()   
                self.pi_hi_optimizer.zero_grad()
                loss.backward()
                for option in range(0,self.option_dim):
                    self.pi_lo_optimizer[option].step()
                    self.pi_b_optimizer[option].step()
                self.pi_hi_optimizer.step()
                print('loss:', float(loss))
        
            return loss   
        
        def likelihood_approximation(self):
            T = self.TrainingSet.shape[0]
            for t in range(T):
                state = torch.FloatTensor(self.TrainingSet[t,:].reshape(1,self.state_dim)).to(device)
                action = self.Labels[t]    
                partial = 0
                for o_past in range(self.option_dim):
                    for b in range(self.termination_dim):
                        for o in range(self.option_dim):
                            partial = partial + SoftmaxHierarchicalActor.BatchBW.Pi_hi(o_past, self.pi_hi, state)*SoftmaxHierarchicalActor.BatchBW.Pi_combined(o, o_past, action, b, self.pi_hi, self.pi_lo[o], 
                                                                                                                                                               self.pi_b[o_past], state, self.option_dim)
                if t == 0:
                    likelihood = partial
                else:
                    likelihood = likelihood+partial
    
    
            likelihood = (likelihood/T).data.numpy()
            
            return likelihood
                
        def Baum_Welch(self, N, likelihood_online=1):
    # =============================================================================
    #         batch BW for HIL
    # =============================================================================
            
            T = self.TrainingSet.shape[0]
            likelihood = []
            time_init = time.time()
            Time_list = [0]
                
            for n in range(N):
                print('iter Loss', n+1, '/', N)
            
                alpha = SoftmaxHierarchicalActor.BatchBW.Alpha(self)
                beta = SoftmaxHierarchicalActor.BatchBW.Beta(self)
                gamma = SoftmaxHierarchicalActor.BatchBW.Gamma(self, alpha, beta)
                gamma_tilde = SoftmaxHierarchicalActor.BatchBW.GammaTilde(self, alpha, beta)
            
                print('Expectation done')
                print('Starting maximization step')
                
                gamma_tilde_reshaped = SoftmaxHierarchicalActor.BatchBW.GammaTildeReshape(gamma_tilde, self.option_dim)
                gamma_actions = SoftmaxHierarchicalActor.BatchBW.GammaReshapeActions(T, self.option_dim, self.action_space_discrete, gamma, self.Labels)
                gamma_reshaped_options = SoftmaxHierarchicalActor.BatchBW.GammaReshapeOptions(gamma)
                m,n,o = gamma_actions.shape
                auxiliary_vector = np.zeros((m,n))
                for l in range(m):
                    for k in range(n):
                        if gamma_actions[l,k,0]!=0:
                            auxiliary_vector[l,k] = 1
        
    
                loss = SoftmaxHierarchicalActor.BatchBW.OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector)
                Time_list.append(time.time() - time_init)      
             
            likelihood = np.append(likelihood, SoftmaxHierarchicalActor.BatchBW.likelihood_approximation(self))  
            print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))
    
            
            return self.pi_hi, self.pi_lo, self.pi_b, likelihood, Time_list 
        
    


