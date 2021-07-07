#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:58:25 2020

@author: vittorio
"""

import numpy as np
import World
import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as kb
import os
import time

def PreprocessData(bc_data_dir):

    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))

    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            shard_states, unprocessed_actions = zip(*data)
            shard_states = [x.flatten() for x in shard_states]
            
            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)/2
    
    return states, actions

class NN_PI_LO:
# =============================================================================
#     class for Neural network for pi_lo
# =============================================================================
    def __init__(self, action_space, size_input):
        self.action_space = action_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=0),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.action_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=1)),
                keras.layers.Softmax()
                                 ])              
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_lo.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model
    
    def PreTraining(self, TrainingSet, Labels, Nepoch):
        model = NN_PI_LO.NN_model(self)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=Nepoch)
        
        return model
        
            
class NN_PI_B:
# =============================================================================
#     class for Neural network for pi_b
# =============================================================================
    def __init__(self, termination_space, size_input):
        self.termination_space = termination_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(10, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=2),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.termination_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=3)),
                keras.layers.Softmax()
                                 ])               
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_b.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model        
    
    def PreTraining(self, TrainingSet, Labels):
        model = NN_PI_B.NN_model(self)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=200)
        
        return model    
            
class NN_PI_HI:
# =============================================================================
#     class for Neural Network for pi_hi
# =============================================================================
    def __init__(self, option_space, size_input):
        self.option_space = option_space
        self.size_input = size_input
                
    def NN_model(self):
        model = keras.Sequential([
                keras.layers.Dense(5, activation='relu', input_shape=(self.size_input,),
                                   kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=4),
                                   bias_initializer=keras.initializers.Zeros()),
                keras.layers.Dense(self.option_space, kernel_initializer=keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=5)),
                keras.layers.Softmax()
                                ])                
        return model
    
    def NN_model_plot(self,model):
        tf.keras.utils.plot_model(model, to_file='Figures/FiguresBatch/NN_pi_hi.png', 
                                  show_shapes=True, 
                                  show_layer_names=True,
                                  expand_nested=True)     
    def save(model, name):
        model.save(name)
        
    def load(name):
        NN_model = keras.models.load_model(name)
        return NN_model
    
    def PreTraining(self, TrainingSet):
        model = NN_PI_HI.NN_model(self)
        Labels = TrainingSet[:,3]
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model.fit(TrainingSet, Labels, epochs=200)
        
        return model
    

class BatchHIL:
    def __init__(self, TrainingSet, Labels, option_space, M_step_epoch, size_batch, optimizer): 
        self.TrainingSet = TrainingSet
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.action_space = len(np.unique(Labels,axis=0))
        self.action_dictionary = np.unique(Labels, axis = 0)
        self.size_action = Labels.shape[1]
        
        labels = np.zeros((len(Labels)))
        for i in range(len(Labels)):
            for j in range(self.action_space):
                if np.sum(Labels[i,:] == self.action_dictionary[j,:]) == self.size_action:
                    labels[i] = j
                    
        self.Labels = labels    
        
        self.termination_space = 2
        self.zeta = 0.0001
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        pi_hi = NN_PI_HI(self.option_space, self.size_input)
        NN_options = pi_hi.NN_model()
        self.NN_options = NN_options
        NN_low = []
        NN_termination = []
        pi_lo = NN_PI_LO(self.action_space, self.size_input)
        pi_b = NN_PI_B(self.termination_space, self.size_input)
        for options in range(self.option_space):
            NN_low.append(pi_lo.NN_model())
            NN_termination.append(pi_b.NN_model())
        self.NN_actions = NN_low
        self.NN_termination = NN_termination
        self.epochs = M_step_epoch
        self.size_batch = size_batch
        self.optimizer = optimizer
        self.Lambda_Lb = 1
        self.Lambda_Lv = 0.1
        self.Lambda_DKL = 0.01
        
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = BatchHIL.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
        else:
            o_prob_tilde = np.divide(zeta,option_space)
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state):
        Pi_lo = Pi_lo_parameterization(state)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state):
        Pi_b = Pi_b_parameterization(state)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = np.clip(BatchHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(BatchHIL.Pi_lo(a, Pi_lo_parameterization, state),0.0001,1)
        Pi_b_eval = np.clip(BatchHIL.Pi_b(b, Pi_b_parameterization, state),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
    
    def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                         Pi_b_parameterization, state, zeta, option_space, termination_space):
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
                    Pi_comb[ot_past] = BatchHIL.Pi_combined(ot, ot_past, a, bt, 
                                                       Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                       state, zeta, option_space)
                alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
        alpha = np.divide(alpha,np.sum(alpha))
            
        return alpha
    
    def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                              Pi_b_parameterization, state, zeta, option_space, termination_space):
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
                    Pi_comb[ot_past] = BatchHIL.Pi_combined(ot, ot_past, a, bt, 
                                                            Pi_hi_parameterization, Pi_lo_parameterization[ot], Pi_b_parameterization[ot_past], 
                                                            state, zeta, option_space)
                    alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
        alpha = np.divide(alpha, np.sum(alpha))
            
        return alpha

    def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, zeta, option_space, termination_space):
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
                        beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*BatchHIL.Pi_combined(ot_next, ot, a, b_next, 
                                                                                                    Pi_hi_parameterization, Pi_lo_parameterization[ot_next], 
                                                                                                    Pi_b_parameterization[ot], state, zeta, option_space)
        beta = np.divide(beta,np.sum(beta))
    
        return beta

    def Alpha(self):
        alpha = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('alpha iter', t+1, '/', len(self.TrainingSet))
            if t ==0:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.Labels[t]
                alpha[:,:,t] = BatchHIL.ForwardFirstRecursion(self.mu, action, self.NN_options, 
                                                              self.NN_actions, self.NN_termination, 
                                                              state, self.zeta, self.option_space, self.termination_space)
            else:
                state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
                action = self.Labels[t]
                alpha[:,:,t] = BatchHIL.ForwardRecursion(alpha[:,:,t-1], action, self.NN_options, 
                                                         self.NN_actions, self.NN_termination, 
                                                         state, self.zeta, self.option_space, self.termination_space)
           
        return alpha

    def Beta(self):
        beta = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)+1))
        beta[:,:,len(self.TrainingSet)] = np.divide(np.ones((self.option_space,self.termination_space)),2*self.option_space)
    
        for t_raw in range(len(self.TrainingSet)):
            t = len(self.TrainingSet) - (t_raw+1)
            print('beta iter', t_raw+1, '/', len(self.TrainingSet))
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.Labels[t]
            beta[:,:,t] = BatchHIL.BackwardRecursion(beta[:,:,t+1], action, self.NN_options, 
                                                       self.NN_actions, self.NN_termination, state, self.zeta, 
                                                       self.option_space, self.termination_space)
        
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
                    Pi_b_parameterization, state, zeta, option_space, termination_space):
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
                    gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*BatchHIL.Pi_combined(ot, ot_past, a, b, 
                                                                                                         Pi_hi_parameterization, Pi_lo_parameterization[ot], 
                                                                                                         Pi_b_parameterization[ot_past], state, zeta, option_space)
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
        gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
        return gamma_tilde

    def Gamma(self, alpha, beta):
        gamma = np.empty((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(len(self.TrainingSet)):
            print('gamma iter', t+1, '/', len(self.TrainingSet))
            gamma[:,:,t]=BatchHIL.Smoothing(self.option_space, self.termination_space, alpha[:,:,t], beta[:,:,t])
        
        return gamma

    def GammaTilde(self, alpha, beta):
        gamma_tilde = np.zeros((self.option_space,self.termination_space,len(self.TrainingSet)))
        for t in range(1,len(self.TrainingSet)):
            print('gamma tilde iter', t, '/', len(self.TrainingSet)-1)
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.Labels[t]
            gamma_tilde[:,:,t]=BatchHIL.DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                                        self.NN_options, self.NN_actions, self.NN_termination, 
                                                        state, self.zeta, self.option_space, self.termination_space)
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
    
    def Regularizer_Lb(pi_hi):
        tau = 0.5
        epsilon = 0.000001
        Lb = kb.sum(kb.sqrt(kb.square(kb.sum(pi_hi,0)/pi_hi.shape[0] - tau)+epsilon))
        return Lb
    
    def Regularizer_Lv(pi_hi):
        Lv = kb.sum(kb.sum(pi_hi-kb.sum(pi_hi,0)/pi_hi.shape[0],0)/pi_hi.shape[0])
        return Lv
    
    def Regularizer_KL_divergence(NN_actions, TrainingSet, gamma_actions, auxiliary_vector):
        option_space = len(NN_actions)
        DKL = 0
        epsilon = 0.000001
        T = len(TrainingSet)                
        for i in range(option_space):
            for j in range(option_space):
                if i!=j:
                    pi_lo_o_i = kb.sum(auxiliary_vector*NN_actions[i](TrainingSet,training=True),1)
                    pi_lo_o_j = kb.sum(auxiliary_vector*NN_actions[j](TrainingSet,training=True),1)
                    DKL = DKL + kb.sum(pi_lo_o_i*kb.log(kb.clip(pi_lo_o_i/(pi_lo_o_j+epsilon),1e-10,1.0)),0)/T
                    
        return DKL
    
    def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector,
                    NN_termination, NN_options, NN_actions, T, TrainingSet, Lambda_Lb, Lambda_Lv, Lambda_DKL):
# =============================================================================
#         Compute batch loss function to minimize
# =============================================================================
        loss = 0
        option_space = len(NN_actions)
        for i in range(option_space):
            pi_b = NN_termination[i](TrainingSet[:],training=True)
            loss = loss -kb.sum(gamma_tilde_reshaped[:,:,i]*kb.log(kb.clip(pi_b[:],1e-10,1.0)))/(T)
            pi_lo = NN_actions[i](TrainingSet,training=True)
            loss = loss -(kb.sum(gamma_actions[:,:,i]*kb.log(kb.clip(pi_lo,1e-10,1.0))))/(T)
            
        pi_hi = NN_options(TrainingSet,training=True)
        loss_options = -kb.sum(gamma_reshaped_options*kb.log(kb.clip(pi_hi,1e-10,1.0)))/(T)
        loss = loss + loss_options
        # Lb = BatchHIL.Regularizer_Lb(pi_hi)
        # Lv = BatchHIL.Regularizer_Lv(pi_hi)
        # DKL = BatchHIL.Regularizer_KL_divergence(NN_actions, TrainingSet, gamma_actions, auxiliary_vector)
        # loss = loss + Lambda_Lb*Lb - Lambda_Lv*Lv - Lambda_DKL*DKL
    
        return loss    

    
    def OptimizeLoss(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions):
# =============================================================================
#         minimize Loss all toghether
# =============================================================================
        weights = []
        loss = 0
        
        T = self.TrainingSet.shape[0]
        
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
        
            with tf.GradientTape() as tape:
                for i in range(self.option_space):
                    weights.append(self.NN_termination[i].trainable_weights)
                    weights.append(self.NN_actions[i].trainable_weights)
                weights.append(self.NN_options.trainable_weights)
                tape.watch(weights)
                loss = BatchHIL.Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, 
                                     self.NN_termination, self.NN_options, self.NN_actions, T, self.TrainingSet, self.Lambda_Lb, self.Lambda_Lv, self.Lambda_DKL)
            
            grads = tape.gradient(loss,weights)
            j=0
            for i in range(0,2*self.option_space,2):
                self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                j = j+1
            self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
            print('options loss:', float(loss))
        
        return loss        
    

    def OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector):
# =============================================================================
#         optimize loss in mini-batches
# =============================================================================
        weights = []
        loss = 0
        
        n_batches = np.int(self.TrainingSet.shape[0]/self.size_batch)

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            
            for n in range(n_batches):
                print("\n Batch %d" % (n+1,))
        
                with tf.GradientTape() as tape:
                    for i in range(self.option_space):
                        weights.append(self.NN_termination[i].trainable_weights)
                        weights.append(self.NN_actions[i].trainable_weights)
                    weights.append(self.NN_options.trainable_weights)
                    tape.watch(weights)
                    loss = BatchHIL.Loss(gamma_tilde_reshaped[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                         gamma_reshaped_options[n*self.size_batch:(n+1)*self.size_batch,:], 
                                         gamma_actions[n*self.size_batch:(n+1)*self.size_batch,:,:], 
                                         auxiliary_vector[n*self.size_batch:(n+1)*self.size_batch,:],
                                         self.NN_termination, self.NN_options, self.NN_actions, self.size_batch, 
                                         self.TrainingSet[n*self.size_batch:(n+1)*self.size_batch,:], self.Lambda_Lb, self.Lambda_Lv, self.Lambda_DKL)
            
                grads = tape.gradient(loss,weights)
                j=0
                for i in range(0,2*self.option_space,2):
                    self.optimizer.apply_gradients(zip(grads[i][:], self.NN_termination[j].trainable_weights))
                    self.optimizer.apply_gradients(zip(grads[i+1][:], self.NN_actions[j].trainable_weights))
                    j = j+1
                self.optimizer.apply_gradients(zip(grads[-1][:], self.NN_options.trainable_weights))
                print('loss:', float(loss))
        
        return loss   
    
    def likelihood_approximation(self):
        T = self.TrainingSet.shape[0]
        for t in range(T):
            state = self.TrainingSet[t,:].reshape(1,len(self.TrainingSet[t,:]))
            action = self.Labels[t]    
            partial = 0
            for o_past in range(self.option_space):
                for b in range(self.termination_space):
                    for o in range(self.option_space):
                        partial = partial + BatchHIL.Pi_hi(o_past, self.NN_options, state)*BatchHIL.Pi_combined(o, o_past, action, b, self.NN_options, self.NN_actions[o], 
                                                                                                                self.NN_termination[o_past], state, self.zeta, self.option_space)
            if t == 0:
                likelihood = partial
            else:
                likelihood = likelihood+partial


        likelihood = (likelihood/T).numpy()
        
        return likelihood
            
    def Baum_Welch(self, N, likelihood_online):
# =============================================================================
#         batch BW for HIL
# =============================================================================
        
        T = self.TrainingSet.shape[0]
        likelihood = []
        time_init = time.time()
        Time_list = [0]
            
        for n in range(N):
            print('iter Loss', n+1, '/', N)
        
            alpha = BatchHIL.Alpha(self)
            beta = BatchHIL.Beta(self)
            gamma = BatchHIL.Gamma(self, alpha, beta)
            gamma_tilde = BatchHIL.GammaTilde(self, alpha, beta)
        
            print('Expectation done')
            print('Starting maximization step')
            
            gamma_tilde_reshaped = BatchHIL.GammaTildeReshape(gamma_tilde, self.option_space)
            gamma_actions = BatchHIL.GammaReshapeActions(T, self.option_space, self.action_space, gamma, self.Labels)
            gamma_reshaped_options = BatchHIL.GammaReshapeOptions(gamma)
            m,n,o = gamma_actions.shape
            auxiliary_vector = np.zeros((m,n))
            for l in range(m):
                for k in range(n):
                    if gamma_actions[l,k,0]!=0:
                        auxiliary_vector[l,k] = 1
    

            loss = BatchHIL.OptimizeLossBatch(self, gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions, auxiliary_vector)
            Time_list.append(time.time() - time_init)
            
        likelihood = np.append(likelihood, BatchHIL.likelihood_approximation(self))         
        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
        return self.NN_options, self.NN_actions, self.NN_termination, likelihood, Time_list 

            
        
