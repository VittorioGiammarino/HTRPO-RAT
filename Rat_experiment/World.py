#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 06:43:28 2021

@author: vittorio
"""
import numpy as np

class Rat_foraging:
    class env:
        def __init__(self,  init_state = np.array([[500, 500, 0]])):
            self.state = init_state
            self.observation_size = len(self.state)
            
            self.R1_rectangle = [[400,800], [400,1000], [500,1000], [500,800]]
            self.R2_rectangle = [[250,0], [350,0], [350,200], [250,200]]
            
        def reset(self, init_state = np.array([[0, 1000, 0]])):
            self.state = init_state
             
            return self.state
                
        def Transition(state,action):
            state_plus1 = np.zeros((1,2))
            direction = np.random.normal(action[0], action[1])
            state_plus1[0,0] = state[0] + np.cos(direction)
            state_plus1[0,1] = state[1] + np.sin(direction)
                        
            return state_plus1     
        
        def UpdateReward(self, state):
            epsilon1 = 0.99
            u1 = np.random.rand()
            epsilon2 = 0.99
            u2 = np.random.rand()
            
            psi = state[0,2]
            x = state[0,0:2]
            
            Agent_in_R1 = x[0]>=self.R1_rectangle[0][0] and x[0]<=self.R1_rectangle[2][0] and x[1]>=self.R1_rectangle[0][1] and x[1]<=self.R1_rectangle[1][1]
            Agent_in_R2 = x[0]>=self.R2_rectangle[0][0] and x[0]<=self.R2_rectangle[2][0] and x[1]>=self.R1_rectangle[0][1] and x[1]<=self.R1_rectangle[2][1]

            # psi becomes 0 if R1 active, 1 if R2 active, 2 if both active, 0 if nothing active
            if psi == 0 and u2 > epsilon2:
                psi = 2
            elif psi == 1 and u1 > epsilon1:
                psi = 2
            elif psi == 3 and u1 > epsilon1:
                psi = 0
            elif psi == 3 and u2 > epsilon2:
                psi = 1
            elif psi == 3 and u1 > epsilon1 and u2 > epsilon2:
                psi = 2
    
            reward = 0
            if psi == 0 and Agent_in_R1:
                psi = 3
                reward = 1
            elif psi == 1 and Agent_in_R2:
                psi = 3 
                reward = 1
            elif psi == 2 and Agent_in_R1:
                psi = 1
                reward = 1
            elif psi == 2 and Agent_in_R2:
                psi = 0
                reward = 1
            
            return psi, reward

        def step(self, action):
            
            state_partial = self.state[0,0:2]
            # given action, draw next state
            state_plus1_partial = Rat_foraging.env.Transition(state_partial, action)
                
            if state_plus1_partial[0,0]>1000 or state_plus1_partial[0,0]<0:
                state_plus1_partial[0,0] = state_partial[0] 

            if state_plus1_partial[0,1]>1000 or state_plus1_partial[0,1]<0:
                state_plus1_partial[0,1] = state_partial[1]  
                    
            psi, reward = Rat_foraging.env.UpdateReward(self, self.state)
                    
            state_plus1 = np.concatenate((state_plus1_partial, np.array([[psi]])),1)
            self.state = state_plus1
            
            return state_plus1, reward
        
        def heuristic_policy(self):
            psi = self.state[0,2]
            x = self.state[0,0:2]
            action = np.zeros((2))
            
            if psi == 0:
                goal = self.R1_rectangle[0]
            elif psi == 1:
                goal = self.R2_rectangle[2]
            elif psi == 2:
                index_min = np.linalg.norm(x-self.R1_rectangle[0]) - np.linalg.norm(x-self.R2_rectangle[2])
                if index_min<=0:
                    goal = self.R1_rectangle[0]
                else:
                    goal = self.R2_rectangle[2]
            elif psi == 3:
                goal = x
                    
            x2 = goal[0]-x[0]
            x1 = goal[1]-x[1]
            rad = np.arctan2(x1,x2)
            action[0] = rad
            action[1] = 1
            
            return action
        
                
        
        
        
        
        
        
        
        