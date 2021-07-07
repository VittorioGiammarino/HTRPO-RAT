#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:12:25 2021

@author: vittorio
"""
import numpy as np
import csv

class Rat_Foraging:      
    def TransitionCheck4Labels(state,state_next):
        x2 = state_next[0]-state[0]
        x1 = state_next[1]-state[1]
        step_length = np.array([abs(x2),abs(x1)])
        rad = np.arctan2(x1,x2)
        deg = rad*180/np.pi
        direction = Rat_Foraging.GetDirectionFromAngle(deg)
        
        return direction, step_length
    
    def StateTransition(init, step, step_length):
        next_state = np.zeros((1,2))
        next_state[0,0] = init[0] + step[0]*step_length[0]
        next_state[0,1] = init[1] + step[1]*step_length[1]
            
        return next_state
    
    def GetDirectionFromAngle(angle):
        if angle<0:
            angle = angle + 360
        sampling = 22.5
        slots = np.arange(sampling/2,360+sampling,sampling)
        label_direction = np.min(np.where(angle<=slots)[0])
        if label_direction==len(slots)-1:
            label_direction = 0            
         
        return label_direction
    
    def GetAngleFromDirection(direction):
        step_dictionary = [[1,0],[1,0.5],[1,1],[0.5,1],[0,1],[-0.5,1],[-1,1],[-1,0.5],[-1,0],[-1,-0.5],[-1,-1],[-0.5,-1],[0,-1],[0.5,-1],[1,-1],[1,-0.5]]
        step = step_dictionary[direction]        

        return step
        
    def TrueData():
        
        with open("Rat_Data/processed.csv") as f:
            data_raw = f.readlines()
        
        agent_data = csv.reader(data_raw)
        Training_set = []
        Real_set = []
        # True_set = np.empty((0,2))
        # time = np.empty((0,1))
        
        line_count = 0
        for row in agent_data:
            if line_count == 0:
                for i in range(len(row)):
                    if row[i]=='midpoint':
                        midpoint_index = i
            
            if line_count > 0:
                data = []
                processed_row = row[midpoint_index][1:-1].split()
                if len(processed_row)==0:
                    break
                data = [np.round(float(processed_row[0])), np.round(float(processed_row[1]))]
                data_real = [float(processed_row[0]), float(processed_row[1])]
                Training_set.append(data)
                Real_set.append(data_real)
                
            line_count+=1
            
        T_set = np.array(Training_set) 
        Real_set = np.array(Real_set)
        State_space, Training_set_index = np.unique(T_set, return_index=True, axis=0)    
        Training_set_cleaned = T_set[np.sort(Training_set_index),:]
        
        return Training_set_cleaned, Real_set
            
        
    def ProcessData():
        
        with open("Rat_Data/processed.csv") as f:
            data_raw = f.readlines()
        
        agent_data = csv.reader(data_raw)
        Training_set = []
        # True_set = np.empty((0,2))
        # time = np.empty((0,1))
        
        line_count = 0
        for row in agent_data:
            if line_count == 0:
                for i in range(len(row)):
                    if row[i]=='midpoint':
                        midpoint_index = i
            
            if line_count > 0:
                data = []
                processed_row = row[midpoint_index][1:-1].split()
                if len(processed_row)==0:
                    break
                data = [np.round(float(processed_row[0])), np.round(float(processed_row[1]))]
                # data = [float(processed_row[0]), float(processed_row[1])]
                Training_set.append(data)
                
            line_count+=1
            
        T_set = np.array(Training_set) 
        State_space, Training_set_index = np.unique(T_set, return_index=True, axis=0)    
        Training_set_cleaned = T_set[np.sort(Training_set_index),:]

        Labels = np.empty((0,1))
        States = np.empty((0,len(Training_set_cleaned[0,:])))
        States = np.append(States, Training_set_cleaned[0,:].reshape(1,len(Training_set_cleaned[0,:])), 0)
        for i in range(len(Training_set_cleaned)-1):
            action, step_length = Rat_Foraging.TransitionCheck4Labels(Training_set_cleaned[i,:], Training_set_cleaned[i+1,:])
            if action == 8:
                dummy = 0
            else:
                Labels = np.append(Labels, action)
            step = Rat_Foraging.GetAngleFromDirection(action)
            next_state = Rat_Foraging.StateTransition(States[i,:], step, step_length)
            States = np.append(States, next_state, 0)
                
        return States, Labels
    
    
    class env:
        def __init__(self,  Folder, expert_traj, init_state = np.array([0,0,0,8]), version = 'complete', version_coins = 'full_coins'):
            self.state = init_state
            self.version = version
            self.Folder = Folder
            self.expert_traj = expert_traj
            self.version_coins = version_coins
            self.observation_size = len(self.state)
            if version == 'complete':
                self.action_size = 8
            elif version == 'simplified':
                self.action_size = 4
                
            
        def reset(self, version = 'standard', init_state = np.array([0,0,0,8])):
            if version == 'standard':
                self.state = init_state
            else:
                state = 0.1*np.random.randint(-100,100,2)
                init_state = np.concatenate((state, np.array([0,8])))
                self.state = init_state
                
            return self.state
                
        def Transition(state,action):
            Transition = np.zeros((9,2))
            Transition[0,0] = state[0] + 0.1
            Transition[0,1] = state[1] + 0
            Transition[1,0] = state[0] + 0.1
            Transition[1,1] = state[1] + 0.1
            Transition[2,0] = state[0] + 0
            Transition[2,1] = state[1] + 0.1
            Transition[3,0] = state[0] - 0.1
            Transition[3,1] = state[1] + 0.1
            Transition[4,0] = state[0] - 0.1
            Transition[4,1] = state[1] + 0
            Transition[5,0] = state[0] - 0.1
            Transition[5,1] = state[1] - 0.1
            Transition[6,0] = state[0] + 0
            Transition[6,1] = state[1] - 0.1
            Transition[7,0] = state[0] + 0.1
            Transition[7,1] = state[1] - 0.1
            Transition[8,:] = state
            state_plus1 = Transition[int(action),:]
            
            return state_plus1     

    
        def step(self, action):
            
            r=0
            state_partial = self.state[0:2]
            # given action, draw next state
            state_plus1_partial = Rat_Foraging.env.Transition(state_partial, action)
                
            if state_plus1_partial[0]>10 or state_plus1_partial[0]<-10:
                state_plus1_partial[0] = state_partial[0] 

            if state_plus1_partial[1]>10 or state_plus1_partial[1]<-10:
                state_plus1_partial[1] = state_partial[1]                 
                    
            # Update psi and reward and closest coin direction
            dist_from_coins = np.linalg.norm(self.coin_location-state_plus1_partial,2,1)
            l=0
            psi = 0
                
            if np.min(dist_from_coins)<=0.8:
                index_min = np.argmin(dist_from_coins,0)
                closer_coin_position = self.coin_location[index_min,:]
                closer_coin_relative_position = np.array([closer_coin_position[0]-state_plus1_partial[0],closer_coin_position[1]-state_plus1_partial[1]])
                angle = np.arctan2(closer_coin_relative_position[1],closer_coin_relative_position[0])*180/np.pi
                coin_direction = Rat_Foraging.GetDirectionFromAngle(angle, self.version)  
            else:
                coin_direction = 8   
            
            for p in range(len(dist_from_coins)):
                if dist_from_coins[p]<=0.8:
                    psi = 1
                if dist_from_coins[p]<=0.3:
                    self.coin_location = np.delete(self.coin_location, l, 0)
                    r = r+1
                else:
                    l=l+1
                    
            state_plus1 = np.concatenate((state_plus1_partial, [psi], [coin_direction]))
            self.state = state_plus1
            
            return state_plus1, r
        













