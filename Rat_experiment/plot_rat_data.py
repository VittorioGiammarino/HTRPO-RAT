#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 05:42:27 2021

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt

from Process_Data import Rat_Foraging

Training_set_cleaned, Real_set = Rat_Foraging.TrueData()
Simulated_data, Labels = Rat_Foraging.ProcessData()

minute = 14.7
init_time = minute*60
final_time = 15*60 + 24

T_set = Training_set_cleaned
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cleaned data')
plt.savefig('Figures/cleaned_data_time_init{}.eps'.format(minute), format='eps')
plt.show() 

T_set = Real_set
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real data')
plt.savefig('Figures/Real_data_time_init{}.eps'.format(minute), format='eps')
plt.show() 

T_set = Simulated_data
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simulated data')
plt.savefig('Figures/Simulated_time_init{}.eps'.format(minute), format='eps')
plt.show() 

minute = 14
init_time = minute*60
final_time = 15*60 + 24

T_set = Training_set_cleaned
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cleaned data')
plt.savefig('Figures/cleaned_data_time_init{}.eps'.format(minute), format='eps')
plt.show() 

T_set = Real_set
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real data')
plt.savefig('Figures/Real_data_time_init{}.eps'.format(minute), format='eps')
plt.show() 

T_set = Simulated_data
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:,0], T_set[init:,1], c=time, marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simulated data')
plt.savefig('Figures/Simulated_time_init{}.eps'.format(minute), format='eps')
plt.show() 


minute = 10
init_time = minute*60
final_time = 15*60 + 24
last_step = 3000

T_set = Training_set_cleaned
init = int((len(T_set[:,0])*init_time)/final_time)
time = np.linspace(init_time,final_time, len(T_set[init:init+last_step,0])) 
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:init+last_step,0], T_set[init:init+last_step,1], c=time[:], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cleaned data')
plt.savefig('Figures/cleaned_data_last_step{}.eps'.format(init+last_step), format='eps')
plt.show() 

T_set = Real_set
init = int((len(T_set[:,0])*init_time)/final_time)
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:init+last_step,0], T_set[init:init+last_step,1], c=time[:], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real data')
plt.savefig('Figures/Real_data_last_step{}.eps'.format(last_step), format='eps')
plt.show() 

T_set = Simulated_data
init = int((len(T_set[:,0])*init_time)/final_time)
fig, ax = plt.subplots()
plot_data = plt.scatter(T_set[init:init+last_step,0], T_set[init:init+last_step,1], c=time[:], marker='o', cmap='cool') #-1], marker='o', cmap='cool')
cbar = fig.colorbar(plot_data, ticks=[10, 100, 200, 300, 600, 900])
cbar.ax.set_yticklabels(['time = 0', 'time = 100', 'time = 200', 'time = 300', 'time = 600', 'time = 900'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simulated data')
plt.savefig('Figures/Simulated_last_step{}.eps'.format(last_step), format='eps')
plt.show() 


