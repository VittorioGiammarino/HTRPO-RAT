#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:32:32 2021

@author: vittorio
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        # architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        return torch.sigmoid(self.get_logits(state, action))

    def get_logits(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        sa = torch.cat([state, action], 1)
        d = F.relu(self.l1(sa))
        d = F.relu(self.l2(d))
        d = self.l3(d)
        return d

class Gail(object):
    def __init__(self, state_dim, action_dim, expert_states, expert_actions):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.expert_states = torch.FloatTensor(expert_states)
        self.expert_actions = torch.FloatTensor(expert_actions)
        self.discriminator = Discriminator(self.state_dim, self.action_dim)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())
        
    def update(self, learner_states, learner_actions):
        self.discriminator.train()
        learner_states = torch.FloatTensor(learner_states)
        learner_actions = torch.FloatTensor(learner_actions)
        expert_scores = self.discriminator.get_logits(self.expert_states, self.expert_actions)
        learner_scores = self.discriminator.get_logits(learner_states, learner_actions)
        
        self.discriminator_optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(expert_scores, torch.zeros_like(expert_scores)) + F.binary_cross_entropy_with_logits(learner_scores, torch.ones_like(learner_scores))
        loss.backward()
        self.discriminator_optimizer.step()
        
        
        
        