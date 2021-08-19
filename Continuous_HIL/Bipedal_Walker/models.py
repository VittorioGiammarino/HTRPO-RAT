import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhGaussianHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(TanhGaussianHierarchicalActor.NN_PI_LO, self).__init__()
            
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
        		
        def forward(self, state):        
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std) 
            return mean.cpu(), std.cpu()
        
        def sample(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.max_action[0]
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.max_action[0] * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.max_action[0]
            return action.cpu(), log_prob.cpu(), mean.cpu()    
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(TanhGaussianHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(self.l3(b))              
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(TanhGaussianHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(self.l3(o))
        
class SoftmaxHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,128)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            return self.lS(self.l3(a))
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(SoftmaxHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(self.l3(b))              
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(SoftmaxHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(self.l3(o))
        
class DeepDeterministicHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(DeepDeterministicHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.action_dim = action_dim
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim),
            )
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.max_action = max_action
            
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            
        def forward(self, states):
            mean = self.net(states)     
            return self.max_action[0] * torch.tanh(mean)
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(DeepDeterministicHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(self.l3(b))              
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(DeepDeterministicHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(self.l3(o))
        

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, option_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sao))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1