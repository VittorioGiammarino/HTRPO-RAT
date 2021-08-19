import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Buffer import ReplayBuffer
from models import DeepDeterministicHierarchicalActor
from models import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class H_TD3(object):
    def __init__(self, state_dim, action_dim, option_dim, termination_dim, max_action, 
                 l_rate_pi_lo=3e-4, l_rate_pi_hi=3e-4 , l_rate_pi_b=3e-4, l_rate_critic = 3e-4, discount=0.99, 
                 tau=0.005, eta = 0.001, policy_noise=0.2, noise_clip=0.5, pi_lo_freq=2, pi_b_freq=5, pi_hi_freq=5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.option_dim = option_dim
        self.termination_dim = termination_dim
        
        self.Buffer = [[None]*1 for _ in range(option_dim)]
        self.pi_hi = DeepDeterministicHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
        self.Critic = Critic(state_dim,action_dim, option_dim).to(device)
        self.Critic_target = copy.deepcopy(self.Critic)
        self.pi_lo = [[None]*1 for _ in range(option_dim)]
        self.pi_b = [[None]*1 for _ in range(option_dim)]
        self.pi_hi_target = copy.deepcopy(self.pi_hi)
        self.pi_lo_target = [[None]*1 for _ in range(option_dim)]
        self.pi_b_target = [[None]*1 for _ in range(option_dim)]
        
        pi_lo_temp = DeepDeterministicHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
        pi_b_temp = DeepDeterministicHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
        for option in range(option_dim):
            self.Buffer[option] = ReplayBuffer(state_dim, action_dim)
            self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b[option] = copy.deepcopy(pi_b_temp)
            self.pi_lo_target[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b_target[option] = copy.deepcopy(pi_b_temp)            
            
        # define optimizer 
        self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=l_rate_pi_hi)
        self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=l_rate_critic)  
        for option in range(self.option_dim):
            self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=l_rate_pi_lo)
            self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=l_rate_pi_b)  
                    
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.pi_lo_freq = pi_lo_freq
        self.pi_b_freq = pi_b_freq
        self.pi_hi_freq = pi_hi_freq
        self.learning_rate_pi_lo = l_rate_pi_lo
        self.learning_rate_pi_b = l_rate_pi_b
        self.learning_rate_pi_hi = l_rate_pi_hi
        
        self.total_it = 0
            
    def select_action(self, state, option):
        state = torch.FloatTensor(state).to(device)
        self.pi_lo[option].eval()
        action = self.pi_lo[option](state)
        action = (action).cpu().data.numpy()
        return action
    
    def select_action_target(self, state, option):
        state = torch.FloatTensor(state).to(device)
        self.pi_lo_target[option].eval()
        action = self.pi_lo_target[option](state)
        action = (action).cpu().data.numpy()
        return action
    
    def select_option(self, state, b, previous_option):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)     
        if b == 1:
            b_bool = True
        else:
            b_bool = False

        o_prob_tilde = np.empty((1,self.option_dim))
        if b_bool == True:
            o_prob_tilde = self.pi_hi(state).cpu().data.numpy()
        else:
            o_prob_tilde[0,:] = 0
            o_prob_tilde[0,previous_option] = 1

        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        temp = np.where(draw_o<=prob_o_rescaled)[1]
        if temp.size == 0:
             option = np.argmax(prob_o)
        else:
             option = np.amin(np.where(draw_o<=prob_o_rescaled)[1])
             
        return option
    
    def select_termination(self, state, option):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)         
        self.pi_b[option].eval()
        # Termination
        prob_b = self.pi_b[option](state).cpu().data.numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        temp = np.where(draw_b<=prob_b_rescaled)[1]
        if temp.size == 0:
            b = np.argmax(prob_b)
        else:
            b = np.amin(np.where(draw_b<=prob_b_rescaled)[1])
            
        return int(b)  

    def train(self, option, batch_size=256):
        
        self.total_it += 1
        
        # Sample replay buffer 
        state, action, next_state, reward, cost, not_done = self.Buffer[option].sample(batch_size)
        option_vector = torch.ones_like(reward[:,0] , dtype=int)
        
        with torch.no_grad():
        	# Compute the target Q value
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action_target_option_i = []
            for option_i in range(self.option_dim):
                next_action_target_option_i.append(((torch.FloatTensor(H_TD3.select_action_target(self, next_state, option_i)) + noise).clamp(-self.max_action[0], self.max_action[0])).to(device))
            
            next_action_target = next_action_target_option_i[option]
            first_term_target_Q1, first_term_target_Q2 = self.Critic_target(next_state, next_action_target, F.one_hot(option*option_vector, num_classes=self.option_dim))
            pi_b_target = self.pi_b_target[option](next_state).cpu()
            target_Q1 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q1
            target_Q2 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q2
            
            for option_i in range(self.option_dim):
                next_action_target = next_action_target_option_i[option_i]
                second_term_target_Q1, second_term_target_Q2 = self.Critic_target(next_state, next_action_target, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                target_Q1 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q1
                target_Q2 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q2
                 
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward-cost + not_done * self.discount * target_Q
            
		# Get current Q estimates
        current_Q1, current_Q2 = self.Critic(state, action, F.one_hot(option*option_vector, num_classes=self.option_dim))
        		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates actor-critic
        if self.total_it % self.pi_lo_freq == 0:
            
            self.pi_lo[option].train()
            action_pi_lo = (self.pi_lo[option](state)).cpu()
            
            pi_lo_loss = -(self.Critic.Q1(state, action_pi_lo, F.one_hot(option*option_vector, num_classes=self.option_dim))).mean()
            			
            # Optimize pi_lo 
            self.pi_lo_optimizer[option].zero_grad()
            pi_lo_loss.backward()
            self.pi_lo_optimizer[option].step()

			# Update the frozen target models
            for param, target_param in zip(self.Critic.parameters(), self.Critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.pi_lo[option].parameters(), self.pi_lo_target[option].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    

		# Delayed policy updates pi_b
        if self.total_it % self.pi_b_freq == 0:
            
            # Compute pi_b loss
            pi_b = self.pi_b[option](next_state).cpu()
            
            next_action_option_i = []
            for option_i in range(self.option_dim):
                next_action_option_i.append(torch.FloatTensor(H_TD3.select_action(self, next_state, option_i)).to(device))
            
            next_action = next_action_option_i[option]
            pi_b_loss = pi_b[:,1].reshape(-1,1)*(self.Critic.Q1(next_state, next_action, F.one_hot(option*option_vector, num_classes=self.option_dim)) + self.eta)
            for option_i in range(self.option_dim):
                next_action = next_action_option_i[option_i]
                pi_b_loss -= pi_b[:,1].reshape(-1,1)*self.pi_hi(next_state)[:,option_i].reshape(-1,1)*self.Critic.Q1(next_state, next_action, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                
            # Optimize pi_lo 
            pi_b_loss = torch.mean(pi_b_loss)
            self.pi_b_optimizer[option].zero_grad()
            pi_b_loss.backward()
            self.pi_b_optimizer[option].step()

			# Update the frozen target models    
            for param, target_param in zip(self.pi_b[option].parameters(), self.pi_b_target[option].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        if self.total_it % self.pi_hi_freq == 0:
            # Compute pi_hi loss
            next_action = next_action_option_i[option]
            pi_hi_loss = self.Critic.Q1(next_state, next_action, F.one_hot(option*option_vector, num_classes=self.option_dim))
            for option_i in range(self.option_dim):
                next_action = next_action_option_i[option_i]
                pi_hi_loss -= self.pi_hi(next_state)[:,option_i].reshape(-1,1)*self.Critic.Q1(next_state, next_action, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                
            pi_hi_loss = -torch.mean(torch.log((self.pi_hi(next_state)[:,option].reshape(-1,1)).clamp(1e-10,1))*pi_hi_loss)
            
            self.pi_hi_optimizer.zero_grad()
            pi_hi_loss.backward()
            self.pi_hi_optimizer.step()
        
            for param, target_param in zip(self.pi_hi.parameters(), self.pi_hi_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            

    def save_actor(self, filename):
        torch.save(self.pi_hi.state_dict(), filename + "_pi_hi")
        torch.save(self.pi_hi_optimizer.state_dict(), filename + "_pi_hi_optimizer")
        
        for option in range(self.option_dim):
            torch.save(self.pi_lo[option].state_dict(), filename + f"_pi_lo_option_{option}")
            torch.save(self.pi_lo_optimizer[option].state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
            torch.save(self.pi_b[option].state_dict(), filename + f"_pi_b_option_{option}")
            torch.save(self.pi_b_optimizer[option].state_dict(), filename + f"_pi_b_optimizer_option_{option}")  
    
    
    def load_actor(self, filename):
        self.pi_hi.load_state_dict(torch.load(filename + "_pi_hi"))
        self.pi_hi_optimizer.load_state_dict(torch.load(filename + "_pi_hi_optimizer"))
        
        for option in range(self.option_dim):
            self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
            self.pi_lo_optimizer[option].load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
            self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))
            self.pi_b_optimizer[option].load_state_dict(torch.load(filename + f"_pi_b_optimizer_option_{option}"))
            
    def save_critic(self, filename):
        torch.save(self.Critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.Critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.Critic_target = copy.deepcopy(self.Critic)