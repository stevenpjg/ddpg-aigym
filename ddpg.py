import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from collections import deque
from gym.spaces import Box, Discrete
import random
from tensorflow_grad_inverter import grad_inverter

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA=0.99
is_grad_inverter = True
class DDPG:
    
    """ Deep Deterministic Policy Gradient Algorithm"""
    def __init__(self,env, is_batch_norm):
        self.env = env 
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        
        
        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.num_states, self.num_actions) 
            self.actor_net = ActorNet_bn(self.num_states, self.num_actions)
            
        else:
            self.critic_net = CriticNet(self.num_states, self.num_actions) 
            self.actor_net = ActorNet(self.num_states, self.num_actions)
        
        #Initialize Buffer Network:
        self.replay_memory = deque()
        
        #Intialize time step:
        self.time_step = 0
        self.counter = 0
        
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()        
        action_bounds = [action_max,action_min] 
        self.grad_inv = grad_inverter(action_bounds)
        
        
    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)
    
    def add_experience(self, observation_1, observation_2, action, reward, done):
        self.observation_1 = observation_1
        self.observation_2 = observation_2
        self.action = action
        self.reward = reward
        self.done = done
        self.replay_memory.append((self.observation_1, self.observation_2, self.action, self.reward,self.done))
        self.time_step = self.time_step + 1
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
            
        
    def minibatches(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        #state t
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        #state t+1        
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array( self.state_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),self.num_actions])
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)  
                  
                 
    def train(self):
        #sample a random minibatch of N transitions from R
        self.minibatches()
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.state_t_1_batch)
        #Q'(s_i+1,a_i+1)        
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_batch,self.action_t_1_batch) 
        self.y_i_batch=[]         
        for i in range(0,BATCH_SIZE):
                           
            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            else:
                
                self.y_i_batch.append(self.reward_batch[i] + GAMMA*q_t_1[i][0])                 
        
        self.y_i_batch=np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch,[len(self.y_i_batch),1])
        
        # Update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_batch, self.action_batch,self.y_i_batch)
        
        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self.state_t_batch) 
        
        if is_grad_inverter:        
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)#/BATCH_SIZE            
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a,action_for_delQ) 
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)[0]#/BATCH_SIZE
        
        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(self.state_t_batch,self.del_Q_a)
 
        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        
                
        
        
        
                
        
        
        
                     
                 
        



