#Implementation of Deep Deterministic Gradient with Theano and Tensor Flow"
# Author: Steven Spielberg Pon Kumar stevenpjg

import gym
from gym.spaces import Box, Discrete
import numpy as np

from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
episodes=10000
steps=1000 #steps per episode
 
def main():
    experiment= 'InvertedPendulum-v1'
    env= gym.make(experiment)
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    
    #saving reward:
    reward_st = np.array([0])
    
    
    
    for i in xrange(episodes):
        observation = env.reset()
    
        reward_per_episode = 0
        for t in xrange(steps):
            #rendering environmet (optional)            
            #env.render()
            
            x = observation
            #select action using actor network model
            action = agent.evaluate_actor(np.reshape(x,[num_actions,num_states]))
            
            noise = exploration_noise.noise()
            
                       
            action = action[0] + noise
            
            
            print 'Agent.Action :',action
            print '\n'
            print '\n'
            
                      
            observation,reward,done,[]=env.step(action)
            #add s_t,s_t+1,action,reward to experience memeroy
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()            
            
            reward_per_episode+=reward
            
            counter+=1
            #check if episode ends:
            if done:
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                exploration_noise.reset()
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n'
                print '\n'
                break
    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    

if __name__ == '__main__':
    main()    