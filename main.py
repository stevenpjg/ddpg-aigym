#Implementation of Deep Deterministic Gradient with Theano and Tensor Flow"
# Author: stevenget

import gym
from gym.spaces import Box, Discrete
import time
import numpy as np

from ddpg import DDPG

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
    counter=0
    total_reward=0
    for i in xrange(episodes):
        observation_1 = env.reset()
        
        reward_per_episode = 0
        for t in xrange(steps):
            #rendering environmet (optional)            
            #env.render()
            
            
            #select action using actor network model
            action = agent.evaluate_actor(np.reshape(observation_1,[1,4]))
            #action plus linear decay noise for expoloration
            action= action[0] + np.random.randn(env.action_space.shape[0]) / (i + 1)
            
            print 'Agent.Action :',action
            print '\n'
            print '\n'
            #time.sleep(3)            
            #raw_input("Press Enter to continue...")
                      
            observation_2,reward,done,[]=env.step(action)
            #add s_t,s_t+1,action,reward to experience memeroy
            agent.add_experience(observation_1,observation_2,action,reward,done)
            #train critic and actor network
            if counter > 1000: 
                agent.train()            
            
            reward_per_episode+=reward
            observation_1 = observation_2
            counter+=1
            #check if episode ends:
            if done:
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print '\n'
                print '\n'
                break
    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    

if __name__ == '__main__':
    main()    