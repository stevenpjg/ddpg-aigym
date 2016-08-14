import tensorflow as tf
import math
from batch_normalization.batch_norm import batch_norm
import numpy as np
LEARNING_RATE= 0.001
TAU = 0.001
BATCH_SIZE = 64
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300
class CriticNet_bn:
    """ Critic Q value model with batch normalization of the DDPG algorithm """
    def __init__(self,num_states,num_actions):
        
        tf.reset_default_graph()
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #Critic Q Network:
            self.critic_state_in =  tf.placeholder("float",[None,num_states])
            self.critic_action_in = tf.placeholder("float",[None,num_actions]) 
            self.W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))  
            self.B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            self.W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            self.W3_c = tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
            self.B3_c = tf.Variable(tf.random_uniform([1],-0.003,0.003))
            
            self.is_training = tf.placeholder(tf.bool, [])
            self.H1_t = tf.matmul(self.critic_state_in,self.W1_c)
            self.H1_c_bn = batch_norm(self.H1_t,N_HIDDEN_1,self.is_training,self.sess)
            
            self.H1_c = tf.nn.softplus(self.H1_c_bn.bnorm) + self.B1_c

        
            self.H2_t = tf.matmul(self.H1_c,self.W2_c)+tf.matmul(self.critic_action_in,self.W2_action_c)
            self.H2_c_bn = batch_norm(self.H2_t,N_HIDDEN_2,self.is_training,self.sess)
            self.H2_c = tf.nn.tanh(self.H2_c_bn.bnorm) + self.B2_c
            
            self.critic_q_model = tf.matmul(self.H2_c,self.W3_c)+self.B3_c
            
           # Target Critic Q Network:
            self.t_critic_state_in =  tf.placeholder("float",[None,num_states])
            self.t_critic_action_in = tf.placeholder("float",[None,num_actions])
            self.t_W1_c = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.t_B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.t_W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))  
            self.t_W2_action_c = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            self.t_B2_c= tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
            self.t_W3_c = tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
            self.t_B3_c = tf.Variable(tf.random_uniform([1],-0.003,0.003))
            
            self.t_H1_t = tf.matmul(self.t_critic_state_in,self.t_W1_c)
            self.t_H1_c_bn = batch_norm(self.t_H1_t,N_HIDDEN_1,self.is_training,self.sess,self.H1_c_bn)        
            self.t_H1_c = tf.nn.softplus(self.t_H1_c_bn.bnorm) + self.t_B1_c

            self.t_H2_t = tf.matmul(self.t_H1_c,self.t_W2_c)+tf.matmul(self.t_critic_action_in,self.t_W2_action_c)
            self.t_H2_c_bn = batch_norm(self.t_H2_t,N_HIDDEN_2,self.is_training,self.sess,self.H2_c_bn)
            self.t_H2_c = tf.nn.tanh(self.t_H2_c_bn.bnorm) + self.t_B2_c
            
            self.t_critic_q_model = tf.matmul(self.t_H2_c,self.t_W3_c)+self.t_B3_c
            
            
            self.q_value_in=tf.placeholder("float",[None,1]) #supervisor
            #self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c) 
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.W2_c,2))             
            self.cost=tf.pow(self.critic_q_model-self.q_value_in,2)/BATCH_SIZE + self.l2_regularizer_loss#/tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])] #this is just divided by batch size
            #from simple actor net:
            self.check_fl = self.action_gradients             
            
            #initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())
            
            #To initialize critic and target with the same values:
            # copy target parameters
            self.sess.run([
				self.t_W1_c.assign(self.W1_c),
				self.t_B1_c.assign(self.B1_c),
				self.t_W2_c.assign(self.W2_c),
				self.t_W2_action_c.assign(self.W2_action_c),
				self.t_B2_c.assign(self.B2_c),
				self.t_W3_c.assign(self.W3_c),
				self.t_B3_c.assign(self.B3_c)
			])
            
    
    def train_critic(self, state_t_batch, action_batch, y_i_batch ):
        self.sess.run([self.optimizer,self.H1_c_bn.train_mean,self.H1_c_bn.train_var,self.H2_c_bn.train_mean,self.H2_c_bn.train_var,self.t_H1_c_bn.train_mean,self.t_H1_c_bn.train_var,self.t_H2_c_bn.train_mean,self.t_H2_c_bn.train_var], feed_dict={self.critic_state_in: state_t_batch,self.t_critic_state_in: state_t_batch, self.critic_action_in:action_batch,self.t_critic_action_in:action_batch, self.q_value_in: y_i_batch,self.is_training: True})
        
    def evaluate_target_critic(self,state_t_1,action_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1,self.is_training: False})    
        
        
    def compute_delQ_a(self,state_t,action_t):
        return self.sess.run(self.action_gradients, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t,self.is_training: False})

    def update_target_critic(self):
        self.sess.run([
				self.t_W1_c.assign(TAU*self.W1_c+(1-TAU)*self.t_W1_c),
                  self.t_B1_c.assign(TAU*self.B1_c+(1-TAU)*self.t_B1_c),  
				self.t_W2_c.assign(TAU*self.W2_c+(1-TAU)*self.t_W2_c),
				self.t_W2_action_c.assign(TAU*self.W2_action_c+(1-TAU)*self.t_W2_action_c),
				self.t_B2_c.assign(TAU*self.B2_c+(1-TAU)*self.t_B2_c),
                  self.t_W3_c.assign(TAU*self.W3_c+(1-TAU)*self.t_W3_c),
				self.t_B3_c.assign(TAU*self.B3_c+(1-TAU)*self.t_B3_c),
                  self.t_H1_c_bn.updateTarget,
                  self.t_H2_c_bn.updateTarget
			])
         


