import tensorflow as tf
import math
from batch_normalization.batch_norm import batch_norm
import numpy as np
LEARNING_RATE = 0.0001
TAU = 0.001
BATCH_SIZE = 64
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300

class ActorNet_bn:
    """ Actor Network Model with Batch Normalization of DDPG Algorithm """
    
    def __init__(self,num_states,num_actions):
        tf.reset_default_graph()
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #actor network model parameters:
            self.actor_state_in = tf.placeholder("float",[None,num_states]) 
            self.W1_a = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.B1_a=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.W2_a = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
            self.B2_a=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
            self.W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-0.003,0.003))
            self.B3_a = tf.Variable(tf.random_uniform([num_actions],-0.003,0.003))
            
            self.is_training = tf.placeholder(tf.bool, [])
            self.H1_t= tf.matmul(self.actor_state_in,self.W1_a)
            self.H1_a_bn = batch_norm(self.H1_t,N_HIDDEN_1, self.is_training, self.sess)
            self.H1_a = tf.nn.softplus(self.H1_a_bn.bnorm) + self.B1_a
            
            self.H2_t=tf.matmul(self.H1_a,self.W2_a)
            self.H2_a_bn = batch_norm(self.H2_t,N_HIDDEN_2,self.is_training,self.sess)
            self.H2_a = tf.nn.tanh(self.H2_a_bn.bnorm) + self.B2_a
            self.actor_model=tf.matmul(self.H2_a,self.W3_a) + self.B3_a
            
                                   
            #target actor network model parameters:
            self.t_actor_state_in = tf.placeholder("float",[None,num_states]) 
            self.t_W1_a = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.t_B1_a=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
            self.t_W2_a = tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
            self.t_B2_a=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
            self.t_W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-0.003,0.003))
            self.t_B3_a = tf.Variable(tf.random_uniform([num_actions],-0.003,0.003))
            
            self.t_is_training = tf.placeholder(tf.bool, [])
            self.t_H1_t= tf.matmul(self.t_actor_state_in,self.t_W1_a)
            self.t_H1_a_bn = batch_norm(self.t_H1_t,N_HIDDEN_1, self.t_is_training, self.sess,self.H1_a_bn)
            self.t_H1_a = tf.nn.softplus(self.t_H1_a_bn.bnorm) + self.t_B1_a
            
            self.t_H2_t=tf.matmul(self.t_H1_a,self.t_W2_a)
            self.t_H2_a_bn = batch_norm(self.t_H2_t,N_HIDDEN_2,self.t_is_training,self.sess,self.H2_a_bn)
            self.t_H2_a = tf.nn.tanh(self.t_H2_a_bn.bnorm) + self.t_B2_a
            self.t_actor_model=tf.matmul(self.t_H2_a,self.t_W3_a) + self.t_B3_a
            
            #cost of actor network:
            self.q_gradient_input = tf.placeholder("float",[None,num_actions]) #gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.W1_a, self.B1_a, self.W2_a, self.B2_a,self.W3_a, self.B3_a, self.H1_a_bn.scale,self.H1_a_bn.beta,self.H2_a_bn.scale,self.H2_a_bn.beta]
            self.parameters_gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)#/BATCH_SIZE) changed -self.q_gradient to -
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,epsilon=1e-08).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))  
            #initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())    
            
            #To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
				self.t_W1_a.assign(self.W1_a),
				self.t_B1_a.assign(self.B1_a),
				self.t_W2_a.assign(self.W2_a),
				self.t_B2_a.assign(self.B2_a),
				self.t_W3_a.assign(self.W3_a),
				self.t_B3_a.assign(self.B3_a)])
            
        
    def evaluate_actor(self,state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in:state_t,self.is_training: False})        
        
        
    def evaluate_target_actor(self,state_t_1):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_actor_state_in: state_t_1,self.t_is_training: False})
        
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run([self.optimizer,self.H1_a_bn.train_mean,self.H1_a_bn.train_var,self.H2_a_bn.train_mean,self.H2_a_bn.train_var,self.t_H1_a_bn.train_mean,self.t_H1_a_bn.train_var,self.t_H2_a_bn.train_mean, self.t_H2_a_bn.train_var], feed_dict={ self.actor_state_in: actor_state_in,self.t_actor_state_in: actor_state_in, self.q_gradient_input: q_gradient_input,self.is_training: True,self.t_is_training: True})
        
    def update_target_actor(self):
        self.sess.run([
				self.t_W1_a.assign(TAU*self.W1_a+(1-TAU)*self.t_W1_a),
                  self.t_B1_a.assign(TAU*self.B1_a+(1-TAU)*self.t_B1_a),  
				self.t_W2_a.assign(TAU*self.W2_a+(1-TAU)*self.t_W2_a),
                  self.t_B2_a.assign(TAU*self.B2_a+(1-TAU)*self.t_B2_a),  
				self.t_W3_a.assign(TAU*self.W3_a+(1-TAU)*self.t_W3_a),
				self.t_B3_a.assign(TAU*self.B3_a+(1-TAU)*self.t_B3_a),
                  self.t_H1_a_bn.updateTarget,
                  self.t_H2_a_bn.updateTarget,
    ])    
        
        