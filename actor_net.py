import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001
class ActorNet:
    """ Actor Network Model of DDPG Algorithm """
    
    def __init__(self,num_states,num_actions):
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
           
            #actor network model parameters:
            self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,\
            self.actor_state_in, self.actor_model = self.create_actor_net(num_states, num_actions)
            
                                   
            #target actor network model parameters:
            self.t_W1_a, self.t_B1_a, self.t_W2_a, self.t_B2_a, self.t_W3_a, self.t_B3_a,\
            self.t_actor_state_in, self.t_actor_model = self.create_actor_net(num_states, num_actions)
            
            #cost of actor network:
            self.q_gradient_input = tf.placeholder("float",[None,num_actions]) #gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a]
            self.parameters_gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)#/BATCH_SIZE) 
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))  
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
        


    def create_actor_net(self, num_states=4, num_actions=1):
        """ Network that takes states and return action """
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        actor_state_in = tf.placeholder("float",[None,num_states])    
        W1_a=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1_a=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2_a=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        B2_a=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        W3_a=tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-0.003,0.003))
        B3_a=tf.Variable(tf.random_uniform([num_actions],-0.003,0.003))
    
        H1_a=tf.nn.softplus(tf.matmul(actor_state_in,W1_a)+B1_a)
        H2_a=tf.nn.tanh(tf.matmul(H1_a,W2_a)+B2_a)
        actor_model=tf.matmul(H2_a,W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, actor_state_in, actor_model
        
        
    def evaluate_actor(self,state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in:state_t})        
        
        
    def evaluate_target_actor(self,state_t_1):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_actor_state_in: state_t_1})
        
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run(self.optimizer, feed_dict={ self.actor_state_in: actor_state_in, self.q_gradient_input: q_gradient_input})
    
    def update_target_actor(self):
        self.sess.run([
				self.t_W1_a.assign(TAU*self.W1_a+(1-TAU)*self.t_W1_a),
				self.t_B1_a.assign(TAU*self.B1_a+(1-TAU)*self.t_B1_a),
				self.t_W2_a.assign(TAU*self.W2_a+(1-TAU)*self.t_W2_a),
				self.t_B2_a.assign(TAU*self.B2_a+(1-TAU)*self.t_B2_a),
				self.t_W3_a.assign(TAU*self.W3_a+(1-TAU)*self.t_W3_a),
				self.t_B3_a.assign(TAU*self.B3_a+(1-TAU)*self.t_B3_a)])    

        