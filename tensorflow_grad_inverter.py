#Reference:
#https://github.com/MOCR/

import tensorflow as tf



class grad_inverter:
    def __init__(self, action_bounds):

        self.sess = tf.InteractiveSession()       
        
        self.action_size = len(action_bounds[0])
        
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.pmax = tf.constant(action_bounds[0], dtype = tf.float32)
        self.pmin = tf.constant(action_bounds[1], dtype = tf.float32)
        self.prange = tf.constant([x - y for x, y in zip(action_bounds[0],action_bounds[1])], dtype = tf.float32)
        self.pdiff_max = tf.div(-self.action_input+self.pmax, self.prange)
        self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])
        self.grad_inverter = tf.select(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.mul(self.act_grad, self.pdiff_max), tf.mul(self.act_grad, self.pdiff_min))        
    
    def invert(self, grad, action):

        
        return self.sess.run(self.grad_inverter, feed_dict = {self.action_input: action, self.act_grad: grad[0]})
