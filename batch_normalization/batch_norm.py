import tensorflow as tf
decay = 0.95
TAU = 0.001

class batch_norm:
    def __init__(self,inputs,size,is_training,sess,parForTarget=None,bn_param=None):
        
        self.sess = sess        
        self.scale = tf.Variable(tf.random_uniform([size],0.9,1.1))
        self.beta = tf.Variable(tf.random_uniform([size],-0.03,0.03))
        self.pop_mean = tf.Variable(tf.random_uniform([size],-0.03,0.03),trainable=False)
        self.pop_var = tf.Variable(tf.random_uniform([size],0.9,1.1),trainable=False)        
        self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])        
        self.train_mean = tf.assign(self.pop_mean,self.pop_mean * decay + self.batch_mean * (1 - decay))  
        self.train_var = tf.assign(self.pop_var,self.pop_var * decay + self.batch_var * (1 - decay))
                
        def training(): 
            return tf.nn.batch_normalization(inputs,
                self.batch_mean, self.batch_var, self.beta, self.scale, 0.0000001 )
    
        def testing(): 
            return tf.nn.batch_normalization(inputs,
            self.pop_mean, self.pop_var, self.beta, self.scale, 0.0000001)
        
        if parForTarget!=None:
            self.parForTarget = parForTarget
            self.updateScale = self.scale.assign(self.scale*(1-TAU)+self.parForTarget.scale*TAU)
            self.updateBeta = self.beta.assign(self.beta*(1-TAU)+self.parForTarget.beta*TAU)
            self.updateTarget = tf.group(self.updateScale, self.updateBeta)
            
        self.bnorm = tf.cond(is_training,training,testing) 
        