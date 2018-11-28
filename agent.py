#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:46:54 2017

@author: aidanrocke & ildefonsmagrans
"""
import tensorflow as tf
import numpy as np

class agent_cognition:
    
    """
        An agent that reasons using a measure of empowerment. 
        Here we assume that env refers to an initialised environment class. 
    """
    
    def __init__(self,planning_horizon,sess,seed, bound):
        self.sess = sess
        self.seed = seed
        self.horizon = planning_horizon        
        self.bound = bound
        
        self.current_state = tf.placeholder(tf.float32, [None, 2])
        self.source_action = tf.placeholder(tf.float32, [None, 2])
        
        self.source_input_n = tf.placeholder(tf.float32, [None, 4])
        self.decoder_input_n = tf.placeholder(tf.float32, [None, 6])
                        
        self.src_mu, self.src_log_sigma = self.source_dist_n()
        self.decoder_mu, self.decoder_log_sigma = self.decoder_dist_n()
    
    def init_weights(self,shape,var_name):
        """
            Xavier initialisation of neural networks
        """
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape),name = var_name)
        
    def two_layer_net(self, X, w_h, w_h2, w_o,bias_1, bias_2):
        """
            A generic method for creating two-layer networks
            
            input: weights
            output: neural network
        """
        
        h = tf.nn.elu(tf.add(tf.matmul(X, w_h),bias_1))
        h2 = tf.nn.elu(tf.add(tf.matmul(h, w_h2),bias_2))
        
        return tf.matmul(h2, w_o)
    
    def empowerment_critic(self):
        """
        This function provides a cheap approximation to empowerment
        upon convergence of the training algorithm. Given that the 
        mutual information is non-negative this function must only
        give non-negative output. 
        
        input: state
        output: empowerment estimate
        """
        
        with tf.variable_scope("critic"):
            
            tf.set_random_seed(self.seed)
    
            w_h = self.init_weights([2,100],"w_h")
            w_h2 = self.init_weights([100,100],"w_h2")
            w_o = self.init_weights([100,1],"w_o")
            
            ### bias terms:
            bias_1 = self.init_weights([100],"bias_1")
            bias_2 = self.init_weights([100],"bias_2")
            bias_3 = self.init_weights([1],"bias_3")
                
            h = tf.nn.elu(tf.add(tf.matmul(self.current_state, w_h),bias_1))
            h2 = tf.nn.elu(tf.add(tf.matmul(h, w_h2),bias_2))
            
        return tf.nn.elu(tf.add(tf.matmul(h2, w_o),bias_3))
        
    def source_dist_n(self):
        
        """
            This is the per-action source distribution, also known as the 
            exploration distribution. 
        """
        
        with tf.variable_scope("source"):
            
            tf.set_random_seed(self.seed)
            
            W_h = self.init_weights([4,100],"W_h")
            W_h2 = self.init_weights([100,50],"W_h2")
            W_o = self.init_weights([50,10],"W_o")
            
            # define bias terms:
            bias_1 = self.init_weights([100],"bias_1")
            bias_2 = self.init_weights([50],"bias_2")
                        
            eta_net = self.two_layer_net(self.source_input_n,W_h, W_h2, W_o,bias_1,bias_2)
            
            W_mu = self.init_weights([10,2],"W_mu")
            W_sigma = self.init_weights([10,2],"W_sigma")
            
            mu = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_mu)),self.bound)
            log_sigma = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_sigma)),self.bound)
        
        return mu, log_sigma
    
    
    def sampler(self,mu,log_sigma):
                        
        return np.random.normal(mu,np.exp(log_sigma))   
    
    def random_actions(self):
        """
            This baseline is used to check that the source isn't completely useless. 
        """
        
        return np.random.normal(0,self.bound,size = (self.horizon,2))
        
    
    def source_actions(self,state):
        
        actions = np.zeros((self.horizon,2))
        
        ### add a zero action to the state:
        AS_0 = np.concatenate((np.zeros(2),state))
        
        mu, log_sigma = self.sess.run([self.src_mu,self.src_log_sigma], feed_dict={ self.source_input_n: AS_0.reshape((1,4))})
                                                
        for i in range(1,self.horizon):
                        
            AS_n = np.concatenate((actions[i-1],state))
            
            mu, log_sigma = self.sess.run([self.src_mu,self.src_log_sigma], feed_dict={ self.source_input_n: AS_n.reshape((1,4))})
                        
            actions[i] = self.sampler(mu, log_sigma)
                    
        return actions


    def source(self):

        mu, log_sigma = self.src_mu, self.src_log_sigma
            
        dist = tf.contrib.distributions.MultivariateNormalDiag(mu, tf.exp(log_sigma))
                
        return tf.log(dist.prob(self.source_action)+tf.constant(1e-8))
        
    def decoder_dist_n(self): 
        
        """
            This is the per-action decoder, also known as the 
            planning distribution. 
        """
        
        with tf.variable_scope("decoder"):
            
            tf.set_random_seed(self.seed)
            
            W_h = self.init_weights([6,100],"W_h")
            W_h2 = self.init_weights([100,50],"W_h2")
            W_o = self.init_weights([50,10],"W_o")
            
            # define bias terms:
            bias_1 = self.init_weights([100],"bias_1")
            bias_2 = self.init_weights([50],"bias_2")
            
            eta_net = self.two_layer_net(self.decoder_input_n,W_h, W_h2, W_o,bias_1,bias_2)
            
            W_mu = self.init_weights([10,2],"W_mu")
            W_sigma = self.init_weights([10,2],"W_sigma")
            
            mu = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_mu)),self.bound)
            log_sigma = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_sigma)),self.bound)
            
        return mu, log_sigma
    
    
    def decoder(self):
        
        mu, log_sigma = self.decoder_mu, self.decoder_log_sigma
                
        dist = tf.contrib.distributions.MultivariateNormalDiag(mu, tf.exp(log_sigma))
                
        return dist.log_prob(self.source_action)
    
    def decoder_actions(self,ss_):
        
        actions = np.zeros((self.horizon,2))
        
        ### add a zero action to the state:
        SS_0 = np.concatenate((np.zeros(2),ss_))
        
        mu, log_sigma = self.sess.run([self.decoder_mu,self.decoder_log_sigma], feed_dict={ self.decoder_input_n: SS_0.reshape((1,6))})
                                                
        for i in range(1,self.horizon):
                        
            SS_n = np.concatenate((actions[i-1],ss_))
    
            mu, log_sigma = self.sess.run([self.decoder_mu,self.decoder_log_sigma], feed_dict={ self.decoder_input_n: SS_n.reshape((1,6))})
                                
            actions[i] = self.sampler(mu, log_sigma)
                    
        return actions
    
    def mean_decoder_actions(self,ss_):
        
        actions, sigmas = np.zeros((self.horizon,2)), np.zeros((self.horizon,2))
        
        ### add a zero action to the state:
        SS_0 = np.concatenate((np.zeros(2),ss_))
        
        mu, log_sigma = self.sess.run([self.decoder_mu,self.decoder_log_sigma], feed_dict={ self.decoder_input_n: SS_0.reshape((1,6))})
                                                
        for i in range(1,self.horizon):
                        
            SS_n = np.concatenate((actions[i-1],ss_))
    
            mu, log_sigma = self.sess.run([self.decoder_mu,self.decoder_log_sigma], feed_dict={ self.decoder_input_n: SS_n.reshape((1,6))})
                                
            actions[i], sigmas[i] = mu, np.exp(log_sigma)
                    
        return actions, sigmas