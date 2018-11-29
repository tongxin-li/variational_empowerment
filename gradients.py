#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:06:26 2018

@author: aidanrockea
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

X_1 = tf.placeholder(tf.float32, [None, 2])
X_2 = tf.placeholder(tf.float32, [None, 6])

def init_weights(shape,var_name):
        """
            Xavier initialisation of neural networks
        """
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape),name = var_name)
        
def two_layer_net(X, w_h, w_h2, w_o,bias_1, bias_2):
    """
        A generic method for creating two-layer networks
        
        input: weights
        output: neural network
    """
    
    h = tf.nn.elu(tf.add(tf.matmul(X, w_h),bias_1))
    h2 = tf.nn.elu(tf.add(tf.matmul(h, w_h2),bias_2))
    
    return tf.matmul(h2, w_o)


with tf.variable_scope("nnet"):
                
    W_h = init_weights([6,100],"W_h")
    W_h2 = init_weights([100,50],"W_h2")
    W_o = init_weights([50,10],"W_o")
    
    # define bias terms:
    bias_1 = init_weights([100],"bias_1")
    bias_2 = init_weights([50],"bias_2")
    
    eta_net = two_layer_net(X_2,W_h, W_h2, W_o,bias_1,bias_2)
    
    W_mu = init_weights([10,2],"W_mu")
    W_sigma = init_weights([10,2],"W_sigma")
    
    mu = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_mu)),1.0)
    log_sigma = tf.multiply(tf.nn.tanh(tf.matmul(eta_net,W_sigma)),1.0)
    
decoder_dist = tfp.distributions.MultivariateNormalDiag(mu, tf.exp(log_sigma))

log_prob = decoder_dist.log_prob(X_1)

## get gradients:
var = tf.trainable_variables("nnet")
gradients = tf.gradients(log_prob,[var[0]])[0]
norm = tf.norm(gradients)


init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    for i in range(3):
    
        train_feed = {X_1:np.zeros((1,2)),X_2:np.ones((1,6))}
        norm_ = session.run(norm,feed_dict=train_feed)
        print(norm_)
    
    
