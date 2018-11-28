#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:32:53 2018

@author: aidanrocke & ildefonsmagrans
"""

#import random
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from agent import agent_cognition
from square_env import square_env
from utils import dual_opt, action_states
from visualisation import heatmap, variance_progression, action_progression

## set random seed:
tf.set_random_seed(42)

# define number of iters:
horizon = 3
seed = 42
bound = 1.0
iters = 20000 ### must be a perfect square
batch_size = 50

# define environment:
env = square_env(duration=horizon,radius=0.5,dimension=2*(horizon-1.0))

def main():
    # the concatenation of a state and action
    count, obs = 0, 0
    
    ## define arrays to contain means and variances:
    actions_, variances_ = np.zeros((20,10,2)), np.zeros((20,10,2))
    
    x, y = np.linspace((horizon-1),2*(horizon-1),20), np.array([(horizon-1)]*20)
    X = np.stack((x,y),1)

        
    with tf.Session() as sess:
                
        A = agent_cognition(horizon,sess,seed,bound)   
            
        ### define the decoder, critic and source:
        log_decoder = A.decoder()
        E = A.empowerment_critic()
        log_source = A.source()
                
        # define a placeholder for beta values in the squared loss:
        beta =tf.placeholder(tf.float32, [None, 1])       
        
        ### it might be a good idea to regularise the squared loss:
        squared_loss = tf.reduce_mean(tf.square(beta*log_decoder-E-log_source))
        decoder_loss = tf.reduce_mean(tf.multiply(tf.constant(-1.0),log_decoder))
        ### define beta schedule:
        #betas = np.linspace(10,1,iters)
        
        betas = 1./np.array([min(0.001 + i/iters,1) for i in range(iters)])
        
        
        ## define the inverse probability to learn from randomness: 
        N = min(iters,10000)
        
        inverse_prob = np.hstack((1./np.array([min(0.001 + i/N,1) for i in range(N)]),np.ones(iters-N)))
                
        ### define the optimiser:
        fast_optimizer = tf.train.AdagradOptimizer(0.01)
        slow_optimizer = tf.train.AdagradOptimizer(0.01)
        
        train_decoder = fast_optimizer.minimize(decoder_loss)
        
        ### define a dual optimizatio method for critic and source:
        train_critic_and_source = dual_opt("critic", "source", squared_loss, slow_optimizer)
        
        ### initialise the variables:
        sess.run(tf.global_variables_initializer())
        
        squared_losses, decoder_losses = np.zeros(iters), np.zeros(iters)
                        
        while count < iters:
            
            env.reset()
            env.random_initialisation()
            #env.cyclic_initialisation(count,iters)
            
            mini_batch = np.zeros((batch_size*horizon,6))
            
            ### train our agent on a minibatch of recent experience:
            for i in range(batch_size):
            
            
                ## reset the environment:
                env.iter = 1          
                
                prob = np.random.rand()
                            
                if prob > 1/inverse_prob[count]:
                    actions = A.random_actions()
                else:
                    actions = A.source_actions(env.state_seq[env.iter])
                        
                ## get responses from the environment:
                env.env_response(actions,A.horizon)
                                    
                ## group actions, initial state, and final state:                        
                axx_ = action_states(env,A,actions)
                
                mini_batch[horizon*i:horizon*(i+1)] = axx_
                
            train_feed_1 = {A.decoder_input_n : mini_batch,A.source_action : mini_batch[:,0:2]}
            sess.run(train_decoder,feed_dict = train_feed_1)
                
            # train source and critic:
            train_feed_2 = {beta: betas[count].reshape((1,1)), A.current_state: mini_batch[:,2:4],A.decoder_input_n : mini_batch, A.source_input_n : mini_batch[:,0:4], A.source_action : mini_batch[:,0:2]}
            sess.run(train_critic_and_source,feed_dict = train_feed_2)
                
            squared_losses[count] = sess.run(squared_loss,feed_dict = train_feed_2)
            decoder_losses[count] = sess.run(decoder_loss,feed_dict = train_feed_1)
            
            ## check variances:
            
            if count % 1000 == 0:
                for j in range(10):
                    # test the decoder:
                    ss_ = np.concatenate((np.zeros(2),X[j],X[j])).reshape((1,6))
                    mu, log_sigma = A.sess.run([A.decoder_mu,A.decoder_log_sigma], feed_dict={ A.decoder_input_n: ss_})                    
                    actions_[obs][j], variances_[obs][j] = mu[0], np.exp(log_sigma[0])
                    
                    print(mu[0])
                    
                obs += 1
            
            count += 1
            
            folder = "/Users/aidanrockea/Desktop/vime/images/expt_8/"
            
            if count % 500 == 0:   
                heatmap(0.1,sess,A,E,env,count,folder)
                
            if count % iters == 0:
                variance_progression(x,variances_,folder)
                action_progression(x,actions_,folder)
                
                #potential(0.1,sess,A,E,env,count,folder)
            
        #view_losses(squared_losses,decoder_losses)
        plt.plot(squared_losses)        
        
        
if __name__ == "__main__":
    main()
    
