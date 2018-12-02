#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:32:53 2018

@author: aidanrocke & ildefonsmagrans
"""

#import random
import tensorflow as tf
import numpy as np

#import matplotlib.pyplot as plt
from agent import agent_cognition
from square_env import square_env
from utils import action_states
from visualisation import heatmap

## set random seed:
tf.set_random_seed(42)

# define training parameters:
horizon = 3
seed = 42
bound = 1.0
iters = 10000
batch_size = 50
lr = 0.01
prob = 0.8
R = 0.5

## define folder where things get saved:
folder = "/Users/aidanrockea/Desktop/vime/heat_maps/expt_7/"

# define environment:
env = square_env(duration=horizon,radius=R,dimension=2*horizon*R)

def main():
            
    with tf.Session() as sess:
                
        A = agent_cognition(horizon,sess,seed,bound)          
        
        ### define beta schedule:
        betas = 1./np.array([min(0.001 + i/iters,1) for i in range(iters)])
        
        ## define inverse probability:
        inverse_prob = betas
        
        ### initialise the variables:
        sess.run(A.init_g)
                                
        for count in range(iters):
            
            ## reset the environment:
            env.reset()
            env.random_initialisation()
            
            mini_batch = np.zeros((batch_size*horizon,6))
            
            ## define mean and variance of environment:
            mu = env.dimension/2.0 - R ## mean of U(0,dimension)
            sigma = ((2*mu)**2)/12 ## variance of U(R,dimension-R)
            
            ### train our agent on a minibatch of recent experience:
            for i in range(batch_size):
                
                env.iter = 0
                                            
                if np.random.rand() > 1/inverse_prob[count]:
                    actions = A.random_actions()
                else:
                    normalised_state = (env.state_seq[env.iter]-mu)/sigma
                    
                    actions = A.source_actions(normalised_state)
                    
                env.iter += 1
                        
                ## get responses from the environment:
                env.env_response(actions,A.horizon)
                                    
                ## group actions, initial state, and final state:                        
                axx_ = action_states(env,A,actions)
                
                mini_batch[horizon*i:horizon*(i+1)] = axx_
            
            ## normalise the state representations:
            mini_batch[:,2:6] = (mini_batch[:,2:6] - mu)/sigma
                
            train_feed_1 = {A.decoder_input_n : mini_batch,A.source_action : mini_batch[:,0:2],\
                            A.prob : prob,A.lr:lr}
            
            sess.run(A.train_decoder,feed_dict = train_feed_1)
                
            # train source and critic:
            train_feed_2 = {A.beta: betas[count].reshape((1,1)), A.current_state: mini_batch[:,2:4],\
                            A.decoder_input_n : mini_batch, A.source_input_n : mini_batch[:,0:4], \
                            A.source_action : mini_batch[:,0:2],
                            A.prob : prob,A.lr:lr}
            
            sess.run(A.train_critic_and_source,feed_dict = train_feed_2)
            
            if count % 500 == 0:   
                heatmap(0.1,sess,A,env,count,folder)
                        
        
if __name__ == "__main__":
    main()
    
