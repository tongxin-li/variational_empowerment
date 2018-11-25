#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:47:40 2017

@author: aidanrocke & ildefonsmagrans
"""

import numpy as np

class square_env:
    def __init__(self,duration,diameter,dimension):
        if diameter > dimension:
            raise Warning("diameter can't exceed dimensions")
        self.R = float(diameter)/2 # radius of agent
        self.dimension = dimension # LxW of the square world
        self.iter = 0 # current iteration
        self.duration = duration # maximum duration of our environment
        self.state_seq = np.zeros((self.duration,2))
                
    def random_initialisation(self):
        # define the objective measure: 
        self.state_seq[self.iter][0] = np.random.uniform(0,self.dimension)
        self.state_seq[self.iter][1] = np.random.uniform(0,self.dimension)
        
        self.iter = 1
        
    def cyclic_initialisation(self,epoch,epochs):
        """
            return a pair of points defining the location of grid points
        """
        root = int(np.sqrt(epochs))
        
        delta = float(self.dimension)/(2*root)
                
        self.state_seq[self.iter][0] = delta*(epoch % root)
        self.state_seq[self.iter][1] = delta*int(epoch/root)
        
        self.iter = 1
        
    def square_initialisation(self,epoch,epochs):
        """
            return a random pair of points radially converging to the centre
        """
        delta = float(self.dimension)/(2*epochs)
        D = self.dimension/2
        
        alpha = D-epoch*delta
                
        self.state_seq[self.iter][0] = D + np.random.uniform(-alpha,alpha)
        self.state_seq[self.iter][1] = D + np.random.uniform(-alpha,alpha)
        
        self.iter = 1
    
    def observation(self,action,noise = 1):
        self.env.step(action)
        
        raw_observation = self.env.state_seq[self.env.iter]
        
        if noise == 1:
        
        # add noise to observation:
            return self.input_noise(raw_observation)
        
        else:
            
            return raw_observation
        
    def boundary_conditions(self,epsilon):
        
        #boundary conditions:
        cond_X = (self.state_seq[self.iter-1][0] >= self.R+epsilon)*(self.state_seq[self.iter-1][0] <= self.dimension-self.R-epsilon)
        cond_Y = (self.state_seq[self.iter-1][1] >= self.R+epsilon)*(self.state_seq[self.iter-1][1] <= self.dimension-self.R-epsilon)

        return cond_X, cond_Y
        
    def step(self, action):
                
        self.state_seq[self.iter] = self.state_seq[self.iter-1] + action
        
        #boundary conditions:
        cond_X, cond_Y = self.boundary_conditions(0)
        
        #both conditions must be satisfied:
        if cond_X*cond_Y == 0:
            self.state_seq[self.iter] -= action
            
        self.iter += 1

            
        if self.iter > self.duration:
            raise Exception("Game over!")            
            
            
    def env_response(self,actions,horizon):
        # update the environment
        
        for i in range(1,horizon):
            self.step(actions[i])
        
        
    def reset(self):
        """
        Return to the initial conditions. 
        """
        self.state_seq = np.zeros((self.duration,2))
        self.iter = 0
        
    def near_boundary(self,epsilon):
        """
            Here epsilon is a measure of proximity to the boundary. 
        """
        
        #boundary conditions:
        cond_X, cond_Y = self.boundary_conditions(epsilon)
        
        if cond_X*cond_Y == 0:
            
            return 1
        
        else:
            
            return 0
        
        
        
    

    

