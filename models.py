# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:32:50 2023

@author: ilepoutre
"""

import numpy, pandas
from abc import ABC, abstractmethod


class RandomPath:
    """
    Class to generate underlying price trajectories relevant for Monte Carlo product pricing. 
    """
        
    def __init__(self, model="black-scholes", T=1, delta_t=1. / 365, vol=0.2, r=.01, S0=10, K=9):
        """
        T: Time period (in years) of the underlying price path. Trajectories are generated in the [0, T] interval
        delta_t: Trajectories are not purely continuous. There are price points every delta_t 
        vol: volatility of the underlying
        r = On average, underlying's return is r between t and t + delta_t ==> dSt / St = r * dt + vol * dWt (Wt is a standard brownian motion)
        S0: initial price
        K: Strike
        """        
        self.model = model 
        self.T = T 
        self.delta_t = delta_t
        self.vol = vol
        self.r = r
        self.S0 = S0

    
    
    def generate(self, n_simulations):
        
        nbr_delta_t = int((1/self.delta_t)*self.T)
        delta_r = numpy.random.normal(loc=1+self.r*self.delta_t, scale=self.vol*numpy.sqrt(self.delta_t), size=(n_simulations, nbr_delta_t))
        delta_r = numpy.concatenate((numpy.ones(shape=(n_simulations,1)), self.delta_r), axis=1)
        
        cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j*self.delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j*self.delta_t)) for j in range(1, nbr_delta_t+1)]
        indexes = ['traj_%d' %k for k in range(1, n_simulations+1)]
        
        delta_r_df = pandas.DataFrame(delta_r, columns=cols, index=indexes)
        trajs = self.S0 * delta_r_df.cumprod(axis=1)
        
        self.trajectories = trajs
        
        return trajs



class Product(ABC):

    def __init__(self, T):    
        self.T = T

    def plot_payoff(self):
        pass




