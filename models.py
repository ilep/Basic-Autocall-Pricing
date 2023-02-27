# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:32:50 2023

@author: ilepoutre
"""

import numpy, pandas
from abc import ABC, abstractmethod


class RandomPaths:
    """
    Class to generate underlying price trajectories relevant for Monte Carlo product pricing. 
    
    For black scholes, see: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#:~:text=Geometric%20Brownian%20motion%20is%20used,model%20of%20stock%20price%20behavior.
    """
        
    def __init__(self, model="black-scholes", T=1, delta_t=1./365, vol=0.2, r=.01, S0=10, n_simulations=100):
        """
        T: Time period (in years) of the underlying price path. Trajectories are generated in the [0, T] interval
        delta_t: Trajectories are not purely continuous. There are price points every delta_t 
        vol: volatility of the underlying
        r: On average, underlying's return is r * delta_t between t and t + delta_t ==> dSt / St = r * dt + vol * dWt (Wt is a standard brownian motion)
        S0: initial price
        """        
        self.model = model 
        self.T = T 
        self.delta_t = delta_t
        self.vol = vol
        self.r = r
        self.S0 = S0
        self.n_simulations = n_simulations

    
    
    def generate(self):
        
        nbr_delta_t = int((1/self.delta_t)*self.T)
        
        np_delta_r = numpy.random.normal(loc= 1 + self.r * self.delta_t, scale=self.vol * numpy.sqrt(self.delta_t), size=(self.n_simulations, nbr_delta_t))
        np_delta_r = numpy.concatenate((numpy.ones(shape=(self.n_simulations, 1)), np_delta_r), axis=1)
        
        cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j * self.delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j * self.delta_t)) for j in range(1, nbr_delta_t+1)]
        indexes = ['traj_%d' %k for k in range(1, self.n_simulations+1)]
        delta_r_df = pandas.DataFrame(np_delta_r, columns=cols, index=indexes)
        
        trajs = self.S0 * delta_r_df.cumprod(axis=1)
        
        self.trajectories = trajs
        
        return trajs




class Product(ABC):

    def __init__(self, T):    
        self.T = T
        super().__init__()

    @abstractmethod
    def get_current_values(self, trajectories):
        pass

    @abstractmethod
    def plot_payoff(self):
        pass

# discount rate
Rd = 0.01

def Call(Product):
    
    """
    K: option strike
    """
    def __init__(self, T, K):    
        super().__init__(id, T)
        self.K = K

    def get_current_values(self, trajectories):
        return numpy.exp(-Rd * self.T) * numpy.maximum(trajectories.iloc[:,-1] - self.K, 0)


    def get_price(self, trajectories):
        return numpy.mean(get_current_values(self, trajectories))


    
class Autocall(Product):
    
    def __init__(self, T=5, coupon_rate=.07):    
        super().__init__(id, T)
        self.coupon_rate = coupon_rate
        


    def _get_current_value(self, traj, delta_t):
        
        S0 = traj.iloc[0]
        Sf = traj.iloc[self.T]
        
        for year in range(1, self.T):
            i_year = int((1 / delta_t) * year)
            assert(traj.index[i_year] == ('t0 + %d * year(s)' % year))
            
            if (Sf / S0) > 1:
                payoff = 100 * (1. + year * self.coupon_rate)
                current_value = numpy.exp(-year*Rd) * payoff
                return current_value
        
        year=self.T
        i_year = int(1 / delta_t) * year
        assert(traj.index[i_year] == 't0 + 5 * year(s)')
        
        if (Sf / S0) > 0.7:
            payoff = 100
            
        else:
            payoff = (Sf / S0) * 100
            
        current_value = numpy.exp(-year*Rd) * payoff
        
        return current_value

    def get_price(self, random_path):
        
        self.df_current_values = random_path.trajectories.apply(self._get_current_value, axis=1, args=(random_path.delta_t,))
        price = self.df_current_values.mean() 

        return price









