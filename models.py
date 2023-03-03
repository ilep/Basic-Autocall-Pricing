# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:32:50 2023

@author: ilepoutre
"""

import numpy, pandas
from abc import ABC, abstractmethod

# discount rate / risk free rate
r = 0.01

class RandomPaths:
    """
    Class to generate underlying price trajectories relevant for Monte Carlo product pricing. 
    
    For black scholes, see: https://en.wikipedia.org/wiki/Geometric_Brownian_motion#:~:text=Geometric%20Brownian%20motion%20is%20used,model%20of%20stock%20price%20behavior.
    """
        
    def __init__(self, model="black-scholes", T=1, delta_t=1./365, vol=0.2, mu=r, S0=10, n_simulations=100):
        """
        - T: Time period (in years) of the underlying price path. Trajectories are generated in the [0, T] interval
        - delta_t: Trajectories are not purely continuous. There are price points every delta_t 
        - vol: volatility of the underlying
        - mu: 
            On average, for small delta_t, underlying's return is equal to mu * delta_t between t and t + delta_t ==> delta_S  / S = mu * delta_t + vol * sqrt(delta_t) * N(0,1)
            mu is the expected continuously compounded rate at which grow the expected price. E(St) = S0 * exp(mu * t)
            In risk neutral universe (W brownian motion under risk neutral probability), mu = r (risk free rate) and payoff is actualised using r to get the price
        - S0: initial price
        """        
        self.model = model 
        self.T = T 
        self.delta_t = delta_t
        self.vol = vol
        self.mu = mu
        self.S0 = S0
        self.n_simulations = n_simulations
        
        # set a dataframe parameters storing all generated random paths
        self._generate(n_simulations)
    
    
    def _generate(self, n_simulations):
        
        nbr_delta_t = int((1/self.delta_t)*self.T)
        
        if self.model == 'black-scholes':
            np_delta_r = numpy.random.normal(loc= 1 + self.mu * self.delta_t, scale=self.vol * numpy.sqrt(self.delta_t), size=(n_simulations, nbr_delta_t))
            np_delta_r = numpy.concatenate((numpy.ones(shape=(n_simulations, 1)), np_delta_r), axis=1)
            
        cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j * self.delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j * self.delta_t)) for j in range(1, nbr_delta_t+1)]
        indexes = ['traj_%d' %k for k in range(1, n_simulations+1)]
        delta_r_df = pandas.DataFrame(np_delta_r, columns=cols, index=indexes)
        
        df = self.S0 * delta_r_df.cumprod(axis=1)
        
        self.df = df

         
    
    def plot(self, **options):
        self.df.T.plot(**options)
        



class Product(ABC):

    def __init__(self, T):    
        self.T = T
        super().__init__()

    @abstractmethod
    def get_current_values(self, trajectories):
        pass



class Call(Product):
    
    """
    K: option strike
    """
    def __init__(self, T, K):    
        super().__init__(T)
        self.K = K

    def get_current_values(self, random_paths):
        return numpy.exp(-r * self.T) * numpy.maximum(random_paths.df.iloc[:,-1] - self.K, 0)


    def get_price(self, random_paths):
        return numpy.mean(self.get_current_values(random_paths))


    
class Autocall(Product):
    
    def __init__(self, T=5, coupon_rate=.07):    
        super().__init__(T)
        self.coupon_rate = coupon_rate
        
        

    def _get_current_value(self, path, delta_t):
        
        S0 = path.iloc[0]
        Sf = path.iloc[-1]
        
        # k_day_subdivision = (1 / delta_t) / 365
        
        for year in range(1, self.T):
            i_year = int((1 / delta_t) * year)
            print(i_year)
            assert(path.index[i_year] == ('t0 + %d * year(s)' % year))
            
            if (Sf / S0) > 1:
                payoff = 100 * (1. + year * self.coupon_rate)
                current_value = numpy.exp(-year*r) * payoff
                return current_value
        
        year=self.T
        i_year = int(1 / delta_t) * year
        assert(path.index[i_year] == 't0 + 5 * year(s)')
        
        if (Sf / S0) > 0.7:
            payoff = 100
            
        else:
            payoff = (Sf / S0) * 100
            
        current_value = numpy.exp(-year*r) * payoff
        
        return current_value

    def get_current_values(self, random_paths):
        
        current_values = random_paths.df.apply(self._get_current_value, axis=1, args=(random_paths.delta_t,))
        self.df_current_values = current_values
        return current_values
        

    def get_price(self, random_paths):
    
        self.get_current_values(random_paths)
        price = self.df_current_values.mean() 
        
        return price









