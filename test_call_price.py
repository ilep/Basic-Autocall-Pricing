# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:51:40 2023

@author: ilepoutre
"""

import pytest

# ! pip install option-price
# pip install -U pytest

from optionprice import Option
from models import RandomPaths, r, Call




def test_call_price_close_to_check():
    
    call = Call(T=1, K=11)
    
    random_paths = RandomPaths(model="black-scholes", T=call.T, delta_t=1./(2*365), vol=.2, mu=r, S0=10, n_simulations=200000)
    price_to_be_checked = call.get_price(random_paths)
    
    check_price = Option(european=True, kind='call', s0=10, k=11, t=365, sigma=0.2, r=r, dv=0).getPrice()

    error_abs = (abs(price_to_be_checked - check_price) / check_price)
    print(f"Error = {error_abs} %")

    assert error_abs < 0.01
