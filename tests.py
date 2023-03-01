# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:51:40 2023

@author: ilepoutre
"""

import pytest

# ! pip install option-price

from optionprice import Option


check_price = Option(european=True, kind='call',s0=10,k=11,t=365,sigma=0.2,r=0.01,dv=0).getPrice()


