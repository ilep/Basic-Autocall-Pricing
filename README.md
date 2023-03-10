<p align="center">
  <img src="https://www.mckinsey.com/~/media/mckinsey/industries/financial%20services/our%20insights/a%20decade%20after%20the%20global%20financial%20crisis%20what%20has%20and%20hasnt%20changed/a-decade-after-the-global-financial-crisis-1536x1536-250.jpg?mw=677&car=42:25"  width="600" height="300">
</p>

# Basic Autocall Pricing

Some simplistic autocall pricing scripts > Go to [Autocall pricing notebook](Autocall%20pricing.ipynb) 


###  Intro
<p>
Pricing is done through Monte Carlo simulations. A large number of random price paths are generated for the underlying (or underlyings) via simulation. Then, for each path the associated exercise value (i.e. "payoff") of the option is calculated. These payoffs are then averaged and discounted to today. This result is the value of the option.
</p>

###  Keys parameters

```python
# Time period (in years) of the underlying price path. Trajectories are generated in the [0, T] interval
T = 1 

# Trajectories are not purely continuous. There are price points every delta_t 
delta_t = 1.0 / (365) # daily prices

# Volatility of the underlying
vol = 0.2 # 20% volatility

# On average, for small delta_t, underlying's return is equal to mu * delta_t between t and t + delta_t ==> delta_S  / S = mu * delta_t + vol * sqrt(delta_t) * N(0,1)
# mu is the expected continuously compounded rate at which grow the expected price. E(St) = S0 * exp(mu * t)
# In risk neutral universe, mu = r (risk free rate) and payoff is actualised using r to get the price
mu = r

# Initial price of underlying
S0 = 10
```


###  Random price path generation


```python
from models import RandomPaths

random_paths = RandomPaths(model="black-scholes", T=T, delta_t=delta_t, vol=vol, mu=mu, S0=S0, n_simulations=5)
random_paths.plot(legend=False, figsize=(10,5), rot=45)
```

###### Examples of some random price paths 
<img src="https://github.com/ilep/Basic-Autocall-Pricing/blob/main/doc/Capture.JPG">



###  Warm-up: European Call pricing

To test our Monte Carlo pricing, we start with a Call. 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Long_call_option.svg/1200px-Long_call_option.svg.png"  width="460" height="300">


```python
from models import Call

# Maturity 1 year, Strike 110%
call = Call(T=1, K=1.1*S0)

random_paths = RandomPaths(model="black-scholes", T=call.T, delta_t=1./(2*365), vol=vol, mu=mu, S0=S0, n_simulations=100000)

call.get_price(random_paths)
```

###  Autocall

###### Payoff explanation

We illustrate the autocall case with the following payoff: 

```python
T = 10 # maturity = 10 years

for year in range(1, maturity):
    
    i_year = int((1 / delta_t) * year)
    
    # Value of the underlying at year
    S_year = path.iloc[i_year]

    if (S_year / S0) > 1:
        payoff = 100 * (1. + year * self.coupon_rate)
        # products ends


S_T = path.iloc[-1]

if (S_T / S0) > 0.7:
    payoff = 100

else:
    payoff = (S_T / S0) * 100
```

Each year until the maturity (10 years):

- If the underlying price is above the starting price S0, the product ends and pays the initial capital invested plus a coupon that is proportional to the number of years since inception. 
- If not, the product continues to live.

A maturity:
- If the underlying price is above the starting price S0 &rarr; 100 + coupon_rate * 10
- If underlying is above 70% of the initial underlying value,  the initial capital is given back but no coupon &rarr; 100
- If underlying is under 70% of the initial underlying value, you start loosing money &rarr; no coupon and you are given back the level of the underlying (as if you were purely exposed to it)



###### Pricing

```python
from models import Autocall

athena = Autocall(T=10, coupon_rate=.07)

random_paths = RandomPaths(model="black-scholes", T=athena.T, delta_t=1./(2*365), vol=vol, mu=mu, S0=S0, n_simulations=50000)

athena.get_price(random_paths)
```

Typically, the "fundamental" obtained price will be around 98. The product is then sold 100 and the remaining 2 difference is shared between distributor and bank. 
