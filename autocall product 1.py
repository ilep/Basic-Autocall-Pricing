#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:02:41 2017

@author: ivan.lepoutre
"""


import numpy, pandas
from models import RandomPaths



T = 1
delta_t = 1.0 / (365)
vol = 0.2
r = 0.01
S0 = 10
K = 9



def generate_bs_trajs(S0, K, T, delta_t, vol, r, n_simus):
    
    nbr_delta_t = int((1/delta_t)*T)
    delta_r = numpy.random.normal(loc=1+r*delta_t, scale=vol*numpy.sqrt(delta_t), size=(n_simus,nbr_delta_t))
    delta_r=numpy.concatenate((numpy.ones(shape=(n_simus,1)), delta_r), axis=1)
    
    cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j*delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j*delta_t)) for j in range(1, nbr_delta_t+1)]
    indexes = ['traj_%d' %k for k in range(1, n_simus+1)]
    
    delta_r_df = pandas.DataFrame(delta_r, columns=cols, index=indexes)
    trajs_bs = S0 * delta_r_df.cumprod(axis=1)
    return trajs_bs
    
    
trajs_bs = generate_bs_trajs(S0, K, T, delta_t, vol, r, 10000)
    
trajs_bs.iloc[1,:].plot(rot=45)

    


def call_price(S0, K, T, delta_t, vol, r, n_simus):    
    
    trajs_bs = generate_bs_trajs(S0, K, T, delta_t, vol, r, n_simus)
    call_price = numpy.exp(-r*T) * numpy.mean(numpy.maximum(trajs_bs.iloc[:,-1] - K, 0))
    return call_price

    
call_price(10, 11, 1, 1.0/(2*365), 0.2, 0.01, 100000)    

# robust price
numpy.mean([call_price(10, 11, 1, 1.0/(2*365), 0.2, 0.01, 10000) for _ in range(0,30)])
    



S0 = 100
K = 100

traj = trajs_bs.iloc[0,:]
def product1_current_value_traj(traj, K, delta_t):
    

    for year in range(1,5):
        i_year = int((1/delta_t) * year)
        assert(traj.index[i_year] == ('t0 + %d * year(s)' % year))
        
        if traj.iloc[i_year] > K:
            payoff = 100 + year *7
            current_value = numpy.exp(-year*r) * payoff
            return current_value
    year=5
    i_year = int(1/delta_t) * year
    assert(traj.index[i_year] == 't0 + 5 * year(s)')
    if traj.iloc[i_year] > 0.7 * K:
        payoff = K
        
    else:
        payoff = (traj.iloc[i_year]/S0)*100
        
    current_value = numpy.exp(-year*r) * payoff
    return current_value


    
trajs_bs = generate_bs_trajs(S0, K, 5, 1.0/365, vol, r, 1000)
  
#athena_bs_price = trajs_bs.apply(athena_current_value_traj, axis=1,args=(K,delta_t,)).mean() 




# 2/ price_athena = f(S, t=0) GRAPHE 
def product1_price(S0, K,delta_t, vol, r, traj_type='bs'):
    if traj_type == 'bs':
        
        trajs_bs = generate_bs_trajs(S0, K, 5, delta_t, vol, r, 10000)
        assert(trajs_bs.shape[1] == (5 *(1/delta_t) +1))    
      
    athena_price = trajs_bs.apply(product1_current_value_traj, axis=1, args=(K,delta_t,)).mean() 
    
    return athena_price

S0 = 100
K=100    
athena_price(S0, K, 1.0/(2*365),vol, r, 'bs')
    
    
S0_s = pandas.Series()
  
# 2/ price_athena = f(S, t=0) GRAPHE    
ath_prices_S0_0 = pandas.Series([athena_price(s, K, 1.0/(2*365), vol, r, 'bs') for s in range(10,150,5)], index = range(10,150,5))
 

import matplotlib.pyplot as plt
ax = ath_prices_S0_0.plot(title="price fonction of S0, K=100, vol 20% r 1%")
ax.get_figure().savefig('ath_prices_S0_0.jpeg')




# 3/ S=S0, t=0, price_athena = f(vol) GRAPHE

vol_range = numpy.linspace(start=0.05, stop=0.4, num=10)
ath_prices_vol = pandas.Series([athena_price(S0, K, 1.0/(2*365), v , r, 'bs') for v in vol_range], index = vol_range) 
ax = ath_prices_vol.plot(title="price fonction of vol S0=100, K=100, r 1%")
ax.get_figure().savefig('ath_prices_vol.jpeg')



# 4/ S=S0, t=0, price_athena = f(r) GRAPHE
S0, K, vol
r_range = numpy.linspace(start=0.005, stop=0.05, num=10)
ath_prices_r = pandas.Series([athena_price(S0, K, 1.0/(2*365), vol , r_, 'bs') for r_ in r_range], index = r_range)
 
ax = ath_prices_r.plot(title="price fonction of r,  S0=100, K=100, vol 20%")
ax.get_figure().savefig('ath_prices_r.jpeg')



# 5/ vol smilee, generation de trajs , prix = f(beta) GRAPHE 15h30
def generate_vol_smilee_trajs(S0, K, T, delta_t, vol, beta, r, n_simus):
    nbr_delta_t = int((1/delta_t)*T)
    
    cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j*delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j*delta_t)) for j in range(1, nbr_delta_t+1)]
    indexes = ['traj_%d' %k for k in range(1, n_simus+1)]
    
    def f_vol(S, S0, vol, beta):
        return numpy.min(vol * ((S/S0) **(-beta)), 4*vol)

    trajs_vol_smilee = pandas.DataFrame(index=indexes, columns=cols)
    
    trajs_vol_smilee.loc['t0'] = S0
    
    for simu_ii in range(0,n_simus):
        
        for j in range(1, nbr_delta_t+1):
            Sjm1 = trajs_vol_smilee.iloc[simu_ii, j-1]
            trajs_vol_smilee.iloc[simu_ii, j] = Sjm1 * (1 + delta_t*r) + f_vol(Sjm1, S0, vol, beta) * numpy.sqrt(delta_t) * numpy.random.normal()
        
    return trajs_vol_smilee

    
def athena_price_vol_smilee(S0, K,delta_t, vol, beta, r):

    trajs_vol_smilee = generate_vol_smilee_trajs(S0, K, 5, delta_t, vol, beta, r, 1000)
    assert(trajs_bs.shape[1] == (5 *(1/delta_t) +1))    
  
    athena_price = trajs_vol_smilee.apply(athena_current_value_traj, axis=1, args=(K,delta_t,)).mean() 
    
    return athena_price
    

S0, K, vol, r
    
athena_price_vol_smilee(S0, K,delta_t, vol, -0.5, r)
    

beta_range = numpy.linspace(start=-0.75, stop=0., num=10)
ath_prices_beta = pandas.Series([athena_price_vol_smilee(S0, K, 1.0/(2*365), vol , b,  r) for b in beta_range], index = beta_range)
 
ax = ath_prices_beta.plot(title="price fonction of beta (vol smilee),  S0=100, K=100, vol 20%, r 1%")
ax.get_figure().savefig('ath_prices_beta.jpeg')



# 6/ generation trajs, taux sto, prix = f(vol_taux) GRAPHE 16h

vol_r=0.05
n_simus=10
def generate_tx_sto_trajs(S0, K, T, delta_t, vol, r,vol_r, n_simus):
    
    
    nbr_delta_t = int((1/delta_t)*T)
    
    eps1= numpy.random.normal(0,1,size=(n_simus,nbr_delta_t))
    
    eps_temp = numpy.random.normal(0,1,size=(n_simus,nbr_delta_t))
    
    eps2 = 0.5 * (eps1 + eps_temp) 
    
    delta_r_det = (1+r*delta_t) * numpy.ones(shape=(n_simus,nbr_delta_t))
    
    delta_r_sto = (vol * eps1 + vol_r * eps2)*numpy.sqrt(delta_t)
    
    #delta_r = numpy.random.normal(loc=1+r*delta_t, scale=vol*numpy.sqrt(delta_t), size=(n_simus,nbr_delta_t))
    
    delta_r = delta_r_det + delta_r_sto
    
    
    delta_r=numpy.concatenate((numpy.ones(shape=(n_simus,1)), delta_r), axis=1)
    
    cols= ['t0'] + [('t0 + %d * delta_t' % j) if ((j*delta_t)%1 !=0.) else ('t0 + %d * year(s)' % int(j*delta_t)) for j in range(1, nbr_delta_t+1)]
    indexes = ['traj_%d' %k for k in range(1, n_simus+1)]
    
    delta_r_df = pandas.DataFrame(delta_r, columns=cols, index=indexes)
    trajs_r_sto = S0 * delta_r_df.cumprod(axis=1)
    return trajs_r_sto

    
def athena_price_tx_sto(S0, K,delta_t, vol, r, vol_r):

    trajs_tx_sto = generate_tx_sto_trajs(S0, K, 5, delta_t, vol, r,vol_r, 10000)
    assert(trajs_tx_sto.shape[1] == (5 *(1/delta_t) +1))    
  
    athena_price = trajs_tx_sto.apply(athena_current_value_traj, axis=1, args=(K,delta_t,)).mean() 
    
    return athena_price


def robust_athena_price_tx_sto(S0, K,delta_t, vol, r, vol_r):
    return numpy.mean([athena_price_tx_sto(S0, K,delta_t, vol, r, vol_r) for _ in range(0,20)])
    
athena_price_tx_sto(S0, K,delta_t, vol, r, 0.005) 
    
vol_sto_range =  numpy.linspace(start=0., stop=0.01, num=20)



ath_prices_vol_sto = pandas.Series([athena_price_tx_sto(S0, K, 1.0/(2*365), vol,r, vol_r_) for vol_r_ in vol_sto_range], index = vol_sto_range) 
ax = ath_prices_vol_sto.plot(title="price fonction of vol sto,  S0=100, K=100, vol stock 20%, r 1%")
ax.get_figure().savefig('ath_prices_vol_sto.jpeg')



ath_prices_vol_sto_2 = pandas.Series([robust_athena_price_tx_sto(S0, K, 1.0/(2*365), vol,r, vol_r_) for vol_r_ in vol_sto_range], index = vol_sto_range) 
ax = ath_prices_vol_sto_2.plot(title="robust price fonction of vol sto,  S0=100, K=100, vol stock 20%, r 1%")
ax.get_figure().savefig('ath_prices_vol_sto_2.jpeg')

# 7/ EDP, schema explicite ==> 18h

# 17 theorie finie

# 18 code fini


# https://github.com/sarahrn/StochasticCalculus/blob/master/functions.R

# bonne explication chgt de variable 
# https://ensiwiki.ensimag.fr/images/8/8e/IRL_Rapport_Bonkoski_Kevin.pdf


# https://ensiwiki.ensimag.fr/images/8/8e/IRL_Rapport_Bonkoski_Kevin.pdf

#http://wwwf.imperial.ac.uk/~mdavis/FDM11/LECTURE_SLIDES2.PDF


# TOP 
#http://www.math.yorku.ca/~hmzhu/Math-6911/lectures/Lecture5/5_BlkSch_FDM.pdf
# http://www.math.yorku.ca/~hmzhu/Math-6911/lectures/Lecture5/5_BlkSch_FDM.pdf


r = 0.01
K = 11
S_min = 0.
S_max = 2. * K

T = 1.0

M = 200 # nombres de pas en espace
N = 365 # nombres de pas de temps

delta_t = T / N
delta_S = (S_max - S_min) / M

V = pandas.DataFrame(index=['0'] + ['%d * delta_S' %i for i in range(1, M+1)], columns= ['0'] + ['%d * delta_t' %j for j in range(1, N+1)])

V.head()

# final condition 


V.iloc[:,-1] = numpy.maximum(numpy.array([i*delta_S for i in range(0, M+1)]) - K, 0)
V.iloc[0,:] = 0
V.iloc[-1,:] = S_max - K * numpy.exp([-r*(N-j)*delta_t for j in range(0,N+1)])

# en reprenant les notations de http://www.math.yorku.ca/~hmzhu/Math-6911/lectures/Lecture5/5_BlkSch_FDM.pdf
def a(i, delta_t, vol, r):
    return 0.5 * delta_t * ((vol*i)**2 - r*i)
    
def b(i, delta_t, vol, r):
    return 1 - delta_t * ((vol*i)**2 +r)

def c(i, delta_t, vol, r):
    return 0.5 * delta_t * ((vol*i)**2 + r*i)
    
    
# V[i, j-1] = ai * V[i-1, j] + bi * V[i,j] + ci * V[i+1,j]

j=[N-j for j in range(0,N+1)][0]
i=range(1,M)[0]

for j in [N-j for j in range(0,N+1)]:
    print(j)
    for i in range(1,M):
        #print(a(i, delta_t, vol, r) * V.iloc[i-1, j] + b(i, delta_t, vol, r) * V.iloc[i,j] + c(i, delta_t, vol, r) * V.iloc[i+1,j])
        
        V.iloc[i, j-1] = a(i, delta_t, vol, r) * V.iloc[i-1, j] + b(i, delta_t, vol, r) * V.iloc[i,j] + c(i, delta_t, vol, r) * V.iloc[i+1,j]
        #print(V.iloc[:,j-1])

V.to_csv('V_EDP.csv')



# Hull page 481, valeur du put grille fi,j
K=11.

T =1.

N = 365 # on decoupe T en N parties , 1 partie = 1 jour du coup
delta_t = T /N

S0_target =  10.

M=200
delta_S = S0_target / 80
S_max = M*delta_S

# fi,j valeur de l' option a 

# i * delta_t (N)

# j * delta_S (M)



f = pandas.DataFrame(index=['%d * delta_t' % i for i in range(0,N+1)], 
                            columns=['%d * delta_S' % j for j in range(0,M+1)])

# a T, le valeur de l option vaut (ST -K)+
f.iloc[N,:] = numpy.maximum(numpy.array([j*delta_S for j in range(0, M+1)]) - K, 0)

# qd le prix de l action est nul, la valeur du cal est nulle
f.iloc[:,0] = 0

# qd S=Smax, on suppose que la valeur du call est 

f.iloc[:,-1] = S_max  - K * numpy.exp([-r * delta_t * (N-i) for i in range(0,N+1)])


def a(j, delta_t, vol, r):
    return (1./(1+delta_t))*(-0.5*r*j*delta_t + 0.5*((vol*j)**2)*delta_t)
    
def b(j, delta_t, vol, r):
    return (1./(1+delta_t))*(1-((vol*j)**2)*delta_t)

def c(j, delta_t, vol, r):
    return (1./(1+delta_t))*(0.5*r*j*delta_t + 0.5*((vol*j)**2)*delta_t)
    
   

for i in [N-l for l in range(1,N)]: # temps
    print(i)
    for j in range(1,M): # prix 
        f.iloc[i,j] = a(j, delta_t, vol, r) * f.iloc[i+1,j-1] +b(j, delta_t, vol, r) * f.iloc[i+1,j] + c(j, delta_t, vol, r) * f.iloc[i+1,j+1]



# verdict, 80*delta_S <=> S0_target a 10 , on devrait trouver 0.46
pos_S0 = int(80*delta_S)
f.iloc[0, pos_S0]




# changement de variable Z =ln(S), hull page 484


K=11.
T =1.
N = 365 # on decoupe T en N parties , 1 partie = 1 jour du coup
delta_t = T /N
delta_Z = vol * numpy.sqrt(3*delta_t)

S0_target =  10.
M=200
delta_S = S0_target / 80
S_max = M*delta_S

def alpha(j, delta_t, vol, r):
    vol2 = vol * vol
    return (delta_t / (2*delta_Z)) * (r - vol2/2) - ((delta_t / (2* (delta_Z)**2))*vol2)
    
def beta(j, delta_t, vol, r):
    vol2 = vol * vol
    return 1 + r*delta_t + ((delta_t / ((delta_Z)**2))*vol2)

def gamma(j, delta_t, vol, r):
    vol2 =vol*vol
    return -(delta_t / (2*delta_Z)) * (r - vol2/2) - ((delta_t / (2* (delta_Z)**2))*vol2)

    
f = pandas.DataFrame(index=['%d * delta_t' % i for i in range(0,N+1)], 
                            columns=['%d * delta_S' % j for j in range(0,M+1)])
# a T, le valeur de l option vaut (ST -K)+
f.iloc[N,:] = numpy.maximum(numpy.array([j*delta_S for j in range(0, M+1)]) - K, 0)

# qd le prix de l action est nul, la valeur du cal est nulle
f.iloc[:,0] = 0

# qd S=Smax, on suppose que la valeur du call est 

f.iloc[:,-1] = S_max  - K * numpy.exp([-r * delta_t * (N-i) for i in range(0,N+1)])


    

for i in [N-l for l in range(1,N)]: # temps
    print(i)
    for j in range(1,M): # prix 
        f.iloc[i,j] = alpha(j, delta_t, vol, r) * f.iloc[i+1,j-1] +beta(j, delta_t, vol, r) * f.iloc[i+1,j] + gamma(j, delta_t, vol, r) * f.iloc[i+1,j+1]


pos_S0 = int(80*delta_S)
f.iloc[0, pos_S0]






# ================ 
# http://www.math.univ-paris13.fr/~mba/enseignement_fichiers/TP3_NumericalAnalysis.pdf

T=1.
r=0
S0=100
K=100
vol=0.05


# Schema explicite
M = 500 # nombres de pas en espace
N = 100 # nombres de pas de temps

S_max = 5 * K; # valeur maximum du sous-jacent

x_max = (log(S_max/S0)-(r-vol**2/2)*T)/sigma;
x_min = -xmax;


delta_x=(xmax-xmin)/(M+1)
delta_t=T/N

lambda_ = delta_t / (delta_x*delta_x);


/////////////////////////
// Matrices
/////////////////////////
disp(’Definition de la matrice A’);
// Matrices diagonales, diagonale inférieure, diagonale centrée,
4
// diagonale supérieure.
for i=1:M-1
A(i,i)=1-lambda;
A(i,i+1)=lambda/2;
A(i+1,i)=lambda/2;
end
A(M,M)=1-lambda;
// Condition initiale
x=(xmin+dx:dx:xmax-dx)’;
Unew=exp(-r*T)*max(0,S0*exp((r-sigma**2/2)*T+sigma*x)-K);
///////////////////////////////
// Boucle en temps
///////////////////////////////
disp(’Boucle en temps...’);
for i=1:M
Unew=A*Unew;
end
///////////////////////////////
// Affichage du prix
///////////////////////////////
i=floor(xmax/dx);
P=Unew(i);
// on peut interpoler pour avoir le prix en S0
disp(P,’prix calculé’)
// calcul du vrai prix
d1=(1/(sigma*sqrt(T)))*(log(S0/K)+(r+sigma**2/2)*T);
d2=d1-sigma*sqrt(T);
// case ’call’
  prix=S0*cdfnor("PQ",d1,0,1)-K*exp(-r*T)*cdfnor("PQ",d2,0,1);
// case ’put’
// prix=-(S0-K*exp(-r*T))+(S0*cdfnor("PQ",d1,0,1)-K*exp(-r*T)*cdfnor("PQ",d2,0,1));
5
disp(prix,’prix exact’);







