from plotter import plot
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from plik_lite import Planck
import emcee
from chainconsumer import ChainConsumer
from metropolis import MetropolisHastings

DATA_DIR = Path(os.environ["NeuralBoltzmann_DATA"])
MODEL_DIR = Path(os.environ["NeuralBoltzmann_MODEL"])

# Define the likelihood
pl = Planck()
lnL = pl.plik_lite

# define parameter ranges
paramranges = np.array([ [1.9, 2.3], [.94, .98], [0.03, 0.07], [0.020, 0.024], [0.115, .125], [1.039, 1.043], [0.99, 1.01]  ])

# Initialize the sampler 
sampler = MetropolisHastings( lnL, paramranges)

# define a covariance matrix to speed up sampling

c = np.loadtxt('../clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt')[:,[37,7,5,2,3,51,8]]
cov = np.cov(c.T)

print( c.mean(axis=0) )
# begin sampling
num_samples= 50_000
samples = sampler.MH( num_samples,cov=cov, update_freq= 1000 )
