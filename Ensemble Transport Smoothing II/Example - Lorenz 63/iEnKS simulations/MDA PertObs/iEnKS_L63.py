# This is an adapted version of a Lorenz-63 example provided in the Dapper 
# toolbox of Raanes et al (https://zenodo.org/record/2029296). 

# The toolbox is available under:
# https://github.com/nansencenter/DAPPER/blob/master/README.md

# I have slightly adapted the toolbox to store CRPS as a performance metric,
# and added a Hidden Markov Model specifications for our L63 and L96 setups.

import dapper as dpr
import dapper.da_methods as da
import numpy as np
import pickle

# Load the Lorenz-63 model
from spantini2019L63 import HMM
    
# Define the random seeds
seeds = 1+np.arange(10)

# Define the model parameters
params = dict(
    N = [5, 10, 25, 50, 100],
    infl = 1+np.array([0, .02, .04, .07, .1, .2]),
    Lag = [1,2,3,4,5,10,15,20,25,30],
)

# Create combinations for different parameters
for_params = dpr.combinator(params, seed=seeds)

# Create iEnKS simulations
xps = dpr.xpList()
xps += for_params(da.iEnKS, upd_a="PertObs", MDA=True) # Sqrt

# Start the simulation
save_as = xps.launch(HMM, __file__, mp=False, free=False)

# Load the results
xps = dpr.load_xps(save_as)

# Create an empty dictionary for the results
output_dict     = {}
# First layer: random seeds
for seed in seeds:   
    output_dict[seed]   = {}
    # Second layer: ensemble sizes
    for N in params['N']:    
        output_dict[seed][N]  = {}
        # Third layer: lag lengths
        for lag in params['Lag']:
            output_dict[seed][N][lag]  = {}
            # Fourth layer: inflation factors
            for infl in params['infl']:
                output_dict[seed][N][lag][infl]     = {}

# Go through all results, write them into the dictionary
for xp in xps:
    
    # Store it in the correct subdict
    output_dict[xp.seed][xp.N][xp.Lag][xp.infl]     = {
        'a'     : {                             # Filtering ensemble mean RMSEs
            'val'   : xp.avrgs.err.rms.a.val,
            'prec'  : xp.avrgs.err.rms.a.prec},
        's'     : {                             # Smoothing ensemble mean RMSEs
            'val'   : xp.avrgs.err.rms.s.val,
            'prec'  : xp.avrgs.err.rms.s.prec},
        'nIter'     : xp.avrgs.iters.val,       # Average number of iterations
        'crps'      : xp.crps}                  # Continuous rank probability score
    
    # Print the current experiment's parameters
    print(str(xp.seed)+' | '+str(xp.N)+' | '+str(xp.Lag)+' | '+str(xp.infl)+' | '+str(xp.avrgs.err.rms.s))
    
# Pickle the dictionary
pickle.dump(output_dict,open('iEnKS_L63_MDA_PertObs.p','wb'))