# See examples/basic_3a.py for comments
import numpy as np
import dapper as dpr
import dapper.da_methods as da
import pickle

# Setup
from spantini2019L96 import HMM

seeds = 1+np.arange(10)

params = dict(
    N = [5, 10, 25, 50, 100],
    infl = 1+np.array([0, .02, .04, .07, .1, .2]),
    Lag = [1, 2, 3, 4, 5],
)
for_params = dpr.combinator(params, seed=seeds)

xps = dpr.xpList()
xps += for_params(da.iEnKS, upd_a="PertObs",MDA=True) # Sqrt

# YOU PROBABLY WANT TO SET mp=True
save_as = xps.launch(HMM, __file__, mp=False, free=False)

# Results
xps = dpr.load_xps(save_as)

# Store the results in a dictionary
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
        'a'     : {
            'val'   : xp.avrgs.err.rms.a.val,
            'prec'  : xp.avrgs.err.rms.a.prec},
        's'     : {
            'val'   : xp.avrgs.err.rms.s.val,
            'prec'  : xp.avrgs.err.rms.s.prec},
        'nIter'     : xp.avrgs.iters.val,
        'crps'      : xp.crps}

pickle.dump(output_dict,open('iEnKS_L96_MDA_PertObs.p','wb'))