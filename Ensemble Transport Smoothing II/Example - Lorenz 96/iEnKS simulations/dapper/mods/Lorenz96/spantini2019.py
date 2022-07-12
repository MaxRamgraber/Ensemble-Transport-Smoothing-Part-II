"""
Comment by Maximilian Ramgraber: I adopted this file from the Sakov L96 setup.

Settings as in `bib.sakov2008deterministic`.

This HMM is used (with small variations) in many DA papers, for example

`bib.bocquet2011ensemble`, `bib.sakov2012iterative`,
`bib.bocquet2015expanding`, `bib.bocquet2013joint`.
"""

import numpy as np
import dapper.mods as modelling
from dapper.mods.Lorenz96 import LPs, Tplot, dstep_dx, step, x0
from dapper.tools.localization import nd_Id_localization

# As far as I understand, this sets up the time discretization. In my simulations,
# I run Lorenz-96 starting from a standard Gaussian prior over a 1000 time step
# burn-in period (with the EnKF), then apply the proper smoother over a length
# of 250 time steps.
# I assimilate observations every 0.4 time units, and I integrate the dynamics
# between with 40 substeps in a fourth-order Runge-Kutta scheme with a sub-step
# length of 0.01. I think the following setup reflects that:
tseq = modelling.Chronology(
    0.01,               # sub-step length
    dto     = .4,       # step length
    Ko      = 350.,     # Total number of steps (1000 spinup + 250 assimilation)
    BurnIn  = 40.,      # Total length of spinup period (400/40/0.01 = 1000)
    Tplot   = Tplot)    # For plotting purposes

# I assume Nx specifies the (otherwise dynamic) dimensionality of Lorenz-96
Nx = 40

# I assume x0 specifies the initial condition
x0 = x0(Nx)     # This seems to create a 40-dimensional vector of zeros except a 1 in the first entry
X0 = modelling.GaussRV(
    mu  = x0,   # The x0 vector becomes the mean of the prior
    C   = 1.0)  # With identity covariance; quite similar to my setup

# This dictionary likely specifies the dynamics
Dyn = {
    'M': Nx,            # Dimensionality of the system: 40
    'model': step,      # This seems like a fourth-order Runge-Kutta scheme
    'linear': dstep_dx, # Something related to the time integration
    'noise': 0,         # A deterministic forecast without noise, I presume
}

# Here, we specify the observation model
# In my setup, I assume we only observe every second state
jj = np.arange(0,Nx,2)  # This, I presume, are the observed dimension indices: [0,2,4,...,36,38]
Obs = modelling.partial_Id_Obs(Nx, jj) # Nx: length of state vector | jj: observed indices
Obs['noise'] = np.sqrt(0.5) # The observation error has a standard deviation of sqrt(0.5)

# This might be special, and seems to specify the localization in the update. 
# I am confident I understand how it works, so I have not touched it, yet. 
Obs['localizer'] = nd_Id_localization((Nx,), (2,))

# Combine everything we have specified so far into a Hidden Markov Model:
# Dynamics, Observation model, time discretization, and initial condition.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

# I assume this relates to plotting; I haven't touched it.
HMM.liveplotters = LPs(jj)
