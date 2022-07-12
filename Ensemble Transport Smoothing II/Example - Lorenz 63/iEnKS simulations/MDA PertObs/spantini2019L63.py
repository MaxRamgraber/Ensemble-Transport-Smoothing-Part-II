"""Reproduce results from Table 1 `bib.sakov2012iterative`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz63 import LPs, Tplot, dstep_dx, step, x0

# tseq = modelling.Chronology(dt=0.05, dko=2, T=200., BurnIn=100., Tplot=Tplot)
# print(tseq)
tseq = modelling.Chronology(
    0.05,               # sub-step length
    dto     = 0.1,      # step length
    Ko      = 1250.,    # Total number of steps (1000 spinup + 250 assimilation)
    BurnIn  = 25.,      # Total length of spinup period (400/40/0.01 = 1000)
    Tplot   = Tplot)    # For plotting purposes

Nx = len(x0)

Dyn = {
    'M': Nx,
    'model': step,
    'linear': dstep_dx,
    'noise': 0,
}

X0 = modelling.GaussRV(C=2, mu=x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)
Obs['noise'] = 4  # modelling.GaussRV(C=CovMat(2*eye(Nx)))

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

HMM.liveplotters = LPs(jj)


####################
# Suggested tuning
####################
# from dapper.mods.Lorenz63.sakov2012 import HMM           # rmse.a:
# xps += Climatology()                                     # 7.6
# xps += OptInterp()                                       # 1.25
# xps += Var3D(xB=0.1)                                     # 1.04
# xps += ExtKF(infl=180)                                   # 0.92
# xps += EnKF('Sqrt',   N=3 ,  infl=1.30)                  # 0.80
# xps += EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.60
# xps += EnKF('PertObs',N=10,  infl=1.04)                  # 0.65
# xps += EnKF('PertObs',N=100, infl=1.01)                  # 0.56
# xps += EnKF_N(        N=3)                               # 0.60
# xps += EnKF_N(        N=10,            rot=True)         # 0.54
# xps += iEnKS('Sqrt',  N=10,  infl=1.02,rot=True)         # 0.31
# xps += PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
# xps += PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
# xps += PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
# xps += PFxN(xN=1000,  N=30  ,Qs=2   ,NER=0.2)            # 0.56
