# This code generates the results figures for the Lorenz-96 experiments, which
# combines outputs of the EnTS and iEnKS simulations. To reproduce the figure,
# run these simulations first.
# Alternatively, please contact the corresponding author (Max Ramgraber). We 
# have the simulation results in storage, but could not upload them due to 
# a file size in excess of 10 GB.

# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
import copy
import scipy.optimize
import scipy.spatial
import pickle
import os

use_latex   = True

if use_latex:
    
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    titlesize   = 14
    labelsize   = 12
    addendum    = "_latex"
    
else:
    
    matplotlib.style.use('default')
    titlesize   = 12
    labelsize   = 10
    addendum    = ""

plt.close("all")

# Find the current path
root_directory = os.path.dirname(os.path.realpath(__file__))

# Get the turbo colormap
cmap        = plt.get_cmap('turbo') 


#%%

# =============================================================================
# Plot RMSE results
# =============================================================================

Ns              = [1000,750,500,375,250,175,100,50]
lambdas_filter  = [0.,0.25,0.5,1.0,1.5,2.0]
lambdas_smoother= [0.,0.025,0.05,0.1,0.15,0.2,0.25,0.5,1.0]
random_seeds    = [0,1,2,3,4,5,6,7,8,9]
orders          = [[1,1],[2,1],[2,2],[3,1],[3,3],[5,1],[5,5]]
gammas          = [0.,0.05,0.1,0.2,0.3]
T               = 1000

RMSEs   = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_smoother),len(orders)))*np.nan
RMSEs_f = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_filter),len(orders)))*np.nan

CRPSs   = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_smoother),len(orders)))*np.nan
CRPSs_f = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_filter),len(orders)))*np.nan

times   = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_smoother),len(orders)))*np.nan
times_f = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_filter),len(orders)))*np.nan

for ri,random_seed in enumerate(random_seeds):
        
    for gi,gamma in enumerate(gammas):

        current_directory = root_directory + "//" + "L63_RS_"+str(random_seed)
        
        os.chdir(current_directory)
        
        print(random_seed)
        
        for ni,N in enumerate(Ns):
            
            for oi,order in enumerate(orders):
                
                order_filter    = order[0]
                order_smoother  = order[1]
                    
                for li,lmbda in enumerate(lambdas_smoother):
                
                    # =============================================================
                    # Smoother
                    # =============================================================
                
                    # Get filename
                    filename    = 'TM_smoother_N='+str(N).zfill(4)+'_RS='+str(random_seed)+'_order_filter='+str(order_filter)+\
                        '_order_smoother='+str(order_smoother)+\
                        '_lambda='+str(lmbda)+'.p'
            
                    # Does filename exist?
                    if filename in list(os.listdir(current_directory)):
                
                        # Load dictionary
                        dct     = pickle.load(open(
                            filename,"rb"))
                        
                        # Check if RMSE list is complete
                        if len(dct["RMSE_list"]) == T:
                            
                            RMSEs[ri,ni,gi,li,oi] = np.mean(dct["RMSE_list"])
                            CRPSs[ri,ni,gi,li,oi] = np.mean(dct["CRPS_list"])
                            times[ri,ni,gi,li,oi] = dct["duration"]
                            
                    # else:
                        
                    #     raise Exception
                    
                for li,lmbda in enumerate(lambdas_filter):
                    
                            
                    # =============================================================
                    # Filter
                    # =============================================================
                    
                    # Get filename
                    filename    = 'TM_filter_N='+str(N).zfill(4)+'_RS='+str(random_seed)+\
                        '_order='+str(order_filter)+'_gamma='+str(gamma)+\
                        '_lambda='+str(lmbda)+'.p'
                        
                    # Does filename exist?
                    if filename in list(os.listdir(current_directory)):
                
                        # Load dictionary
                        dct     = pickle.load(open(
                            filename,"rb"))
                        
                        # Check if RMSE list is complete
                        if len(dct["RMSE_list"]) == T:
                            
                            RMSEs_f[ri,ni,gi,li,oi] = np.mean(dct["RMSE_list"])
                            CRPSs_f[ri,ni,gi,li,oi] = np.mean(dct["CRPS_list"])
                            times_f[ri,ni,gi,li,oi] = dct["duration"]
                                
                
os.chdir(root_directory)

# Average the times
# 1) Remove random seed
times   = np.nanmean(times,axis=0)
times_f = np.nanmean(times_f,axis=0)
# 2) Remove gammas
times   = np.nanmean(times,axis=1)
times_f = np.nanmean(times_f,axis=1)
# 3) Remove lambdas
times   = np.nanmean(times,axis=1)
times_f = np.nanmean(times_f,axis=1)
# # 2) Remove gammas
# times   = times[:,0,...]
# times_f = times_f[:,0,...]
# # 3) Remove lambdas
# times   = times[:,0,...]
# times_f = times_f[:,0,...]

colors1 = ['xkcd:orangish red','xkcd:cerulean','xkcd:grass green','xkcd:tangerine']
colors2 = ['xkcd:crimson','xkcd:cobalt','xkcd:pine','xkcd:deep orange']
colors2 = ['xkcd:tangerine','xkcd:sky blue','xkcd:pine','xkcd:deep orange']

plt.figure(figsize=(12,8))


plt.subplot(2,1,1)

Ns = np.flip(Ns)

ind = np.arange(len(Ns))

ax1 = plt.gca()
# ax2 = ax1.twinx()

ax1.bar(
    ind - 0.3,
    np.flip(times_f[:,0]),
    color = "xkcd:cerulean",
    label   = 'EnTF (1)',
    width = 0.15)

ax1.bar(
    ind - 0.1,
    np.flip(times_f[:,2]),
    color = "xkcd:grass green",
    label   = 'EnTF (2)',
    width = 0.15)

ax1.bar(
    ind + 0.1,
    np.flip(times_f[:,4]),
    color = "xkcd:goldenrod",
    label   = 'EnTF (3)',
    width = 0.15)

ax1.bar(
    ind + 0.3,
    np.flip(times_f[:,6]),
    color = "xkcd:orangish red",
    label   = 'EnTF (5)',
    width = 0.15)

ax1.set_yscale("log")

ax1.set_xticks(ind)
ax1.set_xticklabels([str(N) for N in Ns])
# ax1.set_xlabel("ensemble size", fontsize = labelsize)
ax1.set_ylabel("average runtime [s]", fontsize = labelsize)

plt.legend(frameon = False, fontsize = labelsize, loc = "upper left", ncols = 4)

plt.title("$\mathbf{A}$: computational demand for ensemble transport filters", fontsize = labelsize, loc = "left")

plt.subplot(2,1,2)

ind = np.arange(len(Ns))

ax1 = plt.gca()
# ax2 = ax1.twinx()

ax1.bar(
    ind - 0.3,
    np.flip(times[:,0]),
    color = "xkcd:cerulean",
    label   = 'EnTS (1)',
    width = 0.15)

ax1.bar(
    ind - 0.1,
    np.flip(times[:,2]),
    color = "xkcd:grass green",
    label   = 'EnTS (2)',
    width = 0.15)

ax1.bar(
    ind + 0.1,
    np.flip(times[:,4]),
    color = "xkcd:goldenrod",
    label   = 'EnTS (3)',
    width = 0.15)

ax1.bar(
    ind + 0.3,
    np.flip(times[:,6]),
    color = "xkcd:orangish red",
    label   = 'EnTS (5)',
    width = 0.15)

ax1.set_yscale("log")

ax1.set_xticks(ind)
ax1.set_xticklabels([str(N) for N in Ns])
ax1.set_xlabel("ensemble size", fontsize = labelsize)
ax1.set_ylabel("average runtime [s]", fontsize = labelsize)

plt.legend(frameon = False, fontsize = labelsize, loc = "upper left", ncols = 4)

plt.title("$\mathbf{B}$: computational demand for ensemble transport smoothers", fontsize = labelsize, loc = "left")


plt.savefig('L63_time_demand'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('L63_time_demand'+addendum+'.pdf',dpi=600,bbox_inches='tight')
        