# This code generates the results figures for the Lorenz-96 experiments, which
# combines outputs of the EnTS and iEnKS simulations. To reproduce the figure,
# run these simulations first.
# Alternatively, please contact the corresponding author (Max Ramgraber). We 
# have the simulation results in storage, but could not upload them due to 
# a file size in excess of 10 GB.

# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import copy
import scipy.optimize
import pickle
import os
import matplotlib
from matplotlib import gridspec

use_latex   = False

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
# Load the L96 iEnKS (PertObs) results
# =============================================================================

iEnKS_dct   = pickle.load(open("iEnKS_L96_MDA_PertObs.p","rb"))

# Get all the keys
seeds_iEnKS = list(iEnKS_dct.keys())
Ns_iEnKS    = list(iEnKS_dct[seeds_iEnKS[0]].keys())
Lags_iEnKS  = list(iEnKS_dct[seeds_iEnKS[0]][Ns_iEnKS[0]].keys())
infls_iEnKS = list(iEnKS_dct[seeds_iEnKS[0]][Ns_iEnKS[0]][Lags_iEnKS[0]].keys())
            
# Create matrices for the outputs
rmses       = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan
crps        = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan
iters       = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan

# Extract the results
for si,seed in enumerate(seeds_iEnKS):
    
    for ni,N in enumerate(Ns_iEnKS):
        
        for li,Lag in enumerate(Lags_iEnKS):
            
            for ii,infl in enumerate(infls_iEnKS):
                
                rmses[si,ni,li,ii]      = iEnKS_dct[seed][N][Lag][infl]['s']['val']
                crps[si,ni,li,ii]       = np.mean(np.asarray(iEnKS_dct[seed][N][Lag][infl]['crps']))
                iters[si,ni,li,ii]      = iEnKS_dct[seed][N][Lag][infl]['nIter']
                
# Average these results
rmses       = np.nanmean(rmses,  axis = 0)
crps        = np.nanmean(crps,   axis = 0)
iters       = np.nanmean(iters,  axis = 0)

# Prepare for a scatter plot
model_runs  = []
rmse_list   = []
crps_list   = []
lag_list    = []

for ni,N in enumerate(Ns_iEnKS):
    for li,Lag in enumerate(Lags_iEnKS):
        
        # Find the index where the RMSE is lowest
        minindx     = np.where(rmses[ni,li,:] == np.nanmin(rmses[ni,li,:]))[0][0]
        
        # Append the required model runs, rmse, crps, and lag
        model_runs  .append(int(N * (iters[ni,li,minindx]*Lag + 1)))
        rmse_list   .append(np.nanmin(rmses[ni,li,:]))
        crps_list   .append(np.nanmin(crps[ni,li,:]))
        lag_list    .append(Lag)
        
# Convert those lists to arrays
model_runs  = np.asarray(model_runs)
rmse_list   = np.asarray(rmse_list)
crps_list   = np.asarray(crps_list)

rmses_pertobs   = np.min(rmses,axis=-1)
crps_pertobs    = np.min(crps,axis=-1)

os.chdir(root_directory)

#%%

# =============================================================================
# Load the L96 iEnKS (Sqrt) results
# =============================================================================

# Load the iEnKS results
iEnKS_dct   = pickle.load(open("iEnKS_L96_MDA_Sqrt.p","rb"))

# Get all the keys
seeds_iEnKS = list(iEnKS_dct.keys())
Ns_iEnKS    = list(iEnKS_dct[seeds_iEnKS[0]].keys())
Lags_iEnKS  = list(iEnKS_dct[seeds_iEnKS[0]][Ns_iEnKS[0]].keys())
infls_iEnKS = list(iEnKS_dct[seeds_iEnKS[0]][Ns_iEnKS[0]][Lags_iEnKS[0]].keys())
            
# Create matrices for the outputs
rmses       = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan
crps        = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan
iters       = np.zeros((len(seeds_iEnKS),len(Ns_iEnKS),len(Lags_iEnKS),len(infls_iEnKS)))*np.nan

# Extract the results
for si,seed in enumerate(seeds_iEnKS):
    
    for ni,N in enumerate(Ns_iEnKS):
        
        for li,Lag in enumerate(Lags_iEnKS):
            
            for ii,infl in enumerate(infls_iEnKS):
                
                rmses[si,ni,li,ii]      = iEnKS_dct[seed][N][Lag][infl]['s']['val']
                crps[si,ni,li,ii]       = np.mean(np.asarray(iEnKS_dct[seed][N][Lag][infl]['crps']))
                iters[si,ni,li,ii]      = iEnKS_dct[seed][N][Lag][infl]['nIter']
                
# Average these results
rmses       = np.nanmean(rmses,  axis = 0)
crps        = np.nanmean(crps,   axis = 0)
iters       = np.nanmean(iters,  axis = 0)

rmses_sqrt  = np.min(rmses,axis=-1)
crps_sqrt   = np.min(crps,axis=-1)

# Prepare for a scatter plot
model_runs  = []
rmse_list   = []
crps_list   = []
lag_list    = []

for ni,N in enumerate(Ns_iEnKS):
    for li,Lag in enumerate(Lags_iEnKS):
        
        # Find the index where the RMSE is lowest
        minindx     = np.where(rmses[ni,li,:] == np.nanmin(rmses[ni,li,:]))[0][0]
        
        # Append the required model runs, rmse, crps, and lag
        model_runs  .append(int(N * (iters[ni,li,minindx]*Lag + 1)))
        rmse_list   .append(np.nanmin(rmses[ni,li,:]))
        crps_list   .append(np.nanmin(crps[ni,li,:]))
        lag_list    .append(Lag)
        
# Convert those lists to arrays
model_runs  = np.asarray(model_runs)
rmse_list   = np.asarray(rmse_list)
crps_list   = np.asarray(crps_list)

# Get the colorlist
colorlist   = [cmap((x-np.min(lag_list))/np.max(lag_list)) for x in lag_list]

os.chdir(root_directory)

#%%

Ns                  = [50,100,135,175,250,375,500]#[100,150,200,300,400,600,800,1000]
lambdas_filter      = [0.,1.,3.,7.,10.,15.]
lambdas_smoother    = [0.,1.,3.,7.,10.,15.]
random_seeds        = [0,1,2,3,4,5,6,7,8,9]
orders              = [[1,1],[2,1],[2,2],[3,1],[3,3],[5,1],[5,5]]
gammas              = [0.,0.025,0.05,0.1,0.2]
T                   = 250

# Save the computational times
times_EnTF      = {}
times_EnTS      = {}

RMSEs   = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_smoother),len(orders)))*np.nan
RMSEs_f = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_filter),len(orders)))*np.nan

CRPSs   = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_smoother),len(orders)))*np.nan
CRPSs_f = np.zeros((len(random_seeds),len(Ns),len(gammas),len(lambdas_filter),len(orders)))*np.nan

for ri,random_seed in enumerate(random_seeds):
        
    for gi,gamma in enumerate(gammas):

        current_directory = root_directory + "//" + "L96_RS_"+str(random_seed)
        
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
                        '_order_smoother='+str(order_smoother)+'_lambda='+str(lmbda)+'.p'
            
                    # Does filename exist?
                    if filename in list(os.listdir(current_directory)):
                        
                        print(filename)
                
                        # Load dictionary
                        dct     = pickle.load(open(
                            filename,"rb"))

                        RMSEs[ri,ni,gi,li,oi] = np.mean(dct["RMSE_list"])
                        CRPSs[ri,ni,gi,li,oi] = np.mean(np.asarray(dct["CRPS_list"][1:]))
                        
                        if N not in list(times_EnTS.keys()):
                            times_EnTS[N] = {}
                            
                        if str(order) not in list(times_EnTS[N].keys()):
                            times_EnTS[N][str(order)] = []
                            
                        times_EnTS[N][str(order)].append(dct['duration'])

                for li,lmbda in enumerate(lambdas_filter):
                            
                    # =============================================================
                    # Filter
                    # =============================================================
                    
                    # Get filename
                    filename    = 'TM_filter_N='+str(N).zfill(4)+'_RS='+str(random_seed)+\
                        '_order='+str(order_filter)+'_gamma='+str(gamma)+'_lambda='+str(lmbda)+'.p'
                
                    # Does filename exist?
                    if filename in list(os.listdir(current_directory)):
                
                        # Load dictionary
                        dct     = pickle.load(open(
                            filename,"rb"))
                        
                        RMSEs_f[ri,ni,gi,li,oi] = np.mean(dct["RMSE_list"])
                        CRPSs_f[ri,ni,gi,li,oi] = np.mean(dct["CRPS_list"][1:])
                            
                        if N not in list(times_EnTF.keys()):
                            times_EnTF[N] = {}
                            
                        if str(order) not in list(times_EnTF[N].keys()):
                            times_EnTF[N][str(order)] = []
                            
                        times_EnTF[N][str(order)].append(dct['duration'])
                        

# How long did the EnTF and EnTS take?
print('EnTF ----------------')
for N in Ns:
    print('N='+str(N)+' | '+str(np.mean(np.asarray(times_EnTF[N]['[5, 5]']))/np.mean(np.asarray(times_EnTF[N]['[1, 1]']))))
print('EnTS ----------------')
for N in Ns:
    print('N='+str(N)+' | '+str(np.mean(np.asarray(times_EnTS[N]['[5, 5]']))/np.mean(np.asarray(times_EnTS[N]['[1, 1]']))))
    
plt.figure(figsize=(10,8))

gs  = gridspec.GridSpec(
    nrows           = 2, 
    ncols           = 2, 
    hspace          = 0.25,
    width_ratios    = [5,1])#, wspace = 0.1, hspace = 0.1)

colors  = ['xkcd:cerulean','xkcd:grass green','xkcd:goldenrod','xkcd:orangish red']

plt.subplot(gs[0,0])

plt.title("$\mathbf{A}$: filters versus smoothers",loc="left", fontsize = titlesize)

marker  = ["","x","+","v","","^"]
altmarker  = ["","x","*","o","","s"]

for oi,order in enumerate(orders): 
    
    if order[0] == order[1] and order[0] < 3:
    
        RMSE_f  = copy.copy(RMSEs_f[:,:,:,:,oi])
        RMSE_f  = np.nanmean(RMSE_f,axis=0)
        
        RMSEval_f   = []
        
        for ni,N in enumerate(Ns):
            
            RMSEval_f.append(np.nanmin(RMSE_f[ni,...]))
                
        if order[0] <= 3:
            colororder = order[0]-1
        else:
            colororder = 3
            
        plt.plot(Ns,RMSEval_f, marker = marker[order[0]],label = 'filter (order '+str(order)+')',color = colors[colororder],ls='--')
                   
        RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
        RMSE    = np.nanmean(RMSE,axis=0)
        
        RMSEval     = []
        
        for ni,N in enumerate(Ns):
            
            RMSEval.append(np.nanmin(RMSE[ni,...]))
            
            print('order '+str(order)+' N '+str(N)+ ' '+ str(np.where(RMSE[ni,...] == RMSEval[-1])))
            
        if order[0] == order[1]:
            plt.plot(Ns,RMSEval, marker = marker[order[0]],label = 'smoother (order '+str(order)+')',color = colors[colororder])
        else:
            plt.plot(Ns,RMSEval, marker = altmarker[order[0]],label = 'smoother (order '+str(order)+')',color = colors[colororder],alpha = 0.5)

# plt.xlabel("ensemble size")
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')
plt.gca().set_xticks(Ns, fontsize = labelsize)
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

#%%



plt.subplot(gs[1,0])

plt.title("$\mathbf{B}$: linear versus nonlinear smoothers",loc="left", fontsize = titlesize)

marker  = ["","x","+","v","","^"]
altmarker  = ["","x","*","o","","s"]

for oi,order in enumerate(orders): 
    
    if order[0] <= 3:
        colororder = order[0]-1
    else:
        colororder = 3
    
    RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
    RMSE    = np.nanmean(RMSE,axis=0)
    
    RMSEval     = []
    
    for ni,N in enumerate(Ns):
        
        RMSEval.append(np.nanmin(RMSE[ni,...]))
        
        print('order '+str(order)+' N '+str(N)+ ' '+ str(np.where(RMSE[ni,...] == RMSEval[-1])))
        
    if order[0] == order[1]:
        plt.plot(Ns,RMSEval, marker = marker[order[0]],label = 'smoother ('+str(order)+' )',color = colors[colororder])
    else:
        plt.plot(Ns,RMSEval, marker = altmarker[order[0]],label = 'smoother (order '+str(order)+')',color = colors[colororder],alpha = 0.5)
    
plt.xlabel("model evaluations", fontsize = labelsize)
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')
plt.gca().set_xticks(Ns, fontsize = labelsize)
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

# plt.yscale('log')

#%%

# Create legend

from matplotlib.lines import Line2D

gs2 = gridspec.GridSpecFromSubplotSpec(
    nrows           = 4,
    ncols           = 1,
    wspace          = 0.0, 
    hspace          = 0.0,
    subplot_spec    = gs[:,1])

plt.subplot(gs2[0,0])

legend_elements = [
    Line2D([0], [0], color='xkcd:cerulean', marker = 'x', label=r'EnTF (1)', ls='--'),
    Line2D([0], [0], color='xkcd:cerulean', marker = 'x', label=r'EnTS (1 - 1)')]

plt.legend(handles=legend_elements, loc='center left',frameon=False, fontsize = labelsize)

plt.axis('off')


plt.subplot(gs2[1,0])


legend_elements = [
    Line2D([0], [0], color='xkcd:grass green', marker = '+', label=r'EnTF (2)', ls='--'),
    Line2D([0], [0], color='xkcd:grass green', marker = '+', label=r'EnTS (2 - 2)'),
    Line2D([0], [0], color='xkcd:grass green', marker = '*', label=r'EnTS (2 - 1)', alpha = 0.5)]

plt.legend(handles=legend_elements, loc='center left',frameon=False, fontsize = labelsize)

plt.axis('off')


plt.subplot(gs2[2,0])

legend_elements = [
    Line2D([0], [0], color='xkcd:goldenrod', marker = 'v', label=r'EnTF (3)', ls='--'),
    Line2D([0], [0], color='xkcd:goldenrod', marker = 'v', label=r'EnTS (3 - 3)'),
    Line2D([0], [0], color='xkcd:goldenrod', marker = 'o', label=r'EnTS (3 - 1)', alpha = 0.5)]

plt.legend(handles=legend_elements, loc='center left',frameon=False, fontsize = labelsize)

plt.axis('off')


plt.subplot(gs2[3,0])

legend_elements = [
    Line2D([0], [0], color='xkcd:orangish red', label=r'EnTF (5)', marker = '^', ls='--'),
    Line2D([0], [0], color='xkcd:orangish red', label=r'EnTS (5 - 5)', marker = '^'),
    Line2D([0], [0], color='xkcd:orangish red', label=r'EnTS (5 - 1)', marker = 's', alpha = 0.5)]

plt.legend(handles=legend_elements, loc='center left',frameon=False, fontsize = labelsize)


plt.axis('off')

os.chdir(root_directory)

plt.savefig('results_L96'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('results_L96'+addendum+'.pdf',dpi=600,bbox_inches='tight')

#%%

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:cerulean",
      "xkcd:light sky blue"])

cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:grass green",
     "xkcd:light yellow green"])
                        
plt.figure(figsize=(14,10))

gs  = gridspec.GridSpec(
    nrows           = 2, 
    ncols           = 1, 
    hspace          = 0.5,
    height_ratios   = [5,5])

gs_sub = gs[0,:].subgridspec(
    nrows   = 2,
    ncols   = 1,
    height_ratios   = [1,10],
    hspace  = 0.,
    wspace  = 0.)

plt.subplot(gs_sub[0,:])

bounds = [1,2,3,4,5,6]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("iEnKS (PertObs) lag")

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticks([1.5,2.5,3.5,4.5,5.5], fontsize = labelsize)  # horizontal colorbar
cb1.ax.set_xticklabels([1,2,3,4,5], fontsize = labelsize)  # horizontal colorbar


plt.title("$\mathbf{A}$: Lorenz-96 ensemble mean RMSE (iEnKS-PertObs)",loc="left", fontsize = titlesize)

plt.subplot(gs_sub[1,:])

RMSE_f  = copy.copy(RMSEs_f[:,:,:,:,-1])
RMSE_f  = np.nanmean(RMSE_f,axis=0)

RMSEval_f   = []

oi      = 4
order   = [5,5]

for ni,N in enumerate(Ns):
    
    RMSEval_f.append(np.nanmin(RMSE_f[ni,...]))

if order[0] <= 3:
    colororder = order[0]-1
else:
    colororder = 3
    
plt.plot(
    Ns,
    RMSEval_f,
    label = 'EnTF (5)',
    marker = '^',
    color = 'xkcd:orangish red',
    ls='--')

RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
RMSE    = np.nanmean(RMSE,axis=0)

RMSEval     = []

for ni,N in enumerate(Ns):
    
    RMSEval.append(np.nanmin(RMSE[ni,...]))
    
    print('order '+str(order)+' N '+str(N)+ ' '+ str(np.where(RMSE[ni,...] == RMSEval[-1])))
    
plt.plot(
    Ns,
    RMSEval,
    label='EnTS (5-5)',
    marker = '^',
    color = 'xkcd:orangish red')

plt.legend(frameon=False,loc='upper right', fontsize = labelsize)

plt.ylabel("time-averaged ensemble mean RMSE")

plt.gca().set_xticks(Ns, fontsize = labelsize)
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

xlim    = plt.gca().get_xlim()
ylim    = plt.gca().get_ylim()

# Draw the iEnKS result
for li,lag in enumerate(Lags_iEnKS):
    
    plt.plot(
        np.asarray(Ns_iEnKS)*lag*10,
        rmses_pertobs[:,li],
        color   = cmap(li/(len(Lags_iEnKS)-1)),#"xkcd:silver",#[0.2,0.2,0.2],
        marker = 'x',
        zorder  = -1)
    
plt.gca().set_xticks([50,500,1000,2000,3000,4000,5000], fontsize = labelsize)
plt.gca().set_xticklabels([50,500,1000,2000,3000,4000,5000], fontsize = labelsize)
plt.yticks(fontsize=labelsize)

plt.xlabel('model evaluations', fontsize = labelsize)

#%%

gs_sub2 = gs[1,:].subgridspec(
    nrows   = 2,
    ncols   = 1,
    height_ratios   = [1,10],
    hspace  = 0.,
    wspace  = 0.)

plt.subplot(gs_sub2[0,:])

bounds = [1,2,3,4,5,6]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap2,
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("iEnKS (Sqrt) lag")

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticks([1.5,2.5,3.5,4.5,5.5], fontsize = labelsize)  # horizontal colorbar
cb1.ax.set_xticklabels([1,2,3,4,5], fontsize = labelsize)  # horizontal colorbar

plt.title("$\mathbf{B}$: Lorenz-96 ensemble mean RMSE (iEnKS-Sqrt)",loc="left", fontsize = titlesize)


#%%

plt.subplot(gs_sub2[1,:])

RMSE_f  = copy.copy(RMSEs_f[:,:,:,:,-1])
RMSE_f  = np.nanmean(RMSE_f,axis=0)

RMSEval_f   = []

oi      = 4
order   = [5,5]

for ni,N in enumerate(Ns):
    
    RMSEval_f.append(np.nanmin(RMSE_f[ni,...]))

if order[0] <= 3:
    colororder = order[0]-1
else:
    colororder = 3
    
plt.plot(
    Ns,
    RMSEval_f,
    label = 'EnTF (5)',
    marker = '^',
    color = 'xkcd:orangish red',
    ls='--')
           
RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
RMSE    = np.nanmean(RMSE,axis=0)

RMSEval     = []

for ni,N in enumerate(Ns):
    
    RMSEval.append(np.nanmin(RMSE[ni,...]))
    
    print('order '+str(order)+' N '+str(N)+ ' '+ str(np.where(RMSE[ni,...] == RMSEval[-1])))
    
plt.plot(
    Ns,
    RMSEval,
    label='EnTS (5-5)',
    marker = '^',
    color = 'xkcd:orangish red')

plt.legend(frameon=False,loc='upper right', fontsize = labelsize)

plt.ylabel("time-averaged ensemble mean RMSE")

plt.gca().set_xticks(Ns, fontsize = labelsize)
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

xlim    = plt.gca().get_xlim()
ylim    = plt.gca().get_ylim()

# Draw the iEnKS result
for li,lag in enumerate(Lags_iEnKS):
    
    plt.plot(
        np.asarray(Ns_iEnKS)*lag*10,
        rmses_sqrt[:,li],
        color   = cmap2(li/(len(Lags_iEnKS)-1)),
        marker = 'x',
        zorder  = -1)
    
    
plt.gca().set_xticks([50,500,1000,2000,3000,4000,5000], fontsize = labelsize)
plt.gca().set_xticklabels([50,500,1000,2000,3000,4000,5000], fontsize = labelsize)
plt.yticks(fontsize=labelsize)

plt.xlabel('model evaluations', fontsize = labelsize)

#%%

os.chdir(root_directory)

plt.savefig('results_L96_EnTS_and_iEnKS'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('results_L96_EnTS_and_iEnKS'+addendum+'.pdf',dpi=600,bbox_inches='tight')