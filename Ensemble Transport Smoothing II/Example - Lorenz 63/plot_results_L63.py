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

# Define a function which extracts the lower edge of the convex hull, to show
# only the best results
def get_edge(array):
    
    # Make a local copy of the array, to avoid pointer-weirdness
    array       = copy.copy(array)
    
    # To plot the best iEnKS results, find the convex hull
    ch          = scipy.spatial.ConvexHull(array)
    
    # Get the vertices
    ch_verts    = ch.vertices
    
    # Only keep the vertices on the lower side of the hull. Start by finding the 
    # vertex with the lowest x position
    vert_min    = np.where(array[:,0][ch_verts] == np.min(array[:,0][ch_verts]))[0][0]
    
    # Re-arrange the vertex list so this vertex is first
    ch_verts    = np.asarray(list(ch_verts[vert_min:]) + list(ch_verts[:vert_min]))
    
    # Find the differences between the indices
    xdiff       = np.diff(array[:,0][ch_verts])
    if np.min(xdiff) < 0:
        ch_verts    = ch_verts[:np.where(xdiff < 0)[0][0] + 1]
    
    # Reduce the array to these indices
    array       = array[ch_verts,:]
    
    return array

#%%

# =============================================================================
# Plot iEnKS results
# =============================================================================

# Load the iEnKS results

iEnKS_dct   = pickle.load(open("iEnKS_L63_MDA_PertObs.p","rb"))

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

# Get the colorlist
colorlist   = [cmap(x/np.max(lag_list)) for x in lag_list]
        
# Get the edge for the RMSEs and CRPSs
rmse_edge   = get_edge(np.column_stack((
    model_runs,
    rmse_list)))
crps_edge   = get_edge(np.column_stack((
    model_runs,
    crps_list)))

# Plot the full iEnKS

plt.figure(figsize=(12,12))

gs  = matplotlib.gridspec.GridSpec(
    nrows           = 3,
    ncols           = 1,
    hspace          = 0.3,
    height_ratios   = [0.1,2.,2.])

plt.subplot(gs[0])

cmap = matplotlib.cm.get_cmap('turbo')
import matplotlib
norm = matplotlib.colors.Normalize(vmin=np.min(lag_list), vmax=np.max(lag_list))
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    ticks       = lag_list,
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("iEnKS lag", fontsize = labelsize)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

plt.xticks(fontsize=labelsize)

plt.subplot(gs[1])

# plt.xlabel("model evaluations (log scale)")
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.gca().set_xscale('log')

# Plot the Lags
plt.scatter(
    model_runs,
    rmse_list,
    c       = colorlist,
    label   = "iEnKS (PertObs)",
    marker  = "x")

# Draw the iEnKS result
plt.plot(
    rmse_edge[:,0],
    rmse_edge[:,1],
    color   = "xkcd:grey",#[0.2,0.2,0.2],
    label   = "iEnKS (PertObs, best results)")


plt.title("$\mathbf{A}$: Lorenz-63 ensemble mean RMSE (iEnKS-PertObs)",loc="left", fontsize = titlesize)

plt.legend(frameon=False,ncol = 2, fontsize = labelsize, loc = "upper right")

plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)

os.chdir(root_directory)


#%%

# Load the iEnKS results
iEnKS_dct   = pickle.load(open("iEnKS_L63_MDA_sqrt.p","rb"))

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

# Get the colorlist
colorlist   = [cmap(x/np.max(lag_list)) for x in lag_list]

# Get the edge for the RMSEs and CRPSs
rmse_edge_sqrt   = get_edge(np.column_stack((
    model_runs,
    rmse_list)))
crps_edge_sqrt   = get_edge(np.column_stack((
    model_runs,
    crps_list)))

plt.subplot(gs[2])

# plt.xlabel("model evaluations (log scale)")
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.gca().set_xscale('log')

# Plot the Lags
plt.scatter(
    model_runs,
    rmse_list,
    c       = colorlist,
    label   = "iEnKS (Sqrt)",
    marker  = "x")

# Draw the iEnKS result
plt.plot(
    rmse_edge_sqrt[:,0],
    rmse_edge_sqrt[:,1],
    color   = "xkcd:grey",#[0.2,0.2,0.2],
    label   = "iEnKS (Sqrt, best results)")


plt.title("$\mathbf{C}$: Lorenz-63 ensemble mean RMSE (iEnKS-Sqrt)",loc="left", fontsize = titlesize)

plt.legend(frameon=False,ncol = 2, fontsize = labelsize, loc = "upper right")

plt.xlabel("model evaluations", fontsize = labelsize)

plt.xticks(fontsize=labelsize)
plt.yticks(fontsize=labelsize)

os.chdir(root_directory)

plt.savefig('results_L63_iEnKS'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('results_L63_iEnKS'+addendum+'.pdf',dpi=600,bbox_inches='tight')


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
                                
                
os.chdir(root_directory)

#%%
         
plt.figure(figsize=(12,8))

gs  = matplotlib.gridspec.GridSpec(
    nrows           = 2,
    ncols           = 1,
    hspace          = 0.3)

colors  = ['xkcd:cerulean','xkcd:grass green','xkcd:goldenrod','xkcd:orangish red','xkcd:orangish red']

plt.subplot(gs[0])

marker  = ["","x","+","v","","^"]
altmarker  = ["","x","*","o","","s"]

for oi,order in enumerate(orders): 
    
    RMSE_f  = copy.copy(RMSEs_f[:,:,:,:,oi])
    RMSE_f  = np.nanmean(RMSE_f,axis=0)
    
    RMSEval_f   = []
    
    for ni,N in enumerate(Ns):
        
        RMSEval_f.append(np.nanmin(RMSE_f[ni,...]))
        
    # # Mask all volatile RMSEs
    # RMSEval_f = [x if x < 0.6 else np.nan for x in RMSEval_f]
        
    if order[0] == order[1]:
        plt.plot(Ns,RMSEval_f,label = 'filter (order '+str(order[0])+')',marker  = marker[order[0]],color=colors[order[0]-1],ls=':')
               

    RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
    RMSE    = np.nanmean(RMSE,axis=0)
    
    RMSEval     = []
    
    for ni,N in enumerate(Ns):
        
        RMSEval.append(np.nanmin(RMSE[ni,...]))
        
    # Mask all volatile RMSEs
    RMSEval = [x if x < 0.6 else np.nan for x in RMSEval]
        
    if order[0] != order[1]:
        pass
        # plt.plot(Ns,RMSEval,label = 'smoother (order '+str(order)+')',color=colors[order[0]-1],linestyle='--',alpha = 0.5)
    else:
        plt.plot(Ns,RMSEval,label = 'smoother (order '+str(order[0])+')',marker  = marker[order[0]],color=colors[order[0]-1])
    
plt.ylim([0,0.65])

plt.xlabel("model evaluations", fontsize = labelsize)
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.xscale("log")
# plt.gca().set_xscale('log')
plt.gca().set_xticks(Ns, fontsize = labelsize),
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

# Fix the x axis limits
xlim    = plt.gca().get_xlim()
plt.gca().set_xlim(xlim)

# # Draw the iEnKS result
# plt.plot(
#     rmse_edge[:,0],
#     rmse_edge[:,1],
#     color   = "xkcd:silver",#[0.2,0.2,0.2],
#     label   = "iEnKS (PertObs)",
#     marker  = 'x',
#     zorder  = -1)
# plt.plot(
#     rmse_edge_sqrt[:,0],
#     rmse_edge_sqrt[:,1],
#     color   = "xkcd:grey",#[0.2,0.2,0.2],
#     label   = "iEnKS (Sqrt)",
#     marker  = 'x',
#     zorder  = -1)

# Plot the legend
plt.legend(frameon=False,ncol = 4, prop={"size":8}, fontsize = labelsize, loc = "upper right")

plt.title("$\mathbf{A}$: Lorenz-63 EnTF and EnTS results",loc="left", fontsize = titlesize)

plt.subplot(gs[1])

oi  = 6
order = [5,5]

RMSE_f  = copy.copy(RMSEs_f[:,:,:,:,oi])
RMSE_f  = np.nanmean(RMSE_f,axis=0)

RMSEval_f   = []

for ni,N in enumerate(Ns):
    
    RMSEval_f.append(np.nanmin(RMSE_f[ni,...]))
    
# Mask all volatile RMSEs
RMSEval_f = [x if x < 0.6 else np.nan for x in RMSEval_f]
    
if order[0] == order[1]:
    plt.plot(Ns,RMSEval_f,label = 'EnTF (order '+str(order[0])+')',marker  = marker[order[0]],color=colors[order[0]-1],ls=':')

RMSE    = copy.copy(RMSEs[:,:,:,:,oi])
RMSE    = np.nanmean(RMSE,axis=0)

RMSEval     = []

for ni,N in enumerate(Ns):
    
    RMSEval.append(np.nanmin(RMSE[ni,...]))
    
# Mask all volatile RMSEs
RMSEval = [x if x < 0.6 else np.nan for x in RMSEval]
    
if order[0] != order[1]:
    pass
    # plt.plot(Ns,RMSEval,label = 'smoother (order '+str(order)+')',color=colors[order[0]-1],linestyle='--',alpha = 0.5)
else:
    plt.plot(Ns,RMSEval,label = 'EnTS (order '+str(order[0])+')',marker  = marker[order[0]],color=colors[order[0]-1])
    
plt.ylim([0,0.65])

plt.xlabel("model evaluations", fontsize = labelsize)
plt.ylabel("time-averaged ensemble mean RMSE", fontsize = labelsize)

# plt.xscale("log")
# plt.gca().set_xscale('log')
plt.gca().set_xticks(Ns, fontsize = labelsize),
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize=labelsize)

# Fix the x axis limits
xlim    = plt.gca().get_xlim()
plt.gca().set_xlim(xlim)

# Draw the iEnKS result
plt.plot(
    rmse_edge[:,0],
    rmse_edge[:,1],
    color   = "xkcd:silver",#[0.2,0.2,0.2],
    label   = "iEnKS (PertObs)",
    marker  = 'x',
    zorder  = -1)
plt.plot(
    rmse_edge_sqrt[:,0],
    rmse_edge_sqrt[:,1],
    color   = "xkcd:grey",#[0.2,0.2,0.2],
    label   = "iEnKS (Sqrt)",
    marker  = '+',
    zorder  = -1)

# Plot the legend
plt.legend(frameon=False,ncol = 2, prop={"size":8}, fontsize = labelsize, loc = "upper right")

plt.title("$\mathbf{B}$: Lorenz-63 iEnKS results",loc="left", fontsize = titlesize)

os.chdir(root_directory)

plt.savefig('results_L63'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('results_L63'+addendum+'.pdf',dpi=600,bbox_inches='tight')