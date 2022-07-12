# Load in a number of libraries we will use
import numpy as np
import scipy.stats
import copy
# from transport_map_125 import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import time
import pickle
import os
import scipy

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close("all")

subsamp     = 10 # 10

def ksd(q,dq):
    
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    if q.shape != dq.shape:
        raise Exception("Shape of q and dq must be equal: "+str(q.shape)+"=/="+str(dq.shape))
    
    # Check if variables are matrices
    if len(q) < 2:
        
        # Append a dimension
        q   = q[:,np.newaxis]
        dq  = dq[:,np.newaxis]
    
    
    c2  = 1.
    
    def k(x, y, r, beta):
        beta    = beta**2
        return (c2 + r**2/beta)**(-0.5)

    def gradxk(x, y, r, beta):
        beta    = beta**2
        return (x - y) / beta * (c2 + r**2 / beta)**(-1.5) 

    def gradyk(x, y, r, beta):
        beta    = beta**2
        return -(x - y) / beta * (c2 + r**2 / beta)**(-1.5) 

    def gradxyk(x, y, r, beta):
        beta    = beta**2
        d       = (c2 + r**2 / beta)
        n       = len(x)
        return d**(-2.5)*( d*n / beta - 3.0/beta * np.inner(x-y,x-y) / beta )
    
    # Convert to vector
    n       = q.shape[0]
    
    # Create a matrix of pairwise distances
    dist    = pdist(q)
    
    # Compute beta
    beta    = np.median(dist)

    pairwise_dists = squareform(dist)

    # Calculate the discrepancy
    discr   = 0.
    for i in np.arange(0,n,1):
        for j in np.arange(i,n,1):
            
            # Get m
            if i == j:
                m   = 1
            else:
                m   = 2

            # Calculate 
            r   = pairwise_dists[i,j]
            
            sks = np.inner(dq[i,:],dq[j,:])*k(q[i,:], q[j,:], r, beta)
            sk  = np.inner(dq[i,:], gradxk(q[i,:], q[j,:], r, beta))
            ks  = np.inner(gradyk(q[i,:], q[j,:], r, beta), dq[j,:])
            trk = gradxyk(q[i], q[j], r, beta)
            
            discr += m * (sks + sk + ks + trk) / (n*(n-1.0))
            
    return discr

# Clear previous figures and pickles
root_directory = os.path.dirname(os.path.realpath(__file__))

# Collect the results
dct     = {}

# First, load the L63 results -------------------------------------------------
current_directory   = root_directory + "//" + "Structure L63"

os.chdir(current_directory)

dct['L63']  = {
    'X_a'   : copy.copy(pickle.load(open(current_directory + "//" + "Z_a_L63.p","rb"))),
    'X_f'   : copy.copy(pickle.load(open(current_directory + "//" + "Z_f_L63.p","rb")))}

# Store idx with worst KSD
idx_L63     = 0
maxval_L63  = 0.

for t in np.arange(-1,-101,-1):
    
    dct['L63'][t]   = {
        'ksds'          : copy.copy(pickle.load(open(current_directory + "//" + "ksd_L63_smoother_t="+str(t).zfill(3)+".p","rb"))),
        'ksds_marg'     : copy.copy(pickle.load(open(current_directory + "//" + "ksd_L63_smoother_marginal_t="+str(t).zfill(3)+".p","rb")))}
    
    if np.nanmean(dct['L63'][t]['ksds']) > maxval_L63:
        idx_L63     = t
        maxval_L63  = np.nanmean(dct['L63'][t]['ksds'])
    

# First, load the L96 results -------------------------------------------------
current_directory   = root_directory + "//" + "Structure L96"

os.chdir(current_directory)

dct['L96']  = {
    'X_a'   : copy.copy(pickle.load(open(current_directory + "//" + "Z_a_L96.p","rb"))),
    'X_f'   : copy.copy(pickle.load(open(current_directory + "//" + "Z_f_L96.p","rb")))}

# Store idx with worst KSD
idx_L96     = 0
maxval_L96  = 0.

for t in np.arange(-1,-101,-1):
    
    dct['L96'][t]   = {
        'ksds'          : copy.copy(pickle.load(open(current_directory + "//" + "ksd_L96_smoother_t="+str(t).zfill(3)+".p","rb"))),
        'ksds_marg'     : copy.copy(pickle.load(open(current_directory + "//" + "ksd_L96_smoother_marginal_t="+str(t).zfill(3)+".p","rb")))}
    
    if np.nanmean(dct['L96'][t]['ksds']) > maxval_L96:
        idx_L96     = t
        maxval_L96  = np.nanmean(dct['L96'][t]['ksds'])

os.chdir(root_directory)

# raise Exception

ksdmax  = np.maximum(np.nanmax(dct['L63'][idx_L63]['ksds']),np.nanmax(dct['L96'][idx_L96]['ksds']))


#%%

# We have found the time with the worst KSDs in the lower block. We want to plot
# the full distributions, though, so let us expand it:
    
# Lorenz 63 -------------------------------------------------------------------

# Augment the ksd array
dct['L63'][idx_L63]['ksds']     = np.row_stack((
    np.zeros(dct['L63'][idx_L63]['ksds'].shape)*np.nan,
    dct['L63'][idx_L63]['ksds']))

# Read the map input
map_input   = np.column_stack((
    copy.copy(dct['L63']['X_f'][idx_L63,...]),
    copy.copy(dct['L63']['X_a'][idx_L63-1,...]) ))

# Whiten the samples
map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
map_input   /= np.std(map_input,axis=0)[np.newaxis,:]

cov         = np.cov(map_input.T)
chol        = scipy.linalg.cholesky(cov)
prec        = scipy.linalg.inv(chol)
map_input   = np.dot(map_input,prec)

for row in range(3):
    for col in range(row):
        
        print('row: '+str(row).zfill(2)+' | col: '+str(col).zfill(2))
                
        samp    = np.column_stack((
            map_input[::subsamp,col],
            map_input[::subsamp,row]))
        
        dct['L63'][idx_L63]['ksds'][row,col] = ksd(
            q   = samp,
            dq  = -samp)
        
if np.nanmax(dct['L63'][idx_L63]['ksds']) > maxval_L63:
    maxval_L63  = np.nanmax(dct['L63'][idx_L63]['ksds'])
        
# Lorenz 96 -------------------------------------------------------------------

# Augment the ksd array
dct['L96'][idx_L96]['ksds']     = np.row_stack((
    np.zeros(dct['L96'][idx_L96]['ksds'].shape)*np.nan,
    dct['L96'][idx_L96]['ksds']))

# Read the map input
map_input   = np.column_stack((
    copy.copy(dct['L96']['X_f'][idx_L96,...]),
    copy.copy(dct['L96']['X_a'][idx_L96-1,...]) ))

# Whiten the samples
map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
map_input   /= np.std(map_input,axis=0)[np.newaxis,:]

cov         = np.cov(map_input.T)
chol        = scipy.linalg.cholesky(cov)
prec        = scipy.linalg.inv(chol)
map_input   = np.dot(map_input,prec)

for row in range(40):
    for col in range(row):
        
        print('row: '+str(row).zfill(2)+' | col: '+str(col).zfill(2))
                
        samp    = np.column_stack((
            map_input[::subsamp,col],
            map_input[::subsamp,row]))
        
        dct['L96'][idx_L96]['ksds'][row,col] = ksd(
            q   = samp,
            dq  = -samp)
        
if np.nanmax(dct['L96'][idx_L96]['ksds']) > maxval_L96:
    maxval_L96  = np.nanmax(dct['L96'][idx_L96]['ksds'])

# raise Exception







#%%

plt.figure(figsize = (12,7))

subgs  = GridSpec(
    nrows           = 2,
    ncols           = 2,
    hspace          = 0.2,
    height_ratios   = [0.05,1.])


#%%

# Plot colorbar

plt.subplot(subgs[0,:])

cmap = matplotlib.cm.get_cmap('turbo')
import matplotlib
norm = matplotlib.colors.Normalize(vmin=0, vmax=ksdmax)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    norm        = norm,
    orientation = 'horizontal')

# cb1.set_label("KSD to standard Gaussian", labelpad=-3)
cb1.set_label("KSD to standard Gaussian")

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

#%%

# Plot L63

gs  = subgs[1,0].subgridspec(
    nrows   = 6,
    ncols   = 6,
    hspace  = 0.,
    wspace  = 0.)

map_input   = np.column_stack((
    copy.copy(dct['L63']['X_f'][idx_L63,...]),
    copy.copy(dct['L63']['X_a'][idx_L63-1,...]) ))

map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
map_input   /= np.std(map_input,axis=0)[np.newaxis,:]

cov         = np.cov(map_input.T)
chol        = scipy.linalg.cholesky(cov)
prec        = scipy.linalg.inv(chol)
map_input   = np.dot(map_input,prec)

superscript_indices = ["a","b","c","a","b","c"]
subscript_indices   = ["t+1","t+1","t+1","t","t","t"]

for row in np.arange(0,6,1):
    for col in np.arange(0,row+1,1):
            
        print(str(row)+' | '+str(col))
        

        
        # # If we are in the upper block, whiten the color
        # if row < 3:
        #     color   = whiten(color)
        
        plt.subplot(gs[row,col])
        
        if row == 0 and col == 0:
            plt.title('$\mathbf{A}$: whitened smoothing distribution, lower block (Lorenz-63)',fontsize=10,loc="left")
            
        if row == col:
            
            # Get the color from KSD
            color   = cmap(dct['L63'][idx_L63]['ksds_marg'][col]/ksdmax)
            
            plt.hist(
                map_input[:,col],
                color   = color,
                range   = [-3,3],
                bins    = 21)
            
        else:
            
            print(dct['L63'][idx_L63]['ksds'][row,col]/ksdmax)
            color   = cmap(dct['L63'][idx_L63]['ksds'][row,col]/ksdmax)
            
            plt.scatter(
                map_input[:,col],
                map_input[:,row],
                s       = .6,
                lw      = 0.,
                color   = color)
            
            plt.xlim([-3,3])
            plt.ylim([-3,3])
            
        
        # if col == 0 and row == 5:
            
        #     # Hide the right and top spines
        #     plt.gca().spines['right'].set_visible(False)
        #     plt.gca().spines['top'].set_visible(False)
            
        #     # Only show ticks on the left and bottom spines
        #     plt.gca().yaxis.set_ticks_position('left')
        #     plt.gca().xaxis.set_ticks_position('bottom')
            
        #     plt.gca().set_xticks([""])
        #     plt.gca().set_xticklabels([""])
            
        #     plt.gca().set_yticks([""])
        #     plt.gca().set_yticklabels([""])
            
        #     plt.xlabel("$x_{t}^{a}$")
        #     plt.ylabel("$x_{t+1}^{c}$")
            
        # for 
        
        
        
        # if col < 3:
        #     time    = "t"
        # else:
        #     time    = "t+1"
        
        if col == 0 and row == 5:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.xlabel("$x_{"+subscript_indices[col]+"}^{"+str(superscript_indices[col])+"}$")
            plt.ylabel("$x_{"+subscript_indices[row]+"}^{"+str(superscript_indices[row])+"}$")
            
        elif col == 0:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.ylabel("$x_{"+subscript_indices[row]+"}^{"+str(superscript_indices[row])+"}$")
            
        elif row == 5:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.xlabel("$x_{"+subscript_indices[col]+"}^{"+str(superscript_indices[col])+"}$")
            
        else:
            
            plt.axis('off')
            
        
        
        
#%%

# Plot L96

# ax  = plt.subplot(subgs[1,1])
# plt.axis("off")

gs  = subgs[1,1].subgridspec(
    nrows   = 80,
    ncols   = 80,
    hspace  = 0.,
    wspace  = 0.)

map_input   = np.column_stack((
    copy.copy(dct['L96']['X_f'][idx_L96,...]),
    copy.copy(dct['L96']['X_a'][idx_L96-1,...]) ))

map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
map_input   /= np.std(map_input,axis=0)[np.newaxis,:]

cov         = np.cov(map_input.T)
chol        = scipy.linalg.cholesky(cov)
prec        = scipy.linalg.inv(chol)
map_input   = np.dot(map_input,prec)

for row in np.arange(0,80,1): #list(np.arange(59,70,1))+[79]: #np.arange(0,80,1): # 80
    for col in np.arange(0,row+1,1):#list(np.arange(19,30,1))+[79]: #np.arange(0,row+1,1):
            
        print(str(row)+' | '+str(col))
        
        
        plt.subplot(gs[row,col])
        
        if row == 0 and col == 0:
            plt.title('$\mathbf{B}$: whitened smoothing distribution, lower block (Lorenz-96)',fontsize=10,loc="left")
            
        if row == col:
            
            # Get the color from KSD
            color   = cmap(dct['L96'][idx_L96]['ksds_marg'][col]/ksdmax)
            
            plt.hist(
                map_input[:,col],
                color   = color,
                range   = [-3,3],
                bins    = 21)
            
        else:
            
            print(dct['L96'][idx_L96]['ksds'][row,col]/ksdmax)
            color   = cmap(dct['L96'][idx_L96]['ksds'][row,col]/ksdmax)
            
            plt.scatter(
                map_input[:,col],
                map_input[:,row],
                s       = .02,
                lw      = 0.,
                color   = color)
            
            plt.xlim([-3,3])
            plt.ylim([-3,3])
            
        plt.axis('off')
        
# %%
    
# Indices
rows    = np.arange(60,60+4,1)
cols    = np.arange(20,20+4,1)


window_small = np.asarray([
    [20-79-0.5,80-64-0.5],
    [24-79+0.5,80-64-0.5],
    [24-79+0.5,80-60+0.5],
    [20-79-0.5,80-60+0.5]])


# -----------------------------------------------------------------------------

# Inset 
inset  = gs[:35,45:].subgridspec(
    nrows   = 4,
    ncols   = 4,
    hspace  = 0.,
    wspace  = 0.)


superscript_indices = ["21","22","23","24","21","22","23","24"]
subscript_indices   = ["t+1","t+1","t+1","t+1","t","t","t","t"]

for row,rval in enumerate(rows):
    for col,cval in enumerate(cols):
        
        plt.subplot(inset[row,col])

        color   = cmap(dct['L96'][idx_L96]['ksds'][rval,cval]/ksdmax)
        
        plt.scatter(
            map_input[:,rval],
            map_input[:,cval],
            s       = .6,
            lw      = 0.,
            color   = color)
        
        plt.xlim([-3,3])
        plt.ylim([-3,3])
        
        if col == 0 and row == 3:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.xlabel("$x_{"+subscript_indices[col]+"}^{"+str(superscript_indices[col])+"}$")
            plt.ylabel("$x_{"+subscript_indices[row+4]+"}^{"+str(superscript_indices[row+4])+"}$")
            
        elif col == 0:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.ylabel("$x_{"+subscript_indices[row+4]+"}^{"+str(superscript_indices[row+4])+"}$")
            
        elif row == 3:
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            
            plt.xlabel("$x_{"+subscript_indices[col]+"}^{"+str(superscript_indices[col])+"}$")
            
        else:
            
            # plt.axis('off')
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])


plt.savefig('smoothing_comparison_triangular.png',dpi=600,bbox_inches='tight')
plt.savefig('smoothing_comparison_triangular.pdf',dpi=600,bbox_inches='tight')
