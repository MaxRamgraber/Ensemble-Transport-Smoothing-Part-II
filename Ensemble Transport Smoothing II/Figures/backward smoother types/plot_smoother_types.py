import numpy as np
import matplotlib
import matplotlib.pyplot as plt

use_latex   = True

if use_latex:
    
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    titlesize   = 14
    labelsize   = 12
    addendum    = "_latex"
    pad         = -20
    
else:
    
    titlesize   = 12
    labelsize   = 10
    addendum    = ""
    pad         = -25

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:midnight blue",
     "xkcd:sky blue",
     "xkcd:light sky blue"])

plt.close('all')

Nnodes  = 5

plt.figure(figsize=(12,4.5))
gs  = matplotlib.gridspec.GridSpec(nrows=3,ncols=4,height_ratios=[0.1,0.5,1.],wspace=0.1,hspace = 0.)

#%%


plt.subplot(gs[0,:])

cmap = matplotlib.cm.get_cmap('turbo')
import matplotlib
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    ticks       = [-1, 1],
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("algorithm progress", labelpad=-10, fontsize = labelsize)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(['start', 'end'], fontsize = labelsize)  # horizontal colorbar


plt.gca().annotate('', xy=(0.1, 1.5), xycoords='axes fraction', xytext=(0.425, 1.5), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(0.9, 1.5), xycoords='axes fraction', xytext=(0.575, 1.5), 
            arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.75
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.75

filter_x    = []
filter_y    = []

T           = 20.


#%%

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.75
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.75

triang_x    = np.asarray([+0.5,0,-0.5])*0.75
triang_y    = np.asarray([-0.5,+0.5,-0.5])*0.75

circ_x      = np.cos(np.linspace(-np.pi,np.pi,36))*0.5*0.75
circ_y      = np.sin(np.linspace(-np.pi,np.pi,36))*0.5*0.75

plt.subplot(gs[1,:])

plt.fill(1 + triang_x, 0 + triang_y, color=cmap(0.25))

plt.fill(2 + circ_x, 0 + circ_y, color=cmap(0.3))

plt.gca().annotate('', xy=(2, 0), xytext=(1, 0), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(3, 0, "filtering forecast", ha="left", va="center",zorder=10,color="k",fontsize=labelsize)

plt.fill(12 + triang_x, 0.5 + triang_y, color=cmap(0.5))

plt.fill(12 + circ_x, -0.5 + circ_y, color=cmap(0.55))

plt.gca().annotate('', xy=(12, 0.5), xytext=(12, -0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(13, 0, "filtering update", ha="left", va="center",zorder=10,color="k", fontsize = labelsize)

plt.fill(22 + square_x, 0.5 + square_y, color=cmap(0.8))

plt.fill(22 + square_x, -0.5 + square_y, color=cmap(0.8))

plt.fill(23 + triang_x, 0.5 + triang_y, color=cmap(0.75))

plt.fill(23 + square_x, -0.5 + square_y, color=cmap(0.75))

plt.gca().annotate('', xy=(22, 0.5), xytext=(23, 0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().annotate('', xy=(22, -0.5), xytext=(23, -0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(24, 0, "smoothing update", ha="left", va="center",zorder=10,color="k", fontsize = labelsize)


# plt.gca().annotate('', xy=(6, 0.5), xytext=(6, -0.5), 
#         arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))


# Dummy point to make axes work
plt.fill(32 + circ_x, 0 + circ_y, color=cmap(0.3),alpha = 0.)
plt.fill(-3 + circ_x, 0 + circ_y, color=cmap(0.3),alpha = 0.)

plt.axis('equal')

ylim    = plt.gca().get_ylim()
plt.gca().set_ylim([ylim[0]-0.3,ylim[1]-0.3])

plt.axis("off")

#%%

plt.subplot(gs[2,0])

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = 208

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    # Forecast
    if t < T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    for s in np.arange(t-1,-1,-1):
        
        counter += 1
        
        if s == t-1: # Analysis
        
            plt.fill(s + triang_x, t + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
            
        else:
            
            plt.fill(s + square_x, t + square_y, color=cmap(counter/maxcounter), edgecolor="None")
            
            
plt.title(r'$\bf{A}$: multi-pass (every step)', loc='left', fontsize = labelsize)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10, fontsize = labelsize)
plt.ylabel('time scubscript $s$', labelpad=-10, fontsize = labelsize)
plt.ylabel('conditioned on data', labelpad=pad, fontsize = labelsize)
ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'], fontsize = labelsize)
ax.set_yticks([2,T-1])
ax.set_yticklabels(['$\mathbf{y}_{1}^{*}$','$\mathbf{y}_{1:t}^{*}$'], fontsize = labelsize)

#%%

plt.subplot(gs[2,1])

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.75
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.75

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = 60

cmap = matplotlib.cm.get_cmap('turbo')


# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    # Forecast
    if t < T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    # Forecast
    plt.fill(t - 1 + triang_x, t + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    if (t+1)%5 == 0:
    
        for s in np.arange(t-2,-1,-1):
            
            counter += 1
            
            plt.fill(s + square_x, t + square_y, color=cmap(counter/maxcounter), edgecolor="None")

plt.title(r'$\bf{B}$: multi-pass (selected steps)', loc='left', fontsize=labelsize)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10, fontsize = labelsize)
ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'], fontsize = labelsize)
ax.set_yticks([2,T-1])
ax.set_yticklabels(['',''])

#%%

plt.subplot(gs[2,2])


xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.75
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.75

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = 117

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    # Forecast
    if t < T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    for s in np.arange(t-1,np.maximum(-1,t-7),-1):
        
        counter += 1
        
        if s == t-1: # Analysis
        
            plt.fill(s + triang_x, t + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
            
        else:
            
            plt.fill(s + square_x, t + square_y, color=cmap(counter/maxcounter), edgecolor="None")


plt.title(r'$\bf{C}$: multi-pass (fixed-lag)', loc='left', fontsize=labelsize)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10, fontsize = labelsize)
ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'], fontsize = labelsize)
ax.set_yticks([2,T-1])
ax.set_yticklabels(['',''])

#%%

plt.subplot(gs[2,3])

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.75
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.75

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = 36

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    # Forecast
    if t < T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    # Analysis
    plt.fill(t + triang_x - 1, t + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    if t == T-1:
    
        for s in np.arange(t-2,-1,-1):
            
            counter += 1
                
            plt.fill(s + square_x, t + square_y, color=cmap(counter/maxcounter), edgecolor="None")

plt.title(r'$\bf{D}$: single-pass', loc='left', fontsize=labelsize)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10, fontsize = labelsize)
ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'], fontsize = labelsize)
ax.set_yticks([2,T-1])
ax.set_yticklabels(['',''])

plt.savefig('smoother_types_backward'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('smoother_types_backward'+addendum+'.pdf',dpi=600,bbox_inches='tight')












