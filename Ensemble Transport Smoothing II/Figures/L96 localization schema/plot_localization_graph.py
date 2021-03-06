import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import turbo_colormap
import colorsys
import pickle
import scipy.linalg

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:midnight blue",
     "xkcd:sky blue",
     "xkcd:light sky blue"])

color1  = 'xkcd:cerulean'
color2  = '#62BEED'

cmap = matplotlib.cm.get_cmap('turbo')

def whiten(color,fac = 0.75):
    
    color   = np.asarray(color)
    
    color   = color + (np.ones(len(color))-color)*fac
    
    return color

plt.close('all')

#%%

# =============================================================================
# Get the localization structure for the filter and smoother
# =============================================================================

O       = 20
D       = 40
    
# Get the localization lengths for the filter

# Load in the filter step's covariance matrix
cov_filter          = pickle.load(open("cov_filter.p","rb"))
covinv_filter       = np.abs(scipy.linalg.inv(cov_filter))

# Threshold value for the absolute covariance entries, limit localization
# to neighbours with larger covariance values
threshold           = 0.01

# For all observed states, determine the optimal localization radius
locrads_filter      = []

# Go through all observed states
for oi,d in enumerate(np.arange(O,O+D,2)):
    
    # Start a counter for the localization radius
    counter     = 0
    
    # Append a new localization radius
    # locrads_filter.append(0)
    locrads_filter.append(np.zeros(39))
    locrads_filter[-1][20]     = 1
    
    # Keep doing this until stop
    repeat      = True
    while repeat:  
        
        # Increase localization radius by one
        counter     += 1
        
        # Find the neighbouring indices
        previdx     = d-counter
        nextidx     = d+counter
        
        # Wrap the index to the left around
        if previdx < O:
            previdx += D
        elif previdx >= O+D:
            previdx -= D+D
            
        # Wrap the index to the right around
        if nextidx < O:
            nextidx += D
        elif nextidx >= O+D:
            nextidx -= D
            
        # If both neighbours are above the threshold, increase localization radius
        if covinv_filter[d,nextidx] > threshold:
            locrads_filter[-1][20+counter] = 1
        
        # If both neighbours are above the threshold, increase localization radius
        if covinv_filter[d,previdx] > threshold:
            locrads_filter[-1][20-counter] = 1
        
        if covinv_filter[d,nextidx] <= threshold and covinv_filter[d,previdx] <= threshold:
            repeat  = False
            
# Average the localization radius
locrad_filter   = np.mean(np.asarray(locrads_filter),axis=0)
locrad_filter   = int(np.ceil((np.sum(locrad_filter)-1)/2))

# -----------------------------------------------------------------------------

# Load in the covariance matrix
cov_smoother        = pickle.load(open("cov_smoother.p","rb"))
covinv_smoother     = np.abs(scipy.linalg.inv(cov_smoother))

locrads_smoother_current     = []
locrads_smoother_future      = []
for d in np.arange(D,D+D,1):
    
    
    # Start a counter for the localization radius
    counter     = 0
    
    # Append a new localization radius
    locrads_smoother_future.append(np.zeros(39))
    locrads_smoother_future[-1][20]     = 1
    
    # Keep doing this until stop
    repeat      = True
    while repeat:  
        
        # Increase localization radius by one
        counter     += 1
        
        # Find the neighbouring indices
        previdx     = d-D-counter
        nextidx     = d-D+counter
        
        # Wrap the index to the left around
        if previdx < 0:
            previdx += D
        elif previdx >= D:
            previdx -= D+D
            
        # Wrap the index to the right around
        if nextidx < 0:
            nextidx += D
        elif nextidx >= D:
            nextidx -= D
            
        # If both neighbours are above the threshold, increase localization radius
        if covinv_smoother[d,nextidx] > threshold:
            locrads_smoother_future[-1][20+counter] = 1
        
        # If both neighbours are above the threshold, increase localization radius
        if covinv_smoother[d,previdx] > threshold:
            locrads_smoother_future[-1][20-counter] = 1
        
        if covinv_smoother[d,nextidx] <= threshold and covinv_smoother[d,previdx] <= threshold:
            repeat  = False
            
    # ---------------------------------------------------------------------
    
    # Start a counter for the localization radius
    counter     = 0
    
    # Append a new localization radius
    locrads_smoother_current.append(np.zeros(39))
    locrads_smoother_current[-1][20]     = 1
    
    # Keep doing this until stop
    repeat      = True
    while repeat:  
        
        # Increase localization radius by one
        counter     += 1
        
        # Find the neighbouring indices
        previdx     = d-counter
        nextidx     = d+counter
        
        # Wrap the index to the left around
        if previdx < D:
            previdx += D
        elif previdx >= D+D:
            previdx -= D+D
            
        # Wrap the index to the right around
        if nextidx < D:
            nextidx += D
        elif nextidx >= D+D:
            nextidx -= D
            
        # If both neighbours are above the threshold, increase localization radius
        if covinv_smoother[d,nextidx] > threshold:
            locrads_smoother_current[-1][20-counter] = 1
        
        # If both neighbours are above the threshold, increase localization radius
        if covinv_smoother[d,previdx] > threshold:
            locrads_smoother_current[-1][20+counter] = 1
        
        if covinv_smoother[d,nextidx] <= threshold and covinv_smoother[d,previdx] <= threshold:
            repeat  = False
            
# Average both localization radii
locrad_smoother_future  = np.mean(np.asarray(locrads_smoother_future),axis=0)
locrad_smoother_current = np.mean(np.asarray(locrads_smoother_current),axis=0)
    
# Average both localization radii
locrad_smoother_future  = np.ceil(locrad_smoother_future)
locrad_smoother_current = np.ceil(locrad_smoother_current)

locrad_smoother_future  = np.asarray(locrad_smoother_future,dtype=int)
locrad_smoother_current = np.asarray(locrad_smoother_current,dtype=int)

# del locrads_smoother_future, locrads_smoother_current, locrads_filter



#%%

Nnodes  = 7

fig = plt.figure(figsize=(10,9))
gs  = matplotlib.gridspec.GridSpec(nrows=3,ncols=2,width_ratios=[1,1],height_ratios=[0.1,1,1.4])

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

cb1.set_label("algorithm progress", labelpad=-10)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(['start', 'end'])  # horizontal colorbar

plt.gca().annotate('', xy=(0.1, 1.75), xycoords='axes fraction', xytext=(0.4, 1.75), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(0.9, 1.75), xycoords='axes fraction', xytext=(0.6, 1.75), 
            arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.subplot(gs[1,:])

plt.title(r'$\bf{A}$: hidden Markov model',fontsize=10,loc="left")


labels  = [
    '_{'+str(1)+'}$',
    '_{'+str(2)+'}$',
    '_{'+str(3)+'}$',
    '\dots$',
    '_{t-2}$',
    '_{t-1}$',
    '_{t}$']

Nnodes  = 7

xscale  = 6

xpos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.ones(Nnodes)))


ypos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.zeros(Nnodes)))


skipnode = 3

# def extremize(x):
    
#     return (2*x-1)**0.2



colorcounter = -1
colorcounter_max = 12

pos     = np.row_stack((
    xpos,ypos))

for n in range(Nnodes):
    
    if n != skipnode:
        
        colorcounter += 1
    
    
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        # textcolor = tuple(list(textcolor)+[0.5])
    
        plt.gca().add_patch(plt.Circle(xpos[n,:], 0.3, color=cmap(colorcounter/colorcounter_max)))
        if n != Nnodes-1:
            plt.gca().text(xpos[n,0], xpos[n,1], '$x'+labels[n], ha="center", va="center",zorder=10,color=textcolor,fontsize=12)
        else:
            plt.gca().text(xpos[n,0], xpos[n,1], '$x'+labels[n], ha="center", va="center",zorder=10,color=[0.8,0.8,0.8],fontsize=12)
        
        if n != 0:
            plt.arrow(
                pos[n,0]-0.7,
                pos[n,1],
                0.325,
                0,
                head_width = 0.05,
                ec=cmap(colorcounter/colorcounter_max),
                fc=cmap(colorcounter/colorcounter_max),
                width=0.01)
            
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
           1 - textcolor[2])
        
        
        colorcounter += 1
        plt.gca().add_patch(plt.Circle(ypos[n,:], 0.3, color=cmap(colorcounter/colorcounter_max)))
        if n != Nnodes-1:
            plt.gca().text(ypos[n,0], ypos[n,1], '$y'+labels[n], ha="center", va="center",zorder=10,color=textcolor,fontsize=12)
        else:
            plt.gca().text(ypos[n,0], ypos[n,1], '$y'+labels[n], ha="center", va="center",zorder=10,color=[0.8,0.8,0.8],fontsize=12)
        
        plt.arrow(
            pos[n,0],
            pos[n,1]-1+0.7,
            0,
            -0.325,
            head_width = 0.05,
            ec=cmap(colorcounter/colorcounter_max),
            fc=cmap(colorcounter/colorcounter_max),
            width=0.01,
            zorder=-1)
        
    else:
        
        plt.arrow(
            pos[n,0]-0.7,
            pos[n,1],
            0.325,
            0,
            head_width = 0.05,
            ec=cmap(colorcounter/colorcounter_max),
            fc=cmap(colorcounter/colorcounter_max),
            width=0.01,
            zorder=-1)
        
        colorcounter += 1
        
        plt.gca().add_patch(plt.Circle(xpos[n,:], 0.03, color=cmap(colorcounter/colorcounter_max)))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.1,0]), 0.03, color=cmap(colorcounter/colorcounter_max)))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.1,0]), 0.03, color=cmap(colorcounter/colorcounter_max)))

            
    
plt.gca().relim()
plt.gca().autoscale_view()
plt.axis('equal')
# plt.axis('off')
# plt.title(r'$\bf{B}$: Hidden Markov Model', loc='left')

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)













#%%

ax = fig.add_subplot(gs[2,0], projection='3d')

plt.gca().view_init(elev=0., azim=-70)

plt.title(r'$\bf{B}$: localized backward smoothing',fontsize=10,loc="left")

color1  = cmap((2)/colorcounter_max) #cmap(0.25)
color2  = cmap((0)/colorcounter_max)#cmap(0.95)#cmap(0.75)

circlepos   = np.zeros((40,3))

radincr     = 2*np.pi/40
radoffset   = 0.05

color1  = cmap((2)/colorcounter_max) #cmap(0.25)
color2  = cmap((0)/colorcounter_max)#cmap(0.95)#cmap(0.75)
# color2  = color1

for d in range(40):

    circlepos[d,:] = [0,np.sin(d*radincr+radoffset),np.cos(d*radincr+radoffset)]
    
    if d >= 15 and d < 26:
        colorloc    = color1
    else:
        colorloc    = [0.5,0.5,0.5]
    
    # plt.gca().add_patch(plt.Circle(np.flip(circlepos[d,:]), 0.03, color=colorloc, zorder = 6))

# Just plot these invisibly, to get the subplot borders right
# Bit hacky, but hey, it's just a plot =P
plt.gca().scatter3D(
    circlepos[:,0],
    circlepos[:,1],
    circlepos[:,2],
    color   = 'xkcd:grey',
    alpha   = 0.)
plt.gca().scatter3D(
    circlepos[:,0]-0.5,
    circlepos[:,1],
    circlepos[:,2],
    color   = 'xkcd:silver',
    alpha   = 0.)




# Let's try this differently, save indices
clique_current  = []

for d in range(40):
    
    # Get the index
    idx = d+10+20
    if idx >= 39:
        idx -= D
    elif idx < 0:
        idx += D
    
    if locrad_smoother_current[idx] > 0 and d <= 30: # It's in the clique
        clique_current      .append(d)
        
clique_future   = []

for d in np.arange(39,-1,-1):
    
    # Get the index
    idx = d+10+20
    if idx >= 39:
        idx -= D
    elif idx < 0:
        idx += D
    
    if locrad_smoother_future[idx] > 0: # It's in the clique
        clique_future      .append(d)
        

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


for d in range(40):
    
    # Plot current scatter
    
    if d in clique_current:
        color   = cmap((0)/colorcounter_max)
    else:
        color   = 'xkcd:grey'
        
    # Plot scatter
    plt.gca().scatter3D(
        circlepos[d,0]-0.5,
        circlepos[d,1],
        circlepos[d,2],
        color   = color,
        alpha   = 1.,
        zorder  = -circlepos[d,1]-1)
    
    # Plot future scatter
    
    if d in clique_future:
        color   = cmap((2)/colorcounter_max)
    else:
        color   = 'xkcd:grey'
        
    # Plot scatter
    plt.gca().scatter3D(
        circlepos[d,0],
        circlepos[d,1],
        circlepos[d,2],
        color   = color,
        alpha   = 1.,
        zorder  = -circlepos[d,1]-1)
    
    
    
    
    idx = d+10+20
    
    if idx >= 39:
        idx -= D
    elif idx < 0:
        idx += D
    
    if locrad_smoother_future[idx] > 0 and d <= 30:#d >= 26 and d <= 35:
        
        color1  = cmap((2)/colorcounter_max) #cmap(0.25)
        color2  = cmap((0)/colorcounter_max)#cmap(0.95)#cmap(0.75)
        
    else:
        
        color1  = 'xkcd:grey'
        color2  = 'xkcd:grey'
    
    v1  = circlepos[d,:]
    v2  = circlepos[d-1,:]
    v3  = circlepos[d-1,:]-[0.5,0,0]
    v4  = circlepos[d,:]-[0.5,0,0]
    
    verts   = np.row_stack((v1,v2,v3,v4))

    # v = [list(zip(verts[:,0], verts[:,1], verts[:,2]))]
    # pc = Poly3DCollection(
    #     v,
    #     zorder=-np.max(verts[:,1])+3,
    #     alpha   = 0.2,
    #     color   = color1,
    #     edgecolor   = "None")
    # ax.add_collection3d(pc)


# Plot the updated point
plt.gca().scatter3D(
    circlepos[30,0]-0.5,
    circlepos[30,1],
    circlepos[30,2],
    facecolor   = cmap((0)/colorcounter_max),
    edgecolor   = "xkcd:sky blue",
    s       = 50,
    lw      = 1.5,
    alpha   = 1.,
    zorder  = -circlepos[d,1]-1,
    label   = "updated state")

plt.legend(frameon = False,borderpad=0,loc='lower right')
        
# Get the vertices for the clique
difference  = [x for x in clique_future if x not in clique_current]
difference  = list(np.flip(difference))
difference_offset   = np.linspace(-0.5,0,len(difference)+1)[1:]



verts_clique_current    = []
for d in clique_current:
    verts_clique_current.append(circlepos[d,:]-[0.5,0,0])
    
# Add the missing vertices
for idx,d in enumerate(difference):
    verts_clique_current.append(circlepos[d,:]+[difference_offset[idx],0,0])

verts_clique_future     = []
for d in clique_future:
    verts_clique_future.append(circlepos[d,:])
verts_clique_future     = np.flip(np.asarray(verts_clique_future),axis=0)

    
verts_clique_future     = np.asarray(verts_clique_future)
verts_clique_current    = np.asarray(verts_clique_current)
        
for idx in range(len(verts_clique_future)-1):
    
    v1  = circlepos[d,:]
    v2  = circlepos[d-1,:]
    v3  = circlepos[d-1,:]-[0.5,0,0]
    v4  = circlepos[d,:]-[0.5,0,0]
    
    verts   = np.row_stack((
        verts_clique_future[idx,:],
        verts_clique_future[idx+1,:],
        verts_clique_current[idx+1,:],
        verts_clique_current[idx,:]))

    v = [list(zip(verts[:,0], verts[:,1], verts[:,2]))]
    pc = Poly3DCollection(
        v,
        zorder=-np.max(verts[:,1])+3,
        alpha   = 0.2,
        color   = 'xkcd:cerulean',
        edgecolor   = "None")
    ax.add_collection3d(pc)
    
# raise Exception





counterclique_current  = list(np.arange(0,clique_current[0]+1,1)) + list(np.arange(clique_current[-1],40,1))
counterclique_future   = list(np.arange(0,clique_future[-1]+1,1)) + list(np.arange(clique_future[0],40,1))
shared_indices          = [d for d in np.arange(40) if d in counterclique_current or d in counterclique_future]

difference  = [x for x in shared_indices if not (x in counterclique_future and x in counterclique_current) ]
difference_offset   = np.linspace(0,-0.5,len(difference)+1)[1:]

prevdiff    = 0

for d in range(39):
    
    if (d not in clique_current and d not in clique_future) or \
        ((d not in clique_current and d not in clique_future) and (d+1 in clique_current and d+1 in clique_future)) or \
        ((d in clique_current and d in clique_future) and (d+1 not in clique_current and d+1 not in clique_future)):
    
    # v1  = circlepos[d,:]
    # v2  = circlepos[d-1,:]
    # v3  = circlepos[d-1,:]-[0.5,0,0]
    # v4  = circlepos[d,:]-[0.5,0,0]
    
        verts   = np.row_stack((
            circlepos[d,:],
            circlepos[d+1,:],
            circlepos[d+1,:]-[0.5,0,0],
            circlepos[d,:]-[0.5,0,0]))
    
        v = [list(zip(verts[:,0], verts[:,1], verts[:,2]))]
        pc = Poly3DCollection(
            v,
            zorder=-np.max(verts[:,1])+3,
            alpha   = 0.2,
            color   = 'xkcd:grey',
            edgecolor   = "None")
        ax.add_collection3d(pc)

    # All vertices are in the clique, skip;
    elif d in clique_current and d+1 in clique_current and d in clique_future and d+1 in clique_future:
        
        pass
    
    else:
        
        try:
            difidx = np.where(np.asarray(difference) == d)[0][0]
            
            verts   = np.row_stack((
                circlepos[d,:]-[prevdiff,0,0]-[0.5,0,0],
                circlepos[d+1,:]-[difference_offset[difidx],0,0]-[0.5,0,0],
                circlepos[d+1,:]-[0.5,0,0],
                circlepos[d,:]-[0.5,0,0]))
            
            prevdiff    = difference_offset[difidx]
            
        except:
            
            verts   = np.row_stack((
                circlepos[d,:],
                circlepos[d+1,:],
                circlepos[d+1,:]-[0.5,0,0],
                circlepos[d,:]-[0.5,0,0]))

        

    
        v = [list(zip(verts[:,0], verts[:,1], verts[:,2]))]
        pc = Poly3DCollection(
            v,
            zorder=-np.max(verts[:,1])+3,
            alpha   = 0.2,
            color   = 'xkcd:grey',
            edgecolor   = "None")
        ax.add_collection3d(pc)

color_tuple = (1, 1, 1, 0)

# make the panes transparent
ax.xaxis.set_pane_color(color_tuple)
ax.yaxis.set_pane_color(color_tuple)
ax.zaxis.set_pane_color(color_tuple)

# make the axis lines transparent
ax.w_xaxis.line.set_color(color_tuple)
ax.w_yaxis.line.set_color(color_tuple)
ax.w_zaxis.line.set_color(color_tuple)

# make the grid lines transparent
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


xlim    = plt.gca().get_xlim()
ylim    = plt.gca().get_ylim()
zlim    = plt.gca().get_zlim()

max_range = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]).max() / 4.0

mid_x = (xlim[0]+xlim[1]) * 0.5
mid_y = (ylim[0]+ylim[1]) * 0.5
mid_z = (zlim[0]+zlim[1]) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

color1  = cmap((2)/colorcounter_max) #cmap(0.25)
color2  = cmap((0)/colorcounter_max)#cmap(0.95)#cmap(0.75)

ax.text2D(0.7, 0.5, 'clique', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color='xkcd:dark grey')

ax.text2D(0.3, 0.5, 'localized \n clique', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=color1)



ax.text2D(0.15, 0.85, '$\mathbf{x}_{1}$', horizontalalignment='center', fontsize = 12,
     verticalalignment='center', transform=ax.transAxes,color=cmap((0)/colorcounter_max))

ax.text2D(0.85, 0.85, '$\mathbf{x}_{2}$', horizontalalignment='center', fontsize = 12,
     verticalalignment='center', transform=ax.transAxes,color=cmap((2)/colorcounter_max))


ax.text2D(0.0975, 0.535, '$x_{1}^{30}$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((0)/colorcounter_max))

ax.text2D(0.102, 0.4725, '$x_{1}^{29}$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((0)/colorcounter_max))

ax.text2D(0.112, 0.4, '$x_{1}^{28}$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((0)/colorcounter_max))

ax.text2D(0.13, 0.325, '$\dots$', horizontalalignment='center', rotation = -75,
     verticalalignment='center', transform=ax.transAxes,color=cmap((0)/colorcounter_max))


plt.gca().annotate('', xy=(0., -0.01), xycoords='axes fraction', xytext=(0., 1.01), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(-0.01, 1.), xycoords='axes fraction', xytext=(1.01, 1.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(1., 1.01), xycoords='axes fraction', xytext=(1., -0.01), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(1.01, 0.), xycoords='axes fraction', xytext=(-0.01, 0.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))


xs  = [0.05,   0.65]
ys  = [1.5,   1.775]

plt.gca().annotate('', xy=(.0, 1.), xycoords='axes fraction', xytext=(xs[0]+0.005, ys[1]+0.005), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.))

plt.gca().annotate('', xy=(1.0, 1.), xycoords='axes fraction', xytext=(xs[1], ys[1]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.))




xl  = xs[0]/ys[0]

m   = (ys[0] - 0.)/(xs[1]-1.)

xr  = xs[1] - (ys[0]-1)/m


plt.gca().annotate('', xy=(xl, 1.), xycoords='axes fraction', xytext=(xs[0], ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.,ls="--"))

plt.gca().annotate('', xy=(xr, 1.), xycoords='axes fraction', xytext=(xs[1], ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.,ls="--"))



plt.gca().annotate('', xy=(xs[0], ys[0]-0.008), xycoords='axes fraction', xytext=(xs[0], ys[1]+0.008), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[0]-0.008, ys[1]), xycoords='axes fraction', xytext=(xs[1]+0.008, ys[1]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[1], ys[1]+0.008), xycoords='axes fraction', xytext=(xs[1], ys[0]-0.008), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[1]+0.008, ys[0]), xycoords='axes fraction', xytext=(xs[0]-0.008, ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))



#%%


ax = fig.add_subplot(gs[2,1], projection='3d')

plt.gca().view_init(elev=30, azim=178)

plt.title(r'$\bf{C}$: decomposed, localized filtering',fontsize=10,loc="left")

color1  = cmap((colorcounter-1)/colorcounter_max) #cmap(0.25)
color2  = cmap((colorcounter-0)/colorcounter_max)#cmap(0.95)#cmap(0.75)

circlepos   = np.zeros((40,3))

radincr     = 2*np.pi/40
radoffset   = 0.05

for d in range(40):

    circlepos[d,:] = [np.sin(d*radincr+radoffset),np.cos(d*radincr+radoffset),0]
    
    if d >= 20-locrad_filter and d <= 20 + locrad_filter:
        colorloc    = color1
    else:
        colorloc    = [0.5,0.5,0.5]
    

v = [list(zip(circlepos[:,0], circlepos[:,1], circlepos[:,2]))]
pc = Poly3DCollection(
    v,
    zorder=-np.max(verts[:,1])+3,
    alpha   = 0.2,
    color   = 'xkcd:grey',
    edgecolor   = "None")
ax.add_collection3d(pc)

import copy
smallclique     = copy.copy(np.flip(circlepos[30-locrad_filter:30+locrad_filter+1,:],axis=0)*1.1)



offset  = 4.3
halfcircle  = np.column_stack((
    np.cos(np.linspace(offset,offset+np.pi,36)),
    np.sin(np.linspace(offset,offset+np.pi,36)),
    np.zeros(36)))*0.1

halfcircle  += circlepos[30-locrad_filter,:]

smallclique     = np.row_stack((
    smallclique,
    halfcircle))



smallclique     = np.row_stack((
    smallclique,
    circlepos[30-locrad_filter:30+locrad_filter+1,:]*0.9))


offset  = 2.4+np.pi
halfcircle  = np.column_stack((
    np.cos(np.linspace(offset,offset+np.pi,36)),
    np.sin(np.linspace(offset,offset+np.pi,36)),
    np.zeros(36)))*0.1

halfcircle  += circlepos[30+locrad_filter,:]

smallclique     = np.row_stack((
    smallclique,
    halfcircle))



v = [list(zip(smallclique[:,0], smallclique[:,1], smallclique[:,2]))]
pc = Poly3DCollection(
    v,
    zorder=-np.max(verts[:,1])+3,
    alpha   = 0.2,
    color   = color1,
    edgecolor   = "None")
ax.add_collection3d(pc)








plt.gca().scatter3D(
    circlepos[:,0],
    circlepos[:,1],
    circlepos[:,2],
    color   = 'xkcd:grey',
    alpha   = 0.)


plt.gca().scatter3D(
    circlepos[:,0],
    circlepos[:,1],
    circlepos[:,2]-1.25,
    color   = 'xkcd:silver',
    alpha   = 0.)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

for d in range(40):
    
    

    if d >= 30-locrad_filter and d <= 30+locrad_filter:
        
        color1  = cmap((colorcounter-1)/colorcounter_max) #cmap(0.25)
        color2  = cmap((colorcounter-0)/colorcounter_max)#cmap(0.95)#cmap(0.75)
        
    else:
        
        color1  = 'xkcd:silver'
        color2  = 'xkcd:grey'

    if d == 30:
            
        # Plot scatter
        plt.gca().scatter3D(
            circlepos[d,0],
            circlepos[d,1],
            circlepos[d,2]-1.25,
            color   = color2,
            alpha   = 1.,
            zorder  = -circlepos[d,1]-1)
        
        plt.gca().plot3D(
            [circlepos[d,0],circlepos[d,0]],
            [circlepos[d,1],circlepos[d,1]],
            [circlepos[d,2],circlepos[d,2]-1.25],
            zorder  = -circlepos[d,1],
            color   = color2,
            alpha   = 1)
        
    elif d%2 == 0:
        
        if d == 20:
        
            plt.gca().plot3D(
                [circlepos[d,0],circlepos[d,0]],
                [circlepos[d,1],circlepos[d,1]],
                [circlepos[d,2],circlepos[d,2]-1.25],
                zorder  = -circlepos[d,1],
                color   = color2,
                alpha   = 1)
            
            # Plot scatter
            plt.gca().scatter3D(
                circlepos[d,0],
                circlepos[d,1],
                circlepos[d,2]-1.25,
                color   = color2,
                alpha   = 1.,
                zorder  = -circlepos[d,1]-1)
            
        else:
    
            plt.gca().plot3D(
                [circlepos[d,0],circlepos[d,0]],
                [circlepos[d,1],circlepos[d,1]],
                [circlepos[d,2],circlepos[d,2]-1.25],
                zorder  = -circlepos[d,1],
                color   = 'xkcd:grey',
                alpha   = 0.5)
                
            # Plot scatter
            plt.gca().scatter3D(
                circlepos[d,0],
                circlepos[d,1],
                circlepos[d,2]-1.25,
                color   = 'xkcd:grey',
                alpha   = 1.,
                zorder  = -circlepos[d,1]-1)
            
    if color1  == 'xkcd:silver':
            
        # Plot scatter
        plt.gca().scatter3D(
            circlepos[d,0],
            circlepos[d,1],
            circlepos[d,2],
            color   = color1,
            alpha   = 1.,
            zorder  = -circlepos[d,1]-1)
        
    else:
    
        # Plot the updated point
        plt.gca().scatter3D(
            circlepos[d,0],
            circlepos[d,1],
            circlepos[d,2],
            facecolor   = color1,
            edgecolor   = "xkcd:pink",
            s       = 50,
            lw      = 1.5,
            alpha   = 1.,
            zorder  = -circlepos[d,1]-1)

        # Plot scatter
        plt.gca().scatter3D(
            circlepos[d,0],
            circlepos[d,1],
            circlepos[d,2],
            color   = color1,
            alpha   = 1.,
            zorder  = -circlepos[d,1]-1)







color_tuple = (1, 1, 1, 0)

# make the panes transparent
ax.xaxis.set_pane_color(color_tuple)
ax.yaxis.set_pane_color(color_tuple)
ax.zaxis.set_pane_color(color_tuple)

# xLabel = ax.set_xlabel('\nXXX xxxxxx xxxx x xx x', linespacing=3.2)
# yLabel = ax.set_ylabel('\nYY (y) yyyyyy', linespacing=3.1)
# zLabel = ax.set_zlabel('\nZ zzzz zzz (z)', linespacing=3.4)

# make the axis lines transparent
ax.w_xaxis.line.set_color(color_tuple)
ax.w_yaxis.line.set_color(color_tuple)
ax.w_zaxis.line.set_color(color_tuple)

# make the grid lines transparent
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


xlim    = plt.gca().get_xlim()
ylim    = plt.gca().get_ylim()
zlim    = plt.gca().get_zlim()

max_range = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]).max() / 3.25

zoffset     = -0.5
xoffset     = 0

mid_x = (xlim[0]+xlim[1]) * 0.5
mid_y = (ylim[0]+ylim[1]) * 0.5
mid_z = (zlim[0]+zlim[1]) * 0.5
ax.set_xlim(mid_x - max_range - xoffset, mid_x + max_range - xoffset)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range + zoffset, mid_z + max_range + zoffset)

plt.show()

ax.text2D(0.5, 0.65, 'clique', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color='xkcd:dark grey')

ax.text2D(0.7, 0.4, 'localized clique', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((colorcounter-0)/colorcounter_max))



ax.text2D(0.85, 0.85, '$\mathbf{x}_{t}$', horizontalalignment='center', fontsize = 12,
     verticalalignment='center', transform=ax.transAxes,color=cmap((colorcounter-2)/colorcounter_max))

ax.text2D(0.85, 0.15, '$\mathbf{y}_{t}$', horizontalalignment='center', fontsize = 12,
     verticalalignment='center', transform=ax.transAxes,color=cmap((colorcounter-0)/colorcounter_max))



ax.text2D(0.51, 0.51, '$x_{t}^{20}$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((colorcounter-2)/colorcounter_max))

ax.text2D(0.505, 0.075, '$y_{t}^{20}$', horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,color=cmap((colorcounter-0)/colorcounter_max))


plt.gca().annotate('', xy=(0., -0.01), xycoords='axes fraction', xytext=(0., 1.01), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(-0.01, 1.), xycoords='axes fraction', xytext=(1.01, 1.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(1., 1.01), xycoords='axes fraction', xytext=(1., -0.01), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))
plt.gca().annotate('', xy=(1.01, 0.), xycoords='axes fraction', xytext=(-0.01, 0.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:black',lw=1.))


xs  = [0.675,   0.95]
ys  = [1.175,   1.775]

plt.gca().annotate('', xy=(.0, 1.), xycoords='axes fraction', xytext=(xs[0]+0.005, ys[1]+0.005), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.))

plt.gca().annotate('', xy=(1.0, 1.), xycoords='axes fraction', xytext=(xs[1], ys[1]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.))




xl  = xs[0]/ys[0]

m   = (ys[0] - 0.)/(xs[1]-1.)

xr  = xs[1] - (ys[0]-1)/m


plt.gca().annotate('', xy=(xl, 1.), xycoords='axes fraction', xytext=(xs[0], ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.,ls="--"))

plt.gca().annotate('', xy=(xr, 1.), xycoords='axes fraction', xytext=(xs[1], ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey',lw=1.,ls="--"))


plt.gca().annotate('', xy=(xs[0], ys[0]-0.008), xycoords='axes fraction', xytext=(xs[0], ys[1]+0.008), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[0]-0.008, ys[1]), xycoords='axes fraction', xytext=(xs[1]+0.008, ys[1]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[1], ys[1]+0.008), xycoords='axes fraction', xytext=(xs[1], ys[0]-0.008), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))
plt.gca().annotate('', xy=(xs[1]+0.008, ys[0]), xycoords='axes fraction', xytext=(xs[0]-0.008, ys[0]), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grey'))

# Plot the updated point
plt.gca().scatter3D(
    circlepos[30,0]-1000,
    circlepos[30,1],
    circlepos[30,2],
    facecolor   = cmap((colorcounter-1)/colorcounter_max),
    edgecolor   = "xkcd:pink",
    s       = 50,
    lw      = 1.5,
    alpha   = 1.,
    zorder  = -circlepos[d,1]-1,
    label   = "updated states")

plt.legend(frameon = False,borderpad=0)


#%%

# # # Save the figure
plt.savefig('graph_localized_no_map.png',dpi=600,bbox_inches='tight')
plt.savefig('graph_localized_no_map.pdf',dpi=600,bbox_inches='tight')

