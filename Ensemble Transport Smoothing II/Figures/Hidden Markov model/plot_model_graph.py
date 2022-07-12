import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import turbo_colormap
import colorsys

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:midnight blue",
     "xkcd:sky blue",
     "xkcd:light sky blue"])

color1  = 'xkcd:cerulean'
color2  = '#62BEED'

cmap = matplotlib.cm.get_cmap('turbo')

plt.close('all')

Nnodes  = 7

plt.figure(figsize=(10,3))
gs  = matplotlib.gridspec.GridSpec(nrows=2,ncols=1,width_ratios=[1],height_ratios=[0.1,1])

#%%

plt.subplot(gs[0,0])


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

plt.subplot(gs[1,0])




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
plt.axis('off')

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)


# Save the figure
plt.savefig('Hidden_Markov_model.png',dpi=600,bbox_inches='tight')
plt.savefig('Hidden_Markov_model.pdf',dpi=600,bbox_inches='tight')