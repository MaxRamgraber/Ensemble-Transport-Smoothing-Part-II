if __name__ == '__main__':    

    # Load in a number of libraries we will use
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy.stats
    import copy
    import scipy.optimize
    from transport_map_138 import *
    import time
    import os
    from matplotlib.lines import Line2D
    import scipy.linalg
    
    # Find the current path
    root_directory = os.path.dirname(os.path.realpath(__file__))

    # This is a small auxiliary function converting "1" to "1st" and "2" to "2nd"
    # and so on, for printing later on
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    
    # Define and initialize a random seed
    random_seed     = 0
    np.random.seed(random_seed)

    # -------------------------------------------------------------------------
    # Set up exercise
    # -------------------------------------------------------------------------
    
    # Define problem dimensions
    O                   = 1 # Observation space dimensions
    D                   = 2 # State space dimensions
    
    # Ensemble size
    N                   = 1000
    
    # Set up time
    T                   = 500 #150  # Full time series length
    
    # Observation error
    obs_sd              = 0.1
    
    # Forecast error
    mod_cov             = np.asarray([[0,0],[0,0]])                     # No error
    # mod_cov             = np.asarray([[0.01**2,0],[0,(0.01/25)**2]])    # Small error
    # mod_cov             = np.asarray([[0.1**2,0],[0,(0.1/25)**2]])      # High error
    
    # Coefficient for the true dynamics
    t_sine_scale        = 25
    
    # Define the order for the filter and smoother
    order_filter        = 2
    order_smoother      = 2
    
    # Define the model forecast dynamics
    F                   = np.asarray([[0, 1],[-(1/25)**2, 0]])
    Phi                 = scipy.linalg.expm(F)

    # =========================================================================
    # Create observations and synthetic reference
    # =========================================================================
    
    # Reset random seed
    np.random.seed(0)    

    # Create the synthetic reference
    synthetic_truth         = np.zeros((T,1,O))
    
    # Pre-allocate space for the observations
    observations            = np.zeros((T,1,O))
    observations[0,0,:]     = scipy.stats.norm.rvs(
        loc     = np.abs(synthetic_truth[0,0,:]),
        scale   = obs_sd,
        size    = 1)
    
    for t in np.arange(0,T-1,1):
        
        # Make a stochastic volatility forecast
        synthetic_truth[t+1,0,:]    = np.sin((t+1)/t_sine_scale)
        
        # Extract the corresponding observation
        observations[t+1,0,:]       = \
            np.abs(scipy.stats.norm.rvs(
                loc     = synthetic_truth[t+1,0,:],
                scale   = obs_sd))
        
        
    # Remove the unnecessary particle index
    synthetic_truth     = synthetic_truth[:,0,:]
    observations        = observations[:,0,:]
            
    # Reset random seed
    np.random.seed(0)    

    # =====================================================================
    # Create the filter reference
    # =====================================================================
        
    # Create the map component definitions for the monotone and nonmonotone terms
    monotone_filter     = []
    nonmonotone_filter  = []
    
    nonmonotone_filter  = [
        [[]],
        [[]] ]

    monotone_filter     = [
        ['RBF 0','RBF 0','RBF 1','RBF 1'],
        ['RBF 0','RBF 0','RBF 1','RBF 1','RBF 2','RBF 2'] ]
            
    # -------------------------------------------------------------------------
    # Prepare the stochastic map filtering
    # -------------------------------------------------------------------------
    
    # Initialize the array for the analysis
    Z_a         = np.zeros((T,N,D))
    Z_a[0,:,:]  = scipy.stats.norm.rvs(
        loc     = 0,
        scale   = 0.2,
        size    = (N,D))
    Z_a[0,:,1]  /= 25 # Correct for time step
    
    # Initialize the array for the forecast
    Z_f         = copy.copy(Z_a)
    
    # Initialize the arrays for the simulated observations
    Y_sim       = np.zeros((T,N,O))
    
    # Initialize the list for the RMSE
    RMSE_list           = []
    
    # Create dummy input
    map_input = np.column_stack((
        Z_f[0,:,0][:,np.newaxis],   # First dimension: simulated observations
        Z_f[0,:,:]))                # Next dimension: predicted states
    
    # Delete any existing transport map objects
    if "tm" in globals():
        del tm

    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone_filter,          # Monotone terms of the map components
        nonmonotone             = nonmonotone_filter,       # Nonmonotone terms of the map components
        X                       = copy.copy(map_input),     # Training samples
        monotonicity            = "integrated rectifier",   # Monotonicity type - integration required for cross-terms
        ST_scale_mode           = "static",                 # RBF scales have a static bandwith, defined below
        ST_scale_factor         = 3.,                       # Bandwidth of the RBFs
        regularization          = "l2",
        regularization_lambda   = 1E-6,
        standardize_samples     = True,                     # Flag whether samples are standardized; should almost always be True
        quadrature_input        = {                         # Keywords for the Gaussian quadrature used for integration
            'order'             : 5,
            'adaptive'          : False,
            'threshold'         : 1E-9,
            'verbose'           : False}) 
        
    # Start the filtering
    for t in np.arange(0,T,1):
            
        # Print an update message
        print('Now assimilating the '+ordinal(t)+' data point.')

        # Copy the forecast into the analysis array
        Z_a[t,:,:]  = copy.copy(Z_f[t,:,:])
    
        # Start the time
        start   = time.time()
        
        # Simulate observations
        Y_sim[t,...] = \
            np.abs(scipy.stats.norm.rvs(
                loc     = Z_f[t,:,:1],
                scale   = obs_sd,
                size    = (N,O)))

        # Create the uninflated map input
        map_input = copy.copy(np.column_stack((
            Y_sim[t,:,:],   # First O dimensions: simulated observations
            Z_a[t,:,:])))   # Next D dimensions: predicted states
            
        # Reset the transport map with the new values
        tm.reset(map_input)
        
        # Start optimizing the transport map
        tm.optimize()
        
        # Push forward
        norm_samples = tm.map(map_input)
        
        # Prepare the conditioning values, copies of the observaiton
        X_star = np.repeat(
            a       = observations[t,:],
            repeats = N, 
            axis    = 0)[:,np.newaxis]
        
        # Pull back
        ret = tm.inverse_map(
            X_star      = X_star,
            Z           = norm_samples)
        
        # Ssave the conditioned samokes
        Z_a[t,...]  = copy.copy(ret)
            
        # Stop the clock and print optimization time
        end     = time.time()
        print('Optimization took '+str(end-start)+' seconds.')
            
        # Calculate RMSE
        RMSE = np.sqrt(np.mean(np.minimum(np.abs(Z_a[t,...] - synthetic_truth[t,:][:,np.newaxis]),np.abs(Z_a[t,...] + synthetic_truth[t,:][:,np.newaxis]))**2))
        RMSE_list.append(RMSE)
        
        # Last 2000 label
        if t >= 2000:
            lastlabel   = str(np.mean(RMSE_list[2000:]))
        else:
            lastlabel   = "t<2000"
        
        print('Average ensemble RMSE is '+str(RMSE)+ '  (N='+str(N)+') | Average so far: '+str(np.mean(RMSE_list)))
        

        # After the analysis step, make a forecast to the next timestep
        if t < T-1:
            
            Z_f[t+1,:,:] = np.dot(Phi,Z_a[t,:,:].T).T
            
            # Make a stochastic volatility forecast
            Z_f[t+1,:,:]    = \
                copy.copy(Z_f[t+1,:,:]) + \
                scipy.stats.multivariate_normal.rvs(
                    cov     = mod_cov,
                    size    = N) 
            

    # Declare victory
    print('Computations finished.')

    #%%
    
    # =========================================================================
    # Smoothing
    # =========================================================================

    # Define the map component structure
    nonmonotone_smoother  = [
        [[]],
        [[]] ]

    monotone_smoother     = [
        ['RBF 0','RBF 0','RBF 1','RBF 1','RBF 2','RBF 2'],
        ['RBF 0','RBF 0','RBF 1','RBF 1','RBF 2','RBF 2','RBF 3','RBF 3'] ]

    # -----------------------------------------------------------------
    # Prepare the backwards smoother
    # -----------------------------------------------------------------
    
    # Initialize the array for the analysis
    Z_s         = np.zeros((T,N,D))
    Z_s         = copy.copy(Z_a)
    
    # Initialize the list for the RMSE
    RMSE_list   = []
    
    # Calculate the last marginal's RMSE
    RMSE = np.sqrt(np.mean(np.minimum(np.abs(Z_s[T-1,...] - synthetic_truth[t,:]),np.abs(Z_s[T-1,...] + synthetic_truth[t,:]))**2))
    RMSE_list.append(RMSE)
    
    # Create dummy map input
    map_input = copy.copy(np.column_stack((
        Z_f[T-1,:,:],    # First D dimensions: forecasted states
        Z_a[T-2,...])))  # Next D dimensions: filtering analysis states
    
    # Delete any existing map objects
    if "tm" in globals():
        del tm
        
    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone_smoother,        # Monotone terms of the map components
        nonmonotone             = nonmonotone_smoother,     # Nonmonotone terms of the map components
        X                       = copy.copy(map_input),     # Training samples
        monotonicity            = "integrated rectifier",   # Monotonicity type - integration required for cross-terms
        ST_scale_mode           = "static",                 # RBF scales have a static bandwith, defined below
        ST_scale_factor         = 3,                        # Bandwidth of the RBFs
        regularization          = "l2",
        regularization_lambda   = 1E-6,
        standardize_samples     = True,                     # Flag whether samples are standardized; should almost always be True
        quadrature_input        = {                         # Keywords for the Gaussian quadrature used for integration
            'order'             : 5,
            'adaptive'          : False,
            'threshold'         : 1E-9,
            'verbose'           : False})


    # Now start smoothing; move backwards through time
    for t in range(T-2,-1,-1):
        
        # Assemble the training samples
        map_input = copy.copy(np.column_stack((
            Z_f[t+1,:,:],       # First D dimensions: forecasted states
            Z_a[t,...])))       # Next D dimensions: filtering analysis states
    
        # We want to condition on the previous smoothing marginal
        X_star = copy.copy(
            Z_s[t+1,...])
        
        # Reset the map
        tm.reset(map_input)
        
        # Start optimizing the map
        start   = time.time()
        tm.optimize()
        end     = time.time()
        print('Optimization took '+str(end-start)+' seconds.')
        
        # Push forward
        norm_samples = tm.map(map_input)
        
        # Pull back
        ret = tm.inverse_map(
            X_star      = X_star,
            Z           = norm_samples) # Only necessary when heuristic is deactivated

        # Save the results
        Z_s[t,...]    = copy.copy(ret)

        # Calculate RMSE
        RMSE = np.sqrt(np.mean(np.minimum(np.abs(Z_s[t,...] - synthetic_truth[t,:]),np.abs(Z_s[t,...] + synthetic_truth[t,:]))**2))
        RMSE_list.append(RMSE)
        
        # Print the RMSE results obtained thus far
        print('Average ensemble RMSE (N='+str(N)+',t='+str(t)+'/'+str(T)+') is '+str(RMSE)+ ' | Average so far: '+str(np.mean(RMSE_list)))
        
    # Declare victory
    print('Computations finished.')
    
    
    
    #%%
    
    # -------------------------------------------------------------------------
    # Linear smoothing (nonlinear filtering basis)
    # -------------------------------------------------------------------------

    # Initialize the array for the analysis
    Z_s_l      = np.zeros((T,N,D))
    Z_s_l      = copy.copy(Z_a)
    
    # Initialize the list for the RMSE
    RMSE_list_l  = []
    
    # Calculate RMSE
    RMSE = np.sqrt(np.mean(np.minimum(np.abs(Z_s_l[T-1,...] - synthetic_truth[t,:]),np.abs(Z_s_l[T-1,...] + synthetic_truth[t,:]))**2))
    RMSE_list_l.append(RMSE)
    
    # Create dummy map input
    map_input = copy.copy(np.column_stack((
        Z_f[T-1,:,:],    # First D dimensions: smoothed states
        Z_a[T-2,...])))  # Next D dimensions: smoothed states
    
    # Delete any existing map object
    if "tm" in globals():
        del tm
        
    nonmonotone_smoother_linear  = [
        [[],[0],[1]],
        [[],[0],[1],[2]] ]

    monotone_smoother_linear     = [
        [[2]],
        [[3]] ]
        
    # Parameterize the transport map
    tm     = transport_map(
        monotone                = monotone_smoother_linear,                  # Monotone terms of the map components
        nonmonotone             = nonmonotone_smoother_linear,               # Nonmonotone terms of the map components
        X                       = copy.copy(map_input),     # Training samples
        monotonicity            = "separable monotonicity", # Monotonicity type - separable is sufficient for linear maps
        standardize_samples     = True)                     # Flag whether samples are standardized; should almost always be True

    # Start smoothing
    for t in range(T-2,-1,-1):
        
        # Assemble the training samples
        map_input = copy.copy(np.column_stack((
            Z_f[t+1,:,:],       # First D dimensions: forecasted states
            Z_a[t,...])))       # Next D dimensions: filtering analysis states
    
        # We want to condition on the previous smoothing marginal
        X_star = copy.copy(
            Z_s_l[t+1,...])
        
        # Reset the map
        tm.reset(copy.copy(map_input))
        
        # Start optimizing the transport map
        start   = time.time()
        tm.optimize()
        end     = time.time()
        print('Optimization took '+str(end-start)+' seconds.')
        
        # Push forward
        norm_samples = tm.map(map_input)
        
        # Pull back
        ret = tm.inverse_map(
            X_star      = X_star,
            Z           = norm_samples) # Only necessary when heuristic is deactivated

        # Save the results
        Z_s_l[t,...]    = copy.copy(ret)

        # Calculate RMSE
        RMSE = np.sqrt(np.mean(np.minimum(np.abs(Z_s_l[t,...] - synthetic_truth[t,:]),np.abs(Z_s_l[t,...] + synthetic_truth[t,:]))**2))
        RMSE_list_l.append(RMSE)
        
        # Print the RMSE results obtained thus far
        print('Average ensemble RMSE (N='+str(N)+',t='+str(t)+'/'+str(T)+') is '+str(RMSE)+ ' | Average so far: '+str(np.mean(RMSE_list)))
        
    # Declare victory
    print('Computations finished.')
    
    #%%
    
    # -------------------------------------------------------------------------
    # Plot the results
    # -------------------------------------------------------------------------
    
    # Initialize a figure with specified size
    plt.figure(figsize=(16,9))
    
    # Define subplot structure
    gs  = GridSpec(
        nrows           = 2,
        ncols           = 2,
        width_ratios    = [10,1],
        height_ratios   = [1,2],
        wspace          = 0,
        hspace          = 0.4)
    
    # Enter the first subplot
    plt.subplot(gs[0,0])
    
    # Scatter the filtering marginal samples
    for n in range(N):
        plt.scatter(np.arange(T),Z_a[:,n,0],1,color='xkcd:silver',alpha=0.01)
    
    # Scatter the observations
    plt.scatter(np.arange(T),observations,1,color='xkcd:cerulean', label = 'observations')

    # Plot the true state
    plt.plot(np.arange(T),synthetic_truth,color='xkcd:crimson',alpha=0.5, label = 'true state', lw = 2)
    
    # Define the axis limits
    ylims   = [-2,2]
    plt.ylim(ylims)
    plt.xlim([0,T])
    
    # Label the axes
    plt.title("nonlinear ensemble transport filter",fontsize=10)
    plt.ylabel("state $x^{a}$")
    plt.xlabel("time step")
    
    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='filtering posterior samples',
                              markerfacecolor='xkcd:silver', markersize=5),
                       Line2D([0], [0], color='xkcd:crimson', lw=1, label='true state'),
                       Line2D([0], [0], marker='o', color='w', label='observations',
                              markerfacecolor='xkcd:cerulean', markersize=5)]
    plt.gca().legend(handles=legend_elements, ncol = 4, frameon = False, loc = 'upper right')
    
    plt.plot(
        [125,125],
        [-2,2],
        color   = 'xkcd:dark grey',
        ls      = '--')
    
    # Define a position in subplot dimensiojs
    xpos    = [-0.05,1.12]
    ypos    = [-0.25,1.15]
    xdif    = np.abs(np.diff(xpos))
    ydif    = np.abs(np.diff(ypos))
    
    # Label this subplot
    plt.text(xpos[0],ypos[1]+0.1,r'$\bf{A}$: Filter', 
        transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top',horizontalalignment='left')
    
    # Draw a grey box around it
    plt.gca().annotate('', xy=(xpos[0], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[1]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[0], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[1]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[1], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[0]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[1], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[0]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    
    plt.subplot(gs[0,1])
    
    resolution  = 31
    binpos      = np.linspace(-2,2,resolution+1)[1:]
    binpos      -= (binpos[1]-binpos[0])/2
    binwidth    = binpos[1]-binpos[0]
    bins        = [[x-binwidth/2,x+binwidth/2] for x in binpos]
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(Z_a[125,:,0] >= bn[0],Z_a[125,:,0] < bn[1])) for bn in bins]:
        indices.append(ind[0])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = 'xkcd:dark grey')
    
    plt.ylim([-2,2])
    
    xlim    = plt.gca().get_xlim()
    plt.gca().set_xlim([0,xlim[-1]*1.1])
    
    plt.plot(
        [0, xlim[-1]*1.1],
        [synthetic_truth[125],synthetic_truth[125]],
        color   = 'xkcd:crimson')
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    # -------------------------------------------------------------------------
    
    # Subdivide the second subplot
    gs2     = gs[1,:].subgridspec(
        nrows   = 2, 
        ncols   = 2,
        wspace  = 0,
        width_ratios = [10,1])
    
    # Enter the first of the subdivided subplots
    plt.subplot(gs2[0,0])
    
    # Scatter the linear smoothing marginals
    for n in range(N):
        plt.scatter(np.arange(T),Z_s_l[:,n,0],1,color='xkcd:silver',alpha=0.01)
        
    # Scatter the observations
    plt.scatter(np.arange(T),observations,1,color='xkcd:cerulean')
    
    # Plot the true state
    plt.plot(np.arange(T),synthetic_truth,color='xkcd:crimson', alpha=0.5, lw = 2)
    
    # Define the axis limits
    plt.ylim(ylims)
    plt.xlim([0,T])
    plt.gca().set_xticklabels([])
    
    # Label the axes
    plt.title("linear ensemble transport smoother",fontsize=10)
    plt.ylabel("state $x^{a}$")
    
    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='smoothing posterior samples',
                              markerfacecolor='xkcd:silver', markersize=5),
                       Line2D([0], [0], color='xkcd:crimson', lw=1, label='true state'),
                       Line2D([0], [0], marker='o', color='w', label='observations',
                              markerfacecolor='xkcd:cerulean', markersize=5)]
    plt.gca().legend(handles=legend_elements, ncol = 4, frameon = False, loc = 'upper right')
    
    # Define a position in subplot dimensions
    xpos    = [-0.05,1.12]
    ypos    = [-1.48,1.15]
    xdif    = np.abs(np.diff(xpos))
    ydif    = np.abs(np.diff(ypos))
    
    plt.plot(
        [125,125],
        [-2,2],
        color   = 'xkcd:dark grey',
        ls      = '--')
    
    # Label the subplot
    plt.text(xpos[0],ypos[1]+0.1,r'$\bf{B}$: Smoothers', 
        transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top',horizontalalignment='left')
    
    # Draw a grey box around both subdivided subplots
    plt.gca().annotate('', xy=(xpos[0], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[1]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[0], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[1]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[1], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[0]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    plt.gca().annotate('', xy=(xpos[1], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[0]), 
                        arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))
    
    
    
    plt.subplot(gs2[0,1])
    
    resolution  = 31
    binpos      = np.linspace(-2,2,resolution+1)[1:]
    binpos      -= (binpos[1]-binpos[0])/2
    binwidth    = binpos[1]-binpos[0]
    bins        = [[x-binwidth/2,x+binwidth/2] for x in binpos]
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(Z_s_l[125,:,0] >= bn[0],Z_s_l[125,:,0] < bn[1])) for bn in bins]:
        indices.append(ind[0])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = 'xkcd:dark grey')
    
    plt.ylim([-2,2])
    
    xlim    = plt.gca().get_xlim()
    plt.gca().set_xlim([0,xlim[-1]*1.1])
    
    plt.plot(
        [0, xlim[-1]*1.1],
        [synthetic_truth[125],synthetic_truth[125]],
        color   = 'xkcd:crimson')
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    
    
    # -------------------------------------------------------------------------
    
    # Enter the second subdivided subplot
    plt.subplot(gs2[1,0])
    
    # Scatter the nonlinear smoothing marginal samples
    for n in range(N):
        plt.scatter(np.arange(T),Z_s[:,n,0],1,color='xkcd:silver',alpha=0.01)
    
    # Scatter the observations
    plt.scatter(np.arange(T),observations,1,color='xkcd:cerulean')
    
    # Plot the true state
    plt.plot(np.arange(T),synthetic_truth,color='xkcd:crimson', alpha=0.5, lw = 2)
    
    # Define the axis limits
    plt.ylim(ylims)
    plt.xlim([0,T])
    
    # Label the subplot and axes
    plt.title("nonlinear ensemble transport smoother",fontsize=10)
    plt.ylabel("state $x^{a}$")
    plt.xlabel("time step")

    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='smoothing posterior samples',
                              markerfacecolor='xkcd:silver', markersize=5),
                       Line2D([0], [0], color='xkcd:crimson', lw=1, label='true state'),
                       Line2D([0], [0], marker='o', color='w', label='observations',
                              markerfacecolor='xkcd:cerulean', markersize=5)]
    plt.gca().legend(handles=legend_elements, ncol = 4, frameon = False, loc = 'upper right')
    
    
    
    plt.plot(
        [125,125],
        [-2,2],
        color   = 'xkcd:dark grey',
        ls      = '--')
    
    plt.subplot(gs2[1,1])
    
    resolution  = 31
    binpos      = np.linspace(-2,2,resolution+1)[1:]
    binpos      -= (binpos[1]-binpos[0])/2
    binwidth    = binpos[1]-binpos[0]
    bins        = [[x-binwidth/2,x+binwidth/2] for x in binpos]
    
    # Get the indices in the bin, sort them by their y position
    indices = []
    for ind in [np.where(np.logical_and(Z_s[125,:,0] >= bn[0],Z_s[125,:,0] < bn[1])) for bn in bins]:
        indices.append(ind[0])
    
    # Plot manually
    for bni,bn in enumerate(bins):
        for j,ind in enumerate(indices[bni]):
            bnred   = binwidth*0.1
            plt.fill(
                [j,j,j+1,j+1],
                [bn[0]+bnred,bn[1]-bnred,bn[1]-bnred,bn[0]+bnred],
                color = 'xkcd:dark grey')
    
    plt.ylim([-2,2])
    
    xlim    = plt.gca().get_xlim()
    plt.gca().set_xlim([0,xlim[-1]*1.1])
    
    plt.plot(
        [0, xlim[-1]*1.1],
        [synthetic_truth[125],synthetic_truth[125]],
        color   = 'xkcd:crimson')
    
    # Remove all axis ticks
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    
    # Save the figure
    plt.savefig('bimodal_smoother_true.png',dpi = 300,bbox_inches='tight')
    
    
    
    
    

        
    