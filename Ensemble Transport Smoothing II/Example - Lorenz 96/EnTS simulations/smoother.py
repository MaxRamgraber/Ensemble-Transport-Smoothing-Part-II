if __name__ == '__main__':    

    # Load in a number of libraries we will use
    import numpy as np
    import scipy.stats
    import copy
    from transport_map_138 import *
    import time
    import pickle
    import os
    
    # Clear previous figures and pickles
    root_directory = os.path.dirname(os.path.realpath(__file__))
    results = []
    results += [each for each in os.listdir(root_directory) if each.endswith('.png') or each.endswith('.p') or each.endswith('.txt') ]
    for fl in range(len(results)):
        if not results[fl].startswith('cov'):
            os.remove(root_directory+'\\'+results[fl])

    # These are the dynamics of the Lorenz-96 system
    def lorenz_dynamics_96(t, Z, F = 8):
        
        dZdt    = np.zeros(Z.shape)
        
        if len(Z.shape) == 1: # Only one particle
        
            D   = len(Z)
        
            for d in range(D):
                
                # Calculate the indices
                idc     = np.asarray([d-2,d-1,d,d+1])
                idc[np.where(idc < 0)] += D
                idc[np.where(idc >= D)] -= D
                
                dZdt[d] = (Z[idc[3]] - Z[idc[0]])*Z[idc[1]] - Z[idc[2]] + F
            
        else:
            
            D   = Z.shape[1]
        
            for d in range(D):
                
                # Calculate the indices
                idc     = np.asarray([d-2,d-1,d,d+1])
                idc[np.where(idc < 0)] += D
                idc[np.where(idc >= D)] -= D
                
                dZdt[:,d] = (Z[:,idc[3]] - Z[:,idc[0]])*Z[:,idc[1]] - Z[:,idc[2]] + F
    
        return dZdt
    
    # Fourth-order Runge-Kutta scheme
    def rk4(Z,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
        
        """
        Parameters
            t       : initial time
            Z       : initial states
            fun     : function to be integrated
            dt      : time step length
            nt      : number of time steps
        
        """
        
        # Prepare array for use
        if len(Z.shape) == 1: # We have only one particle, convert it to correct format
            Z       = Z[np.newaxis,:]
            
        # Go through all time steps
        for i in range(nt):
            
            # Calculate the RK4 values
            k1  = fun(t + i*dt,           Z);
            k2  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k1);
            k3  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k2);
            k4  = fun(t + i*dt + dt,      Z + dt*k3);
        
            # Update next value
            Z   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return Z
    
    # This function calculates the continuous rank probability score
    def crps_ens(forecasts,observations):
    
        """
        Calculates the continuos rank probability score based on a N-by-O ensemble
        of forecasts, and an O vector of observations.
        """
        
        import numpy as np
        import copy
        
        # Create local copies of the variables
        observations    = copy.copy(observations)
        forecasts       = copy.copy(forecasts)
        
        # Convert the observations into a vector
        if np.isscalar(observations):
            observations    = np.asarray([observations])
            
        # Makes sure forecasts are of correct size
        if len(forecasts.shape) < 2:
            forecasts       = forecasts[:,np.newaxis]
        elif len(forecasts.shape) > 2:
            raise ValueError("'forecasts' has to be a N-by-O matrix; current shape: "+str(forecasts.shape))
        
        # Get the ensemble size and number of observations
        N       = forecasts.shape[0]
        O       = forecasts.shape[1]
        
        # Sort the forecasts
        forecasts = np.sort(forecasts, axis=0)
        
        # Pre-allocate a list for the results
        result  = np.zeros(O)
        
        # Now go through all observations
        for o in range(O):
    
            # Initialize basic variables
            obs_cdf         = 0     # Starting value of observation cdf
            forecast_cdf    = 0     # Starting value of forecast cdf
            prev_forecast   = 0     # Previous forecast value
            integral        = 0     # Integral so far
        
            # Go through all samples
            for n, forecast in enumerate(forecasts[:,o]):
                
                # The first time we pass the observation, do this
                if obs_cdf == 0 and observations[o] < forecast:
                    
                    integral += (observations[o] - prev_forecast) * forecast_cdf ** 2
                    integral += (forecast - observations[o]) * (forecast_cdf - 1) ** 2
                    obs_cdf = 1     # The cdf jumps to 1 once we pass it
                
                else:
                    
                    # Add the mismatch between forecast and obs cdf squared
                    integral += ((forecast - prev_forecast)
                                  * (forecast_cdf - obs_cdf) ** 2)
        
                # Incremend the forecast cdf
                forecast_cdf += 1/N
                
                # Current foreacst becomes previous forecast
                prev_forecast = forecast
        
            if obs_cdf == 0:
                # forecast can be undefined here if the loop body is never executed
                # (because forecasts have size 0), but don't worry about that because
                # we want to raise an error in that case, anyways
                integral += observations[o] - forecast
        
            # Save into results
            result[o] = integral
            
        return result
            
    
    # This function constructs the map component function definitions for the filter
    def get_transport_map_functions_filter(D,localization_radius,order_nonmon,order_mon,obs_offset):
        
        # Pre-allocate empty lists for the monotone and nonmonotone parts of 
        # every map component
        monotone    = []
        nonmonotone = []
        
        # Allocate the index list, assuming alternating index permutations:
        # 0 -1 1 -2 2 -3 3 -4 4 etc.
        indexlist   = []
        
        # We implement separate, low-dimensional, sparsified filtering updates
        # As such, the map component doesn't necessarily span all of D, but 
        # only as far as the localization radius permits
        for d in range(np.min([D,1 + 2*localization_radius])):
            
            # Create a list with the state indices so far
            if d == 0:
                indexlist.append(0)
            elif d == 1:
                indexlist.append(1)
            elif np.sign(indexlist[-1]) == 1:
                indexlist.append(- indexlist[-1])
            else:
                indexlist.append(-indexlist[-1]+1)
            
            # ---------------------------------------------------------------------
            # Start with the nonmonotone part
            # ---------------------------------------------------------------------
             
            # Create an empty list for this map component's nonmonotone terms
            nonmonotone.append([])
            
            # Add the constant term
            nonmonotone[-1].append([])
            
            # Only the first dimension depends on the observation; this requires
            # permutating the state vector (done outside)
            if d == 0:
            
                # The d-th map map component contains terms up to dimension d
                for i in np.arange(0,d+obs_offset,1):
                    
                    # Add terms in polynomial order
                    for o in np.arange(1,order_nonmon+1,1):
                    
                        # Add term for the current polynomial order
                        if o > 1: # If it's nonlinear, turn it into a Hermite function
                            nonmonotone[-1].append([i]*o+['HF'])
                        else: # If it's linear, don't
                            nonmonotone[-1].append([i]*o)
                        
        
            # The other dimensions dont
            else:
                
                # The d-th map map component contains terms up to dimension d
                for i in np.arange(1,d+obs_offset,1):
                    
                    # Is this term in the neighbour list?
                    if np.abs(indexlist[i-1] - indexlist[-1]) < localization_radius:
                    
                        # Add terms in polynomial order
                        for o in np.arange(1,order_nonmon+1,1):            
                        
                            # Add term for the current polynomial order
                            if o > 1:
                                nonmonotone[-1].append([i]*o+['HF'])
                            else:
                                nonmonotone[-1].append([i]*o)
        
            # ---------------------------------------------------------------------
            # Now do the monotone part
            # ---------------------------------------------------------------------
        
            # Create an empty list for this map component's monotone terms
            monotone.append([])
            
            if order_mon == 1 or d != 0: # Linear terms are added as is
                
                # Add the linear term
                monotone[-1].append([d+obs_offset])
                
            else: # For nonlinear monotone terms, add order-1 integrated RBFs
            
                # Add left edge term
                monotone[-1].append('LET '+str(d+obs_offset))
                
                # Add integrated RBFs to taste
                for o in np.arange(1,order_mon,1):
                
                    # Add term for the current polynomial order
                    monotone[-1].append('iRBF '+str(d+obs_offset))
                    
                # Add left edge term
                monotone[-1].append('RET '+str(d+obs_offset))
                    
        # print(indexlist)
                    
        return nonmonotone,monotone
    
    # This function constructs the map component function definitions for the smoother
    def get_transport_map_functions_smoother(D,locrad_current,locrad_future,order_nonmon,order_mon):
        
        # Pre-allocate empty lists for the monotone and nonmonotone parts of 
        # every map component
        monotone    = []
        nonmonotone = []
        
        # locrad_future and locrad_current mark which samples relative to index
        # 10 are in the localization neighbourhood; extract the corresponding
        # indices
        locrad_current  = np.where(locrad_current > 0.)[0]-20
        locrad_future   = np.where(locrad_future > 0.)[0]-20
        
        # Go through all dimensions
        for i in range(D):
            
            # What neighbouring blocks does this state depend on? Depends on
            # the localization radius
            # combins_future     = np.arange(i-locrad_future,i+locrad_future+1,1)
            combins_future     = i+locrad_future
            
            # Wrap around indices outside the bounds
            combins_future[np.where(combins_future < 0)] += D
            combins_future[np.where(combins_future >= D)] -= D
            
            # What neighbouring blocks does this state depend on? Depends on
            # the localization radius
            # combins_current = np.arange(i-locrad_current,i+locrad_current+1,1)
            combins_current    = i+locrad_current
            
            # Wrap around indices outside the bounds
            combins_current[np.where(combins_current < 0)] += D
            combins_current[np.where(combins_current >= D)] -= D
            
            # -----------------------------------------------------------------
            # Start with the nonmonotone part
            # -----------------------------------------------------------------
             
            # Create an empty list for this map component's nonmonotone terms
            nonmonotone.append([])
            
            # Add the constant term
            nonmonotone[-1].append([])
                        
            # Go through all localization terms
            # Add dependencies on future states
            for c in combins_future:
                
                # Add terms in polynomial order
                for o in np.arange(1,order_nonmon+1,1):
                
                    # Add term for the current polynomial order
                    if o > 1:
                        nonmonotone[-1].append([c]*o+['HF'])
                    else:
                        nonmonotone[-1].append([c]*o)
                        
            # Go through all localization terms
            # Add dependencies on current states
            for c in combins_current:
                
                if c < i:
                
                    # Add terms in polynomial order
                    for o in np.arange(1,order_nonmon+1,1):
                    
                        # Add term for the current polynomial order
                        if o > 1:
                            nonmonotone[-1].append([c+D]*o+['HF'])
                        else:
                            nonmonotone[-1].append([c+D]*o)
        
            # -----------------------------------------------------------------
            # Now do the monotone part
            # -----------------------------------------------------------------
        
            # Create an empty list for this map component's monotone terms
            monotone.append([])
            
            # If it's a linear term, add it as is
            if order_mon == 1:
                
                # Add the linear term
                monotone[-1].append([i+D])
                
            # Nonlinear monotone terms are realized as a linear term plus
            # integrated radial basis functions
            else:
            
                # Add left edge term
                monotone[-1].append('LET '+str(i+D))
                
                # Add integrated RBFs to taste
                for o in np.arange(1,order_mon,1):
                
                    # Add term for the current polynomial order
                    monotone[-1].append('iRBF '+str(i+D))
                    
                # Add left edge term
                monotone[-1].append('RET '+str(i+D))
        
        return nonmonotone,monotone
    
    # For the spinup, we use a stochastic EnKF
    def stochastic_EnKF(X,y,R,H):
        
        """
        This function implements a stochastic EnKF update. It requires the follow-
        ing variables:
            
            X       - an N-by-D array of samples, where N is the ensemble size, and
                      D is the dimensionality of state space
            y       - a vector of length O containing the observations made
            R       - the O-by-O observation error covariance matrix
            H       - an O-by-D observation operator
        """
        
        # Get the number of particles
        N       = X.shape[0]
        
        # Get the state covariance matrix
        C       = np.cov(X.T)   # We need the transpose of X
        
        # Calculate the Kalman gain
        K       = np.linalg.multi_dot((
            C,
            H.T,
            np.linalg.inv(
                np.linalg.multi_dot((
                    H,
                    C,
                    H.T)) + R)))
        
        # Draw observation error realizations
        v       = scipy.stats.multivariate_normal.rvs(
            mean    = np.zeros(R.shape[0]),
            cov     = R,
            size    = N)
        
        # Perturb the observations
        obs     = y[np.newaxis,:] + v
        
        # Apply the stochastic Kalman update
        X += np.dot(
            K,
            obs.T - np.dot(H,X.T) ).T
        
        return X
        
    # -------------------------------------------------------------------------
    # Set up exercise
    # -------------------------------------------------------------------------
    
    # Define problem dimensions
    O                   = 20 # Observation space dimensions
    D                   = 40 # State space dimensions
    
    # Set up time
    T_spinup            = 100   # EnKF spinup period
    T                   = 250   # Assimilation time steps
    dt                  = 0.4   # Time step length
    
    # Ensemble size
    Ns                  = [100,50]#[500,375,250,175,135,100,50]
    
    # Observation error
    obs_sd              = np.sqrt(0.5)
    R                   = np.identity(O)*0.5
    obsindices          = np.arange(D)[::2] # We observe every second state
    
    # Inflation factor
    gammas              = [0.,0.025,0.05,0.1,0.2]
    
    orders              = [[5,5],[5,1],[3,3],[3,1],[2,2],[2,1],[1,1]]
    
    random_seeds        = [0,1,2,3,4,5,6,7,8,9]
    
    lambdas_filter      = [0.,1.,3.,7.,10.,15.]
    lambdas_smoother    = [0.,1.,3.,7.,10.,15.]
    
    #%%
    
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
    
    #%%

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
        
    #%%
    
    # =========================================================================
    # Start simulating
    # =========================================================================
    
    for random_seed in random_seeds:
    
        # =========================================================================
        # Create or load observations and synthetic reference
        # =========================================================================
    
        # Reset the random seed
        np.random.seed(random_seed)
        
        # If we haven't precalculated observations and truth, do so.
        if "synthetic_truth_L96_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)) or \
           "observations_L96_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)):
        
            # Initialize the synthetic truth
            synthetic_truth     = np.zeros((T+T_spinup,D))
            synthetic_truth[0,:] = scipy.stats.multivariate_normal.rvs(
                mean            = np.zeros(D),
                cov             = np.identity(D),
                size            = 1)
            
            # Initialized the synthetic observations
            observations        = copy.copy(synthetic_truth[:,obsindices])
            observations[0,:]   += scipy.stats.norm.rvs(
                loc             = 0,
                scale           = obs_sd,
                size            = O)
            
            # Go through all timesteps
            for t in np.arange(1,T+T_spinup,1):
                
                # Make a Lorenz forecast
                synthetic_truth[t,:] = rk4(
                    Z           = copy.copy(synthetic_truth[t-1,:][np.newaxis,:]),
                    fun         = lorenz_dynamics_96,
                    t           = 0,
                    dt          = 0.01,
                    nt          = 40)[0,:]
                
                # Calculate the synthetic observation
                observations[t,:]   = copy.copy(synthetic_truth[t,obsindices])
                observations[t,:]   += scipy.stats.multivariate_normal.rvs(
                    mean        = np.zeros(O),
                    cov         = R,
                    size        = 1)
        
            # Save the synthetic truth and observations
            pickle.dump(synthetic_truth,    open("synthetic_truth_L96_RS="+str(random_seed)+".p","wb"))
            pickle.dump(observations,       open("observations_L96_RS="+str(random_seed)+".p","wb"))
        
        else:
            
            # If we already have synthetic observations and truth in storage, retrieve them
            synthetic_truth     = pickle.load(open("synthetic_truth_L96_RS="+str(random_seed)+".p","rb"))
            observations        = pickle.load(open("observations_L96_RS="+str(random_seed)+".p","rb"))
            
        # Go through all ensemble sizes
        for ni,N in enumerate(Ns):
            
            # Go through all maximum map component orders
            for oi,order in enumerate(orders):
                
                # Unpack the orders
                order_filter        = order[0]
                order_smoother      = order[1]
                
                # Create a storage matrix for the RMSEs
                mat_RMSE            = np.zeros((len(gammas),len(lambdas_filter)))*np.nan
            
                # Go through all inflation factors
                for gi,gamma in enumerate(gammas):
                
                    # Go through all l2 regularization factors
                    for li,lmbda in enumerate(lambdas_filter):
                    
                        # Reset the random seed
                        np.random.seed(random_seed)
                        
                        # -------------------------------------------------
                        # Prepare the stochastic map filtering
                        # -------------------------------------------------
                        
                        # Attempt filtering; this ensures that the loop does
                        # not break if one operation becomes unstable
                        try:
                    
                            # Load filtering results from storage, if available
                            if "TM_filter_N="+str(N).zfill(4)+"_RS="+str(random_seed)+"_order="+str(order_filter)+"_gamma="+str(gamma)+"_lambda="+str(lmbda)+".p" in list(os.listdir(root_directory)):
                                
                                # If we compare linear and nonlinear smoothers,
                                # we have already filtered in the second pass
                                # In that case, just do nothing and skip to
                                # filtering
                                
                                # Have we already filtered before?
                                second_pass     = True
                                
                                pass
                                
                            # If no filtering results are available, simulate
                            else:
                                
                                # Have we already filtered before?
                                second_pass     = False
                                
                                print("Now filtering: order="+str(order)+" | N="+str(N)+" | lambda="+str(lmbda)+" | gamma="+str(gamma)+" | RS="+str(random_seed),end="")
                                
                                file = open("log.txt","a")
                                file.write("Now filtering: order="+str(order)+" | N="+str(N)+" | lambda="+str(lmbda)+" | gamma="+str(gamma)+" | RS="+str(random_seed))
                                file.close()
                                
                                # =========================================
                                # Simulate the spinup period
                                # =========================================
                                
                                # If spinup results are available, load them                                    
                                if "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p" in list(os.listdir(root_directory)):
                                    
                                    # Load the spinup results
                                    Z0   = pickle.load(
                                        open(
                                            "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p",
                                            "rb"))
                                    
                                # If not, simulate them
                                else:
                            
                                    # Initiate particles from a standard Gaussian
                                    Z0          = np.zeros((T_spinup+1,N,D))
                                    Z0[0,...]   = scipy.stats.norm.rvs(size=(N,D))
                                    
                                    # Create the observation operator
                                    H           = np.zeros((O,D))
                                    for i,o in enumerate(obsindices):
                                        H[i,o]  = 1
                                        
                                    # Go through the spinup period
                                    for t in np.arange(0,T_spinup+1,1):
                                        
                                        # Stochastic EnKF update
                                        Z0[t,...] = stochastic_EnKF(
                                            X       = copy.copy(Z0[t,...]),
                                            y       = copy.copy(observations[t,:]),
                                            R       = R,
                                            H       = H)

                                        # After the analysis step, make a forecast to the next timestep
                                        if t < T_spinup:
                                            
                                            # Make a Lorenz forecast
                                            Z0[t+1,:,:] = rk4(
                                                Z           = copy.copy(Z0[t,:,:]),
                                                fun         = lorenz_dynamics_96,
                                                t           = 0,
                                                dt          = dt/40,
                                                nt          = 40)

                                    # Store the spinup samples
                                    pickle.dump(Z0,
                                        open(
                                            "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p",
                                            "wb"))
                                
                                #%%
                                
                                # =========================================
                                # Create permutations for the sparse filter updates
                                # =========================================

                                # Create a list for the permutations
                                permutations    = []
                                
                                # Go through all observed dimensions
                                for d in obsindices:
                                    
                                    # Initialize this permutation
                                    perm    = []
                                    
                                    # Append central state
                                    perm    .append(d)
                                    
                                    # Expand localization radius
                                    for lr in range(locrad_filter):
                                        
                                        # Append value to the left
                                        perm.append(d-lr-1)
                                        
                                        # Append value to the right
                                        perm.append(d+lr+1)
                                        
                                    # Convert list to array
                                    perm    = np.asarray(perm, dtype = int)
                                    
                                    # Wrap around indices
                                    perm[perm < 0]  += D
                                    perm[perm >= D] -= D
                                    
                                    # Append to permutation list
                                    permutations.append(copy.copy(perm))
                                        
                                
                                # =========================================
                                # Prepare the stochastic map filtering
                                # =========================================
                                
                                # Reset the random seed
                                np.random.seed(random_seed)
                                
                                # Initialize the filtering samples from the
                                # spinup dataset
                                Z_a         = np.zeros((T,N,D))
                                Z_a[0,:,:]  = copy.copy(Z0[-1,...])
                                
                                # Then delete the spinup data set; we can 
                                # re-load it later
                                del Z0
                                
                                # Initialize the array for the forecast
                                Z_f         = copy.copy(Z_a)
                                
                                # Initialize the list for the RMSE and CRPS
                                RMSE_list           = []
                                CRPS_list           = []
                                
                                # Get the filtering transport map components
                                nonmonotone, monotone = get_transport_map_functions_filter(
                                    D                   = D,
                                    localization_radius = locrad_filter,
                                    order_nonmon        = order_filter, 
                                    order_mon           = order_filter,
                                    obs_offset          = 1)    
                                
                                # If a previous transport map object exists,
                                # delete it to avoid weirdness
                                if "tm" in globals():
                                    del tm
                            
                                # Parameterize the transport map
                                tm     = transport_map(
                                    monotone                = monotone,
                                    nonmonotone             = nonmonotone,
                                    X                       = np.random.uniform(size=(N,len(permutations[0])+1)), # Dummy input
                                    polynomial_type         = "hermite function",
                                    monotonicity            = "separable monotonicity",
                                    regularization          = "l2",
                                    regularization_lambda   = lmbda,
                                    verbose                 = False)
                            
                                # Start time measurement for the filtering
                                time_begin  = time.time()
                            
                                # Go through the entire assimilation period
                                for t in np.arange(0,T,1):
                                    
                                    # Copy the forecast
                                    Z_a[t,...]  = copy.copy(Z_f[t,...])
                                        
                                    # Assimilate the observations one at a time
                                    # Go through each permutation
                                    for idx,perm in enumerate(permutations):
                                        
                                        # Which state are we observing?
                                        obsidx      = obsindices[idx]
                                        
                                        # Do we use inflation?
                                        if gamma > 0: # Yes
                                        
                                            # Inflate the ensemble
                                            Z_a_inflated = copy.copy(Z_a[t,:,:])
                                            Z_a_inflated = np.sqrt(1+gamma)*(Z_a_inflated-np.mean(Z_a_inflated,axis=0)) + np.mean(Z_a_inflated,axis=0)
                                            
                                            # Simulate observations
                                            Y_sim_inflated = copy.copy(Z_a_inflated[:,obsidx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = N)
                                                
                                            # Create the inflated map input
                                            map_input_inflated = copy.copy(np.column_stack((
                                                Y_sim_inflated[:,np.newaxis],   # First dimension: simulated observation
                                                Z_a_inflated[:,perm])))         # Next perm dimensions: permutated states
                                            
                                            # Reset the transport map with the inflated values
                                            tm.reset(map_input_inflated)
                                            
                                            # Simulate non-inflated observations
                                            Y_sim = copy.copy(Z_a[t,:,obsidx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = N)
                                                
                                            # Create the uninflated map input
                                            map_input = copy.copy(np.column_stack((
                                                Y_sim,                 # First dimension: simulated observation
                                                Z_a[t,:,:][:,perm])))           # Next perm dimensions: permutated states
                                            
                                        else: # No
                                            
                                            # Simulate observations
                                            Y_sim = copy.copy(Z_a[t,:,obsidx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = N)
                                                
                                            # Create the uninflated map input
                                            map_input = copy.copy(np.column_stack((
                                                Y_sim,                 # First dimension: simulated observation
                                                Z_a[t,:,:][:,perm])))           # Next perm dimensions: permutated states
                                                
                                            # Reset the transport map
                                            tm.reset(map_input)
                                        
                                        # Start optimizing the transport map
                                        tm.optimize()

                                        # ---------------------------------
                                        # Composite map
                                        # ---------------------------------
                                        
                                        # The composite map uses reference
                                        # samples from the forward map
                                        norm_samples    = tm.map(map_input)
                                        
                                        # Create an array with replicates of 
                                        # the observations
                                        X_star          = np.repeat(
                                            a       = observations[T_spinup+t,idx].reshape((1,1)),
                                            repeats = N, 
                                            axis    = 0)
                                        
                                        # Apply the inverse map
                                        ret = tm.inverse_map(
                                            X_star      = X_star,
                                            Z           = norm_samples)
                                        
                                        # Ssave the result in the analysis array
                                        Z_a[t,...][:,permutations[idx]]  = copy.copy(ret)
                                        
                                    # Calculate RMSE
                                    RMSE = (np.mean(Z_a[t,...],axis=0) - synthetic_truth[T_spinup+t,:])**2
                                    RMSE = np.mean(RMSE)
                                    RMSE = np.sqrt(RMSE)
                                    RMSE_list.append(RMSE)
                                    
                                    # Calculate CRPS
                                    CRPS = crps_ens(
                                        Z_a[t,...],
                                        synthetic_truth[T_spinup+t,:])
                                    CRPS_list.append(CRPS)
                                    
                                    # After the analysis step, make a forecast to the next timestep
                                    if t < T-1:
                                        
                                        # Make a Lorenz forecast
                                        Z_f[t+1,:,:] = rk4(
                                            Z           = copy.copy(Z_a[t,:,:]),
                                            fun         = lorenz_dynamics_96,#lorenz_dynamics,
                                            t           = 0,
                                            dt          = dt/40,
                                            nt          = 40)
                                
                                # Stop the clock
                                time_end    = time.time()
                                
                                # Add the mean ensemble RMSE to the output print
                                print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                                
                                file = open("log.txt","a")
                                file.write(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                                file.write("\n")
                                file.close()
                                
                                # Store the results in the output dictionary
                                output_dictionary                       = {}
                                output_dictionary['Z_a_q0025']          = np.quantile(Z_a,axis=1,q=0.025)
                                output_dictionary['Z_a_q0050']          = np.quantile(Z_a,axis=1,q=0.050)
                                output_dictionary['Z_a_q0250']          = np.quantile(Z_a,axis=1,q=0.250)
                                output_dictionary['Z_a_q0500']          = np.quantile(Z_a,axis=1,q=0.500)
                                output_dictionary['Z_a_q0750']          = np.quantile(Z_a,axis=1,q=0.750)
                                output_dictionary['Z_a_q0950']          = np.quantile(Z_a,axis=1,q=0.950)
                                output_dictionary['Z_a_q0975']          = np.quantile(Z_a,axis=1,q=0.975)
                                output_dictionary['RMSE_list']          = RMSE_list
                                output_dictionary['CRPS_list']          = CRPS_list
                                output_dictionary['duration']           = time_end-time_begin
                                output_dictionary['coverage']           = [[1 if synthetic_truth[T_spinup+t,d] >= output_dictionary['Z_a_q0025'][t,d] and synthetic_truth[T_spinup+t,d] <= output_dictionary['Z_a_q0975'][t,d] else 0 for d in range(D)] for t in range(T)]
                                output_dictionary['spread']             = [np.sqrt(np.mean(np.std(Z_a[t,:,:],axis=0)**2)) for t in range(T)]

                                # Write the result dictionary to a file
                                pickle.dump(output_dictionary,  open(
                                    "TM_filter_N="+str(N).zfill(4)+"_RS="+str(random_seed)+"_order="+str(order_filter)+"_gamma="+str(gamma)+"_lambda="+str(lmbda)+".p",'wb'))
                       
                                # Store the average RMSE of this run
                                mat_RMSE[gi,li]     = np.mean(RMSE_list)
                                
                                if np.nanmin(mat_RMSE) == np.mean(RMSE_list):
                                    
                                    # Store the Z_a and Z_f
                                    pickle.dump(Z_a,  open(
                                        "Z_a_N="+str(N).zfill(4)+"_opt.p",'wb'))
                                    pickle.dump(Z_f,  open(
                                        "Z_f_N="+str(N).zfill(4)+"_opt.p",'wb'))

                                    
                        
                        # Did the filtering fail? Oops.
                        except:
                            
                            # raise Exception
                            
                            file = open("log.txt","a")
                            file.write(" | FAILED")
                            file.write("\n")
                            file.close()
                            
                            pass
                        
                #%%
                
                # Try smoothing; skip to the next operation if something failed
                try:
                
                    # =============================================================
                    # Find the best filtering distributions
                    # =============================================================
                    
                    # Have we already filtered before?
                    if not second_pass:
                    
                        # Go through the RMSE mat, find the optimal RMSEs
                        optidx  = np.where(mat_RMSE == np.nanmin(mat_RMSE))
                        optgi   = optidx[0][0]
                        optli   = optidx[1][0]
                        
                        # Load these samples
                        Z_a     = copy.copy(pickle.load(open(
                            "Z_a_N="+str(N).zfill(4)+"_opt.p",'rb')))
                        Z_f     = copy.copy(pickle.load(open(
                            "Z_f_N="+str(N).zfill(4)+"_opt.p",'rb')))
    
                        # Update the log with this information
                        file = open("log.txt","a")
                        file.write("\n")
                        file.write("Best filtering combination: gamma="+str(gammas[optgi])+" | lambda="+str(lambdas_filter[optli])+" | RMSE="+"{:.3f}".format(np.nanmin(mat_RMSE)))
                        file.write("\n")
                        file.close()
                        
                    #%%
    
                    # =============================================================
                    # Prepare the backward smoothing
                    # =============================================================
    
                    # Go through all l2 regularization factors
                    for li,lmbda in enumerate(lambdas_smoother):
                    
                        # Reset the random seed
                        np.random.seed(random_seed)
                              
                        print("Now smoothing: order="+str(order)+" | N="+str(N)+" | lambda="+str(lmbda)+" | RS="+str(random_seed),end="")
                
                        file = open("log.txt","a")
                        file.write("Now smoothing: order="+str(order)+" | N="+str(N)+" | lambda="+str(lmbda)+" | RS="+str(random_seed))
                        file.close()
                
                        # Reset the random seed
                        np.random.seed(random_seed)
                        
                        try:
                                     
                            # Initialize the array for the smoothing
                            Z_s         = copy.copy(Z_a)
                            
                            # Initialize the list for the RMSE
                            RMSE_list           = []
                            
                            # Initialize the list for the CRPS
                            CRPS_list           = []
                            
                            # Calculate RMSE for the final timestep
                            RMSE = (np.mean(Z_s[-1,...],axis=0) - synthetic_truth[-1,:])**2
                            RMSE = np.mean(RMSE)
                            RMSE = np.sqrt(RMSE)
                            RMSE_list.append(RMSE)
                            
                            # Calculate CRPS for the final timestep
                            CRPS = crps_ens(
                                Z_s[-1,...],
                                synthetic_truth[-1,:])
                            CRPS_list.append(CRPS)
                            
                            # Create the real transport map
                            nonmonotone, monotone = get_transport_map_functions_smoother(
                                D                   = D,
                                locrad_future       = locrad_smoother_future,
                                locrad_current      = locrad_smoother_current,
                                order_nonmon        = order_smoother, 
                                order_mon           = order_smoother)
                            
                            # Delete any pre-existing map objects
                            if "tm" in globals():
                                del tm
                        
                            # Parameterize the transport map
                            tm     = transport_map(
                                monotone                = monotone,
                                nonmonotone             = nonmonotone,
                                X                       = np.random.uniform(size=(N,int(2*D))), 
                                polynomial_type         = "probabilist's hermite",  
                                monotonicity            = "separable monotonicity",
                                regularization          = "l2",
                                regularization_lambda   = lmbda,
                                verbose                 = False)    
                        
                            # Start the clock for the smoothing pass
                            time_begin      = time.time()
                        
                            # Start the backwards pass
                            for t in np.arange(T-2,-1,-1):
                                
                                # Create the uninflated map input
                                map_input = copy.copy(np.column_stack((
                                    Z_f[t+1,:,:],       # First D dimensions: future states
                                    Z_a[t,:,:])))       # Next D dimensions: current states
                                    
                                # Reset the transport map with the new values
                                tm.reset(map_input)
                                
                                # Start optimizing the transport map
                                tm.optimize()
                                
                                # Once the map is optimized, use it to convert the samples to samples from
                                # the reference distribution X
                                norm_samples    = tm.map(map_input)
                                
                                # Create an array with the observations
                                X_star          = copy.copy(Z_s[t+1,:,:])
                                
                                # Apply the inverse map to obtain samples from (Y,Z_a)
                                ret = tm.inverse_map(
                                    X_star      = X_star,
                                    Z           = norm_samples)
                                
                                # Ssave the result in the analysis array
                                Z_s[t,...]  = copy.copy(ret)
                            
                                # Calculate RMSE
                                RMSE = (np.mean(Z_s[t,...],axis=0) - synthetic_truth[T_spinup+t,:])**2
                                RMSE = np.mean(RMSE)
                                RMSE = np.sqrt(RMSE)
                                RMSE_list.append(RMSE)
                                
                                # Calculate CRPS
                                CRPS = crps_ens(
                                    Z_s[t,...],
                                    synthetic_truth[T_spinup+t,:])
                                CRPS_list.append(CRPS)
                            
                            # Stop the clock
                            time_end    = time.time()
                            
                            # Add the mean ensemble RMSE to the output print
                            print(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                            
                            file = open("log.txt","a")
                            file.write(" | RMSE="+"{:.3f}".format(np.mean(RMSE_list)))
                            file.write("\n")
                            file.close()
                        
                            # Store the results in the output dictionary
                            output_dictionary                       = {}
                            output_dictionary['Z_s_q0025']          = np.quantile(Z_s,axis=1,q=0.025)
                            output_dictionary['Z_s_q0050']          = np.quantile(Z_s,axis=1,q=0.050)
                            output_dictionary['Z_s_q0250']          = np.quantile(Z_s,axis=1,q=0.250)
                            output_dictionary['Z_s_q0500']          = np.quantile(Z_s,axis=1,q=0.500)
                            output_dictionary['Z_s_q0750']          = np.quantile(Z_s,axis=1,q=0.750)
                            output_dictionary['Z_s_q0950']          = np.quantile(Z_s,axis=1,q=0.950)
                            output_dictionary['Z_s_q0975']          = np.quantile(Z_s,axis=1,q=0.975)
                            output_dictionary['RMSE_list']          = RMSE_list
                            output_dictionary['CRPS_list']          = CRPS_list
                            output_dictionary['duration']           = time_end-time_begin
                            output_dictionary['coverage']           = [[1 if synthetic_truth[T_spinup+t,d] >= output_dictionary['Z_s_q0025'][t,d] and synthetic_truth[T_spinup+t,d] <= output_dictionary['Z_s_q0975'][t,d] else 0 for d in range(D)] for t in range(T)]
                            output_dictionary['spread']             = [np.sqrt(np.mean(np.std(Z_s[t,:,:],axis=0)**2)) for t in range(T)]
                            
                            # Write the result dictionary to a file
                            pickle.dump(output_dictionary,  open(
                                'TM_smoother_N='+str(N).zfill(4)+"_RS="+str(random_seed)+"_order_filter="+str(order_filter)+"_order_smoother="+str(order_smoother)+"_lambda="+str(lmbda)+'.p','wb'))
    
                        # Smoothing failed? Oops.
                        except:
                            
                            pass
                        
                        
                # Did the smoothing fail? Oops.
                except:
                    
                    file = open("log.txt","a")
                    file.write("--- SMOOTHING OPERATION FAILED ---")
                    file.write("\n")
                    file.close()
                    
                    pass