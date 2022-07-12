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
    
    # Lorenz-63 dynamics
    def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
        
        if len(Z.shape) == 1: # Only one particle
        
            dZ1ds   = - sigma*Z[0] + sigma*Z[1]
            dZ2ds   = - Z[0]*Z[2] + rho*Z[0] - Z[1]
            dZ3ds   = Z[0]*Z[1] - beta*Z[2]
            
            dyn     = np.asarray([dZ1ds, dZ2ds, dZ3ds])
            
        else:
            
            dZ1ds   = - sigma*Z[...,0] + sigma*Z[...,1]
            dZ2ds   = - Z[...,0]*Z[...,2] + rho*Z[...,0] - Z[...,1]
            dZ3ds   = Z[...,0]*Z[...,1] - beta*Z[...,2]
    
            dyn     = np.column_stack((dZ1ds, dZ2ds, dZ3ds))
    
        return dyn
    
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
    O                   = 3 # Observation space dimensions
    D                   = 3 # State space dimensions
    
    # Ensemble size
    Ns                  = [1000,750,500,375,250,175,100,50]
    
    # Set up time
    T                   = 1000  # Full time series length
    T_spinup            = 1000  # EnKF spinup period
    
    # Time step length
    dt                  = 0.1   # Time step length
    dti                 = 0.05  # Time step increment
    
    # Observation error
    obs_sd              = 2
    R                   = np.identity(O)*obs_sd**2
    
    # L2 regularization factor
    lambdas_filter      = [0.,0.25,0.5,1.0,1.5,2.0]
    lambdas_smoother    = [0.,0.025,0.05,0.1,0.15,0.2,0.25,0.5,1.0]
    
    # Inflation factor
    gammas              = [0.,0.05,0.1,0.2,0.3]
    
    # Maximum polynomial order for EnTF / EnTS
    orders              = [[5,5],[5,1],[3,3],[3,1],[2,2],[2,1],[1,1]]
    
    # Random seeds for repeat simulations
    random_seeds        = [0,1,2,3,4,5,6,7,8,9]
        
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
        if "synthetic_truth_L63_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)) or \
           "observations_L63_RS="+str(random_seed)+".p" not in list(os.listdir(root_directory)):
        
            # Create the synthetic reference
            synthetic_truth         = np.zeros((T_spinup+T,1,D))
            synthetic_truth[0,0,:]  = scipy.stats.norm.rvs(size=3)
            
            for t in np.arange(0,T_spinup+T-1,1):
                 
                # Make a Lorenz forecast
                synthetic_truth[t+1,:,:] = rk4(
                    Z           = copy.copy(synthetic_truth[t,:,:]),
                    fun         = lorenz_dynamics,
                    t           = 0,
                    dt          = dti,
                    nt          = int(dt/dti))
                
            # Remove the unnecessary particle index
            synthetic_truth     = synthetic_truth[:,0,:]
                
            # Create observations
            observations        = copy.copy(synthetic_truth) + scipy.stats.norm.rvs(scale = obs_sd, size = synthetic_truth.shape)
            
        
            # Save the synthetic truth and observations
            pickle.dump(synthetic_truth,    open("synthetic_truth_L63_RS="+str(random_seed)+".p","wb"))
            pickle.dump(observations,       open("observations_L63_RS="+str(random_seed)+".p","wb"))
        
        else:
            
            # If we already have synthetic observations and truth in storage, retrieve them
            synthetic_truth     = pickle.load(open("synthetic_truth_L63_RS="+str(random_seed)+".p","rb"))
            observations        = pickle.load(open("observations_L63_RS="+str(random_seed)+".p","rb"))
            
            
        # Go through all ensemble sizes
        for ni,N in enumerate(Ns):
            
            # Go through all maximum map component orders
            for oi,order in enumerate(orders):
                
                # Unpack the orders
                order_filter        = order[0]
                order_smoother      = order[1]
                
                # Create a storage matrix for the RMSEs
                mat_RMSE            = np.zeros((len(gammas),len(lambdas_filter)))*np.nan
                
                # Extract the maximum polynomial order for the EnTF and EnTS
                order_filter    = order[0]
                order_smoother  = order[1]
                
                # Define the map component functions
                if order_filter == 1: # Map is linear
                    
                    nonmonotone_filter  = [
                        [[],[0]],
                        [[],[1]],
                        [[],[1],[2]]]
                
                    monotone_filter     = [
                        [[1]],
                        [[2]],
                        [[3]]]
                    
                else: # Map is nonlinear
                    
                    monotone_filter     = [
                        ['LET 1']+['iRBF 1']*(order_filter-1)+['RET 1'],
                        [[2]],
                        [[3]]]
                
                    nonmonotone_filter  = [
                        [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,order_filter+1,1)],
                        [[],[1]]+[[1]*od+['HF'] for od in np.arange(1,order_filter+1,1)],
                        [[],[1]]+[[1]*od+['HF'] for od in np.arange(1,order_filter+1,1)]+[[2]]+[[2]*od+['HF'] for od in np.arange(1,order_filter+1,1)]]
                
            
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
                                    H           = np.identity(O)
                                        
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
                                                fun         = lorenz_dynamics,
                                                t           = 0,
                                                dt          = dti,
                                                nt          = int(dt/dti))
                                            
                                    # Store the spinup samples
                                    pickle.dump(Z0,
                                        open(
                                            "Z0_N="+str(N).zfill(4)+"_RS="+str(random_seed)+".p",
                                            "wb"))
                                
                                #%%
                                
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
                                
                                # If a previous transport map object exists,
                                # delete it to avoid weirdness
                                if "tm" in globals():
                                    del tm
                            
                                # Parameterize the transport map
                                tm     = transport_map(
                                    monotone                = monotone_filter,
                                    nonmonotone             = nonmonotone_filter,
                                    X                       = np.random.uniform(size=(N,D+1)), # Dummy input
                                    polynomial_type         = "hermite function",
                                    monotonicity            = "separable monotonicity",
                                    regularization          = "l2",
                                    regularization_lambda   = lmbda,
                                    verbose                 = False)
                            
                                # Start time measurement for the filtering
                                time_begin  = time.time()
                            
                                # Start the filtering
                                for t in np.arange(0,T,1):

                                    # Copy the forecast into the analysis matrix
                                    Z_a[t,:,:]  = copy.copy(Z_f[t,:,:])
                                
                                    # Assimilate the observations one at a time
                                    for idx,perm in enumerate([[0,1,2],[1,0,2],[2,1,0]]):
                                        
                                        # If we use inflation, use inflated samples
                                        # for the map training
                                        if gamma != 0:
                                        
                                            # Inflate the ensemble
                                            Z_a_inflated = copy.copy(Z_a[t,:,:])
                                            Z_a_inflated = np.sqrt(1+gamma)*(Z_a_inflated-np.mean(Z_a_inflated,axis=0)) + np.mean(Z_a_inflated,axis=0)
                                            
                                            # Simulate observations
                                            Y_sim = copy.copy(Z_a_inflated[:,idx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = Z_a_inflated[:,idx].shape)
                                                
                                            # Create the inflated map input
                                            map_input_inflated = copy.copy(np.column_stack((
                                                Y_sim[:,np.newaxis],   # First dimension: simulated observations
                                                Z_a_inflated[:,perm])))         # Next D dimensions: predicted states
                                            
                                            # Simulate observations
                                            Y_sim = copy.copy(Z_a[t,:,idx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = Z_a[t,:,idx].shape)
                                                
                                            # Create the uninflated map input
                                            map_input = copy.copy(np.column_stack((
                                                Y_sim[:,np.newaxis],   # First dimension: simulated observations
                                                Z_a[t,:,:][:,perm])))           # Next D dimensions: predicted states
                                                
                                            # Reset the transport map with the new values
                                            tm.reset(copy.copy(map_input_inflated))
                                            
                                        # If we do not use inflation
                                        else:
                                            
                                            # Simulate observations
                                            Y_sim = copy.copy(Z_a[t,:,:][:,idx]) + \
                                                scipy.stats.norm.rvs(
                                                    loc     = 0,
                                                    scale   = obs_sd,
                                                    size    = Z_a[t,:,:][:,idx].shape)
                                                
                                            # Create the uninflated map input
                                            map_input = copy.copy(np.column_stack((
                                                Y_sim[:,np.newaxis],   # First dimension: simulated observation
                                                Z_a[t,:,:][:,perm])))           # Next D dimensions: predicted states
                                                
                                            # Reset the transport map with the new values
                                            tm.reset(copy.copy(map_input))
                                        
                                        # Start optimizing the transport map
                                        tm.optimize()
        
                                        # ------------------------------------
                                        # Composite map
                                        # ------------------------------------
                                        
                                        # For the composite map, we use reference samples
                                        # from a forward filtering pass; extract those
                                        # samples
                                        norm_samples = tm.map(copy.copy(map_input))
                                        
                                        # Create an array with replicated of the observations
                                        X_star = np.repeat(
                                            a       = observations[T_spinup+t,idx].reshape((1,1)),
                                            repeats = N, 
                                            axis    = 0)
                                        
                                        # Apply the inverse map
                                        ret = tm.inverse_map(
                                            X_star      = X_star,
                                            Z           = norm_samples)
                                        
                                        # Undo the permutation of the states
                                        ret = ret[:,perm]
                                        
                                        # Save the result in the analysis array
                                        Z_a[t,...]  = copy.copy(ret)
                                        
                                    # Calculate ensemble mean RMSE
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
                                            fun         = lorenz_dynamics,
                                            t           = 0,
                                            dt          = dti,
                                            nt          = int(dt/dti))
                                
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
                
                # Define the map component functions
                if order_smoother == 1: # Linear smoother
                    
                    nonmonotone_BWS  = [
                        [[],[0],[1],[2]],
                        [[],[0],[1],[2],[3]],
                        [[],[0],[1],[2],[3],[4]]]
                
                    monotone_BWS     = [
                        [[3]],
                        [[4]],
                        [[5]]]
                    
                else: # Nonlinear smoother
                    
                    monotone_BWS     = [
                        [[3]],
                        [[4]],
                        [[5]]]
                    
                    nonmonotone_BWS  = [
                        [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,order_smoother+1,1)],
                        [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[3]]+[[3]*od+['HF'] for od in np.arange(1,order_smoother+1,1)],
                        [[],[0]]+[[0]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[1]]+[[1]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[2]]+[[2]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[3]]+[[3]*od+['HF'] for od in np.arange(1,order_smoother+1,1)] + [[4]]+[[4]*od+['HF'] for od in np.arange(1,order_smoother+1,1)]]

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
                        
                        # Delete any pre-existing map objects
                        if "tm" in globals():
                            del tm
                    
                        # Parameterize the transport map
                        tm     = transport_map(
                            monotone                = monotone_BWS,
                            nonmonotone             = nonmonotone_BWS,
                            X                       = np.random.uniform(size=(N,int(2*D))), 
                            polynomial_type         = "probabilist's hermite",  
                            monotonicity            = "separable monotonicity",
                            regularization          = "l2",
                            regularization_lambda   = lmbda,
                            verbose                 = False)    
                    
                        # Start the clock for the smoothing pass
                        time_begin      = time.time()
                    
                        # Now start with hybrid smoothing
                        for t in range(T-2,-1,-1):
                            
                            # Create input for the single-pass backwards smoothing map
                            map_input = copy.copy(np.column_stack((
                                Z_f[t+1,:,:],    # First D dimensions: smoothed states
                                Z_a[t,...])))  # Next D dimensions: smoothed states
                        
                            # Reset the map
                            tm.reset(copy.copy(map_input))
                            
                            # Start optimizing the transport map
                            tm.optimize()
                            
                            # -----------------------------------------
                            # Composite map
                            # -----------------------------------------
                            
                            # We condition on previous smoothing samples
                            X_star = copy.copy(
                                Z_s[t+1,...])
                            
                            # The composite map uses reference samples 
                            # from an application of the forward map
                            norm_samples = tm.map(map_input)
                            
                            # Invert / Condition the map
                            ret = tm.inverse_map(
                                X_star      = X_star,
                                Z           = norm_samples) # Only necessary when heuristic is deactivated
                    
                            # Copy the results into the smoothing array
                            Z_s[t,...]    = copy.copy(ret)
                    
                            # Calculate RMSE
                            RMSE            = (np.mean(Z_s[t,...],axis=0) - synthetic_truth[T_spinup+t,:])**2
                            RMSE            = np.mean(RMSE)
                            RMSE            = np.sqrt(RMSE)
                            RMSE_list       .append(RMSE)
                            
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