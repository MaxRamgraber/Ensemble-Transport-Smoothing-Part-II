# Load in a number of libraries we will use
import numpy as np
import scipy.stats
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import time
import pickle
import os
import scipy
from transport_map_138 import *

# Clear previous figures and pickles
root_directory = os.path.dirname(os.path.realpath(__file__))
    
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
    
    # for n in range(N):
    #     X[n,:]  += np.dot(
    #         K,
    #         obs[n,:][:,np.newaxis] - np.dot(H,X[n,:][:,np.newaxis] ) )[:,0]
        
    return X
    
# -------------------------------------------------------------------------
# Set up exercise
# -------------------------------------------------------------------------

np.random.seed(0)

# Define problem dimensions
O                   = 20    # Observation space dimensions
D                   = 40    # State space dimensions

# Set up time
T                   = 1000  # EnKF spinup period
dt                  = 0.4   # Time step length
nt                  = 40

# Ensemble size
N                   = 1000

# Observation error
obs_sd              = np.sqrt(0.5)
R                   = np.identity(O)*0.5
obsindices          = np.arange(D)[::2] # We observe every second state

#%%

# =============================================================================
# Generate observations
# =============================================================================

# If we haven't precalculated observations and truth, do so.
if "synthetic_truth_L96.p" not in list(os.listdir(root_directory)) or \
   "observations_L96.p" not in list(os.listdir(root_directory)):
       
    print("Generating observations... ",end="")

    # Initialize the synthetic truth
    synthetic_truth     = np.zeros((T,D))
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
    for t in np.arange(1,T,1):
        
        # Make a Lorenz forecast
        synthetic_truth[t,:] = rk4(
            Z           = copy.copy(synthetic_truth[t-1,:][np.newaxis,:]),
            fun         = lorenz_dynamics_96,
            t           = 0,
            dt          = dt/nt,
            nt          = nt)[0,:]
        
        # Calculate the synthetic observation
        observations[t,:]   = copy.copy(synthetic_truth[t,obsindices])
        observations[t,:]   += scipy.stats.multivariate_normal.rvs(
            mean        = np.zeros(O),
            cov         = R,
            size        = 1)

    # Save the synthetic truth and observations
    pickle.dump(synthetic_truth,    open("synthetic_truth_L96.p","wb"))
    pickle.dump(observations,       open("observations_L96.p","wb"))
    
    print("Done.")

else:
    
    print("Loading observations... ",end="")
    
    # If we already have synthetic observations and truth in storage, retrieve them
    synthetic_truth     = pickle.load(open("synthetic_truth_L96.p","rb"))
    observations        = pickle.load(open("observations_L96.p","rb"))
    
    print("Done.")

#%%

# =============================================================================
# Start filtering
# =============================================================================

# Create the observation operator
H           = np.zeros((O,D))
for i,o in enumerate(obsindices):
    H[i,o]  = 1

# If we haven't precalculated observations and truth, do so.
if "Z_f_L96.p" not in list(os.listdir(root_directory)) or \
   "Z_a_L96.p" not in list(os.listdir(root_directory)):
       
    print("Generating filtering samples... ",end="")
    
    # Initiate particles from a standard Gaussian
    Z0 = scipy.stats.norm.rvs(size=(N,D))

    # Initialize forecast and analysis arrays
    Z_f         = np.zeros((T,N,D))
    Z_a         = np.zeros((T,N,D))
    Y           = np.zeros((T,N,O))
    Z_f[0,:,:]  = copy.copy(Z0)
    Z_a[0,:,:]  = copy.copy(Z0)
    
    # Go through the spinup period
    for t in np.arange(0,T,1):
        
        print(t)
        
        # Write this into the results
        Y[t,...]    = copy.copy(Z_f[t,...][:,obsindices]) + scipy.stats.norm.rvs(
            scale   = obs_sd,
            size    = Z_f[t,...][:,obsindices].shape)
        
        # Stochastic EnKF update
        Z_a[t,...] = stochastic_EnKF(
            X       = copy.copy(Z_f[t,...]),
            y       = copy.copy(observations[t,:]),
            R       = R,
            H       = H)
    
        # After the analysis step, make a forecast to the next timestep
        if t < T-1:
            
            # Make a Lorenz forecast
            Z_f[t+1,:,:] = rk4(
                Z           = copy.copy(Z_a[t,:,:]),
                fun         = lorenz_dynamics_96,
                t           = 0,
                dt          = dt/nt,
                nt          = nt)

    # Store the forecast samples
    pickle.dump(Z_f,
        open(
            "Z_f_L96.p",
            "wb"))
    
    # Store the analysis samples
    pickle.dump(Z_a,
        open(
            "Z_a_L96.p",
            "wb"))
    
    # Store the predictions samples
    pickle.dump(Y,
        open(
            "Y_L96.p",
            "wb"))
    
    print("Done.")
    
else:
    
    print("Loading filtering samples... ",end="")
    
    # Load the forecast samples
    Z_f = pickle.load(open(
        "Z_f_L96.p",
        "rb"))
    
    # Load the analysis samples
    Z_a = pickle.load(open(
        "Z_a_L96.p",
        "rb"))
    
    # Store the predictions samples
    Y = pickle.load(open(
        "Y_L96.p",
        "rb"))
    
    print("Done.")

#%%

for s in range(100):

    # We look at times backwards; doesn't really matter.
    t = -s-1    

    # =============================================================================
    # Post-process the results
    # =============================================================================
    
    if "ksd_L96_smoother_t="+str(t).zfill(3)+".p" not in list(os.listdir(root_directory)):
        
        # Whiten the samples
        map_input   = np.column_stack((
            copy.copy(Z_f[t,...]),
            copy.copy(Z_a[t-1,...]) ))
        map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
        map_input   /= np.std(map_input,axis=0)[np.newaxis,:]
        
        cov         = np.cov(map_input.T)
        chol        = scipy.linalg.cholesky(cov)
        prec        = scipy.linalg.inv(chol)
        map_input   = np.dot(map_input,prec)
    
        # for k in range(map_input.shape[-1]):
            
        #     # Get the sorted indices of these samples
        #     x   = copy.copy(map_input[:,k])
        #     idx = np.argsort(x)
            
        #     # Get the sorted quantiles
        #     q   = np.arange(len(x))/len(x) + 1/len(x)/2
        #     q   = q[idx]
            
        #     # Assume a perfect marginal Gaussianization, from an empirical cdf
        #     map_input[:,k]  = scipy.stats.norm.ppf(
        #         q    = q)
            
        
        # Pre-allocate an array for the KSDs
        ksds_s      = np.zeros((D,D+D))*np.nan
        ksds_marg_s = np.zeros((D+D))*np.nan
        
        # Marginal KSDs
        for idx in range(D+D):
            print(idx)
            ksds_marg_s[idx] = ksd(
                q   = map_input[::10,idx][:,np.newaxis],
                dq  = -map_input[::10,idx][:,np.newaxis])
        
        # Joint KSDs
        for row in np.arange(D,D+D,1):
            for col in np.arange(0,row,1):
                
                print('row: '+str(row).zfill(2)+' | col: '+str(col).zfill(2))
                        
                samp    = np.column_stack((
                    map_input[::10,col],
                    map_input[::10,row]))
                
                ksds_s[row-D,col] = ksd(
                    q   = samp,
                    dq  = -samp)
    
        # Store the results
        pickle.dump(ksds_s,     open("ksd_L96_smoother_t="+str(t).zfill(3)+".p",          "wb"))
        pickle.dump(ksds_marg_s,open("ksd_L96_smoother_marginal_t="+str(t).zfill(3)+".p", "wb"))
        
        
        plt.figure()
        plt.imshow(ksds_s)
        plt.title(np.nanmean(ksds_s))
        plt.colorbar()
        plt.savefig("ksd_smoother_t="+str(t).zfill(3)+".png")
        plt.close('all')
    
    if "ksd_L96_filter_t="+str(t).zfill(3)+".p" not in list(os.listdir(root_directory)):
        
        # Whiten the samples
        map_input   = np.column_stack((
            copy.copy(Y[t,...]),
            copy.copy(Z_f[t,...]) ))
        map_input   -= np.mean(map_input,axis=0)[np.newaxis,:]
        map_input   /= np.std(map_input,axis=0)[np.newaxis,:]
        
        cov         = np.cov(map_input.T)
        chol        = scipy.linalg.cholesky(cov)
        prec        = scipy.linalg.inv(chol)
        map_input   = np.dot(map_input,prec)
        
        # Pre-allocate an array for the KSDs
        ksds_f      = np.zeros((D,O+D))*np.nan
        ksds_marg_f = np.zeros((O+D))*np.nan
        
        # Marginal KSDs
        for idx in range(O+D):
            ksds_marg_f[idx] = ksd(
                q   = map_input[::10,idx][:,np.newaxis],
                dq  = -map_input[::10,idx][:,np.newaxis])
        
        # Joint KSDs
        for row in np.arange(O,O+D,1):
            for col in np.arange(0,row,1):
                
                print('row: '+str(row).zfill(2)+' | col: '+str(col).zfill(2))
                        
                samp    = np.column_stack((
                    map_input[::10,col],
                    map_input[::10,row]))
                
                ksds_f[row-O,col] = ksd(
                    q   = samp,
                    dq  = -samp)
    
        # Store the results
        pickle.dump(ksds_f,     open("ksd_L96_filter_t="+str(t).zfill(3)+".p",            "wb"))
        pickle.dump(ksds_marg_f,open("ksd_L96_filter_marginal_t="+str(t).zfill(3)+".p",   "wb"))
        
        plt.figure()
        plt.imshow(ksds_f)
        plt.title(np.nanmean(ksds_f))
        plt.colorbar()
        plt.savefig("ksd_filter_t="+str(t).zfill(3)+".png")
        plt.close('all')

