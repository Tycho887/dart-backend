# The goal of this test is to make sure the optimizer converges given the data, or gives a clear warning of low data quality

# We first fit a lofi-model + time offset for a single pass. We then report the RMSE in position for 1 hour around TLE epoch.
# This is the reference point

# Later, we 3 passes at a time and fit lofi L + n (two parameters), in a sliding window.

# Later we also fit hifi 6 DOF with an increasing number of passes, and plot how well the accuracy scales with 1 pass to all passes etc.

# The final experiment is to first run the full pass + hifi model, estimate the model covariance. and if certain passes have an extreme
# Bias variance, we interpret this as the pass containing little information, and then remove all passes with information content below
# some threshold.

# In all cases, we first load all of the data and cache it so that we don't need to make ADX requests for each orbit fit. The output should be split
# In two parts: a single experiment.JSON outlining the residual timeseries compared to the fitted oem, and a 1 hour RMSE showing the mean + variance for 
# position and velocity error within 1 hour of fit epoch, defined as either the TLE epoch or the center of the timestamps of doppler values.

