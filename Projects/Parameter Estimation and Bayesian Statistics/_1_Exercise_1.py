import numpy as np
from math import sqrt

# we define the values of the statement
observations = np.array([1.017,2.221,1.416,0.641,0.124,1.728])

# our parameter to estimate is the mean
# we use the arithmetic mean
mean_obs=np.mean(observations)

# we compute the standard deviation of the arithmetic mean
standard_deviation_sample_mean = 0.75/sqrt(6)

def compute_a_and_b(quantile):
    a = mean_obs - standard_deviation_sample_mean*quantile
    b = mean_obs + standard_deviation_sample_mean*quantile
    return a,b

# 1 - alpha = 0.68, quantile = 1.000
quantile = 1.000
print(0.68, compute_a_and_b(quantile))

# 1 - alpha = 0.90, quantile = 1.645
quantile = 1.645
print(0.90, compute_a_and_b(quantile))

# 1 - alpha = 0.95, quantile = 1.960
quantile = 1.960
print(0.95, compute_a_and_b(quantile))

print()
# ----------------------
# now we compute CI for the case sigma is unknown.
# first we estimate sigma with the unbiased estimator (ddof=1)
standard_deviation_estimation = np.std(observations, ddof=1)

standard_deviation_sample_mean_estimation = standard_deviation_estimation/sqrt(6)

def compute_a_and_b_estimation(quantile):
    a = mean_obs - standard_deviation_sample_mean_estimation*quantile
    b = mean_obs + standard_deviation_sample_mean_estimation*quantile
    return a,b

# 1 - alpha = 0.68, quantile = 1.104
quantile = 1.104
print(0.68, compute_a_and_b_estimation(quantile))

# 1 - alpha = 0.90, quantile = 2.015
quantile = 2.015
print(0.90, compute_a_and_b_estimation(quantile))

# 1 - alpha = 0.95, quantile = 2.571
quantile = 2.571
print(0.95, compute_a_and_b_estimation(quantile))
