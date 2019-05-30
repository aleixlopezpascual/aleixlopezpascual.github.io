import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import sqrt

########################################################################################
# I.

# it is important to fix a seed in order to have the same random numbers every time we run the program
# therefore, we can compare every run with different parameters
# if we use different random numbers for each one, the comparison depends on the randomness
np.random.seed(10)
N = 10**4 #number of points of the data sample
mean = np.array([2,1])
sigma_1 = 1
sigma_2 = 2
rho = 0.999
cov = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]

L = np.linalg.cholesky(cov)
# print(L)

uncorrelated = np.random.normal(0,1,(2,N))

#now we apply the transformation

# we cannot do broadcasting if both arrays do not have the same dimensions (ie 2D)
# reshape of mean in order to do broadcasting
data = mean.reshape(2,1) + np.dot(L, uncorrelated)
# print(mean.reshape(2,1)[0,0], mean.reshape(2,1)[1,0])
# print(data.shape)
# (2, N)

plt.figure(1)
plt.scatter(data[0,:], data[1,:], color = "black", s = 0.1)
plt.xticks(np.arange(-12,12,1))
plt.yticks(np.arange(-10,10,1))
plt.xlim(-6,10)
plt.ylim(-7,9)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

#################################################
# alternative: if instead we use
"""
plt.figure(2)
data2 = np.random.multivariate_normal(mean, cov, N)
plt.scatter(data2[:,0], data2[:,1], color = "black", s = 1)
"""

# plt.show()


########################################################################################
# III.

def posterior_pdf_2(mu):
    """
    this is the posterior pdf computed in II. We want to sample this for different values of mu.
    this function brings problems as explained at the end of the function
    """
    mean_1 = mu[0]
    mean_2 = mu[1]
    x_1 = data[0, :]  # data fixed from the first experiment
    x_2 = data[1, :]

    posterior = np.exp(-1 / (2 * (1 - rho ** 2)) * (
    (x_1 - mean_1) ** 2 / sigma_1 ** 2 - 2 * rho * (x_1 - mean_1) * (x_2 - mean_2) / (sigma_1 * sigma_2) +
    (x_2 - mean_2) ** 2 / sigma_2 ** 2))

    # for the moment we have an array 1D (N,)
    # print(posterior, np.shape(posterior))
    return posterior

    # if we perform the product of all the elements
    # posterior = np.prod(posterior)
    # return posterior
    # np.shape(posterior)
    # shape posterior (,) it is a number
    # PROBLEM: the posterior pdf computed alone with the prod operator, gives 0 because overflow
    # But if we compute the ratio, this gives =/ 0
    # Precisely, that's why we use MCMC and we don't plot directly from the pdf


def posterior_pdf(mu):
    """
    this is the posterior pdf computed in II. We want to sample this for different values of mu.
    It's an alternative method using log in order to evade numerical underflow
    """
    mean_1 = mu[0]
    mean_2 = mu[1]
    x_1 = data[0, :]  # data fixed from the first experiment
    x_2 = data[1, :]

    log_posterior = (x_1 - mean_1) ** 2 / sigma_1 ** 2 - 2. * rho * (x_1 - mean_1) * (x_2 - mean_2) / (
        sigma_1 * sigma_2) + (x_2 - mean_2) ** 2 / sigma_2 ** 2
    log_posterior = -(1. / (2. * (1 - rho ** 2))) * np.sum(log_posterior)

    # for the moment we a number, but still we need to perform the exponential of this
    return log_posterior


def proposal_distribution(mu):
    global c #we define c as global, because later (outside the function) we want to print this value
    # mu must be 1D array here
    cov_diag = np.array([[sigma_1 ** 2, 0], [0, sigma_2 ** 2]])
    #    c = 2.4/sqrt(2.) #in our case this optimal value brings problems
    c = 0.001
    proposal = np.random.multivariate_normal(mu, c ** 2 * cov_diag)
    return proposal
    # np.shape(proposal), np.shape(mu)
    # (2,) (2,) ie both are 1-D


def met_hast(f, proposal, old):
    """
    metropolis_hastings algorithm.
    allows proposal asymetric
    _f_ is the unnormalized density function to sample, i.e. the posterior pdf
    _proposal_ is the proposal distirbution J
    _old_ is the value mu^{t-1}, i.e. the last iteration
    """

    # we sample mu* from the proposal distribution
    new = proposal(old)

    # we must be careful with the overflow problems in the ratio.
    # the posterior pdf computed alone with the prod operator, gives 0

    ratio_posterior_pdf_log = f(new) - f(old)
    # print(ratio_posterior_pdf_log)  # this is a number

    ratio_posterior_pdf = np.exp(ratio_posterior_pdf_log)
    # print(ratio_posterior_pdf)

    ratio = ratio_posterior_pdf * proposal(new)/new
    # note we call new = proposal(old) instead of proposal(old) because if we use proposal(old)
    # we would be calling proposal(old) again, so it will be different than new because is random.
    # print(ratio, np.shape(ratio))
    # returns [ratio_1, ratio_2]
    # shape (2,) 1-D array

    # alpha is the acceptance probability, can be 1 or less. That's why we have np.min
    # if it is 1, the acceptance ratio will be 100%
    # we have 2 ratios, then we have to compare 2 objects
    alpha_1 = np.min([ratio[0], 1])
    alpha_2 = np.min([ratio[1], 1])

    # now we start the acceptance or rejection
    # we generate a random uniform number u and we compare it with alpha_1 and alpha_2
    u = np.random.uniform()
    cnt = 0
    if (u <= alpha_1 and u <= alpha_2):
        old = new #this is a 1D array (2,)
        cnt = 1

    print(old)
    return old, cnt
    # if accepted we return old = new and cnt=1, if rejected we return old and cnt=0

def run_chain(chainer, f, proposal, start, n, take=1):
    """
    _chainer_ is the method used: Metropolis, MH, Gibbs ...
    _f_ is the unnormalized density function to sample, i.e. the posterior pdf
    _proposal_ is the proposal distirbution J
    _start_ is the initial start of the Markov Chain, i.e. the initial value of mu_1,mu_2
    _start_ must be an array 1D (2,), ie a list.
    _n_ length of the chain. We can modify it to improve the convergance
    _take_ thinning
    """
    # we initialise the counter of the number of accepted values
    count = 0
    # samples recolect all the mu_1,mu_2 values along the chain
    # but we want to have an array (2, n)
    samples = np.array(start.reshape(2,1))
    for i in range(n): # we iterate
        print(i)
        start, c = chainer(f, proposal, start) # start will be the value of mu_1,mu_2, it's no longer the initial value
        count = count + c # this count only adds +1 when is mu* accepted
        if i%take is 0: # we recolect the values for each iteration, even if not accepted
            samples = np.append(samples, start.reshape(2,1), axis=1)
            # now samples will be an array (2, n+1)
            # ie two rows and n+1 columns. The two rows correspond to mu_1 and mu_2
    return samples, count



start_point = np.array([0,0])
n_iterations = 30000 #number of iterations of the chain
samples, count = run_chain(met_hast, posterior_pdf, proposal_distribution, start=start_point, n=n_iterations)

#print(samples)
#print(np.shape(samples))

# we apply the burn-in period
# the conventional choice is to discard the first half (gelman pag 297)
# however, analyzing the values at each iteration, we see that from 1000 iteration there is already convergence

burn_in = 15000
samples_burn_in = samples[:, burn_in : n_iterations+1]
# remember slices must be integers that's why we use //
print(np.shape(samples_burn_in))

# now we compute the sample mean and sample standard deviation
# we select the axis 1 to obtain 2 means: mean_1 mean_2
# the standard deviation is biased with ddof=0 and ddof=1
# the var is the unbiased estimator for ddof=1
sample_mean = np.mean(samples_burn_in, axis=1)
standard_deviation_mean = np.std(samples_burn_in, axis=1, ddof=1)
print("number iterations =", n_iterations)
print("c =", c)
print("burn-in =", burn_in)
print('Acceptance fraction:', count / float(n_iterations))
print("sample mean:", sample_mean)
print("sample standard deviation:", standard_deviation_mean)

# we try different values of c in order to find an appropiate acceptance fraction
# and we want the sample mean and standard deviation to fit with the true values

plt.figure(3)
plt.scatter(samples_burn_in[0,:], samples_burn_in[1,:], color = "black", s = 0.1)
plt.xticks(np.arange(1.90, 2.10, 0.02))
plt.yticks(np.arange(0.90, 1.10, 0.02))
plt.xlim(1.92,2.08)
plt.ylim(0.92,1.08)
plt.xlabel(r"$\mu_1$")
plt.ylabel(r"$\mu_2$")

np.savetxt("samples_burn_in", samples_burn_in)

plt.show()



###########################################################################################
# IV.
# Gibbs Sampler
# we have two parameters mu_1 and mu_2. Then two steps per iteration t.

# we define a function for Gibbs which fits with the run_chain function.
def gibbs(mu):
    """
    Gibbs sampling algorithm
    _mu_ is the value of mu^{t-1}, ie the last iteration. It must be an array 1D (2,) (mu_1, mu_2)
    output: mu of the iteration t
    """

    mu_1 = mu[0] #note that we don't use this value
    mu_2 = mu[1]
    x_1 = data[0, :]  # data fixed from the first experiment
    x_2 = data[1, :]

    x_1_mean = np.mean(x_1) # number
    x_2_mean = np.mean(x_2)

    # we choose the order (mu_1, mu_2)
    # we start sampling mu_new_1

    # be careful the arguments of the normal are the mean and the std

    normal_mean_1 = x_1_mean + rho * sigma_1 / sigma_2 * (mu_2 - x_2_mean)
    normal_std_1 = sqrt(sigma_1 ** 2 * (1 - rho ** 2) / N)
    # they are numbers

    mu_new_1 = np.random.normal(normal_mean_1, normal_std_1)
    # number

    # now we sample mu_new_2  using mu_new_1
    normal_mean_2 = x_2_mean + rho * sigma_2 / sigma_1 * (mu_new_1 - x_1_mean)
    normal_std_2 = sqrt(sigma_2 ** 2 * (1 - rho ** 2) / N)

    mu_new_2 = np.random.normal(normal_mean_2, normal_std_2)

    mu_new = np.array([mu_new_1, mu_new_2])
    print(mu_new)

    cnt = 1
    return mu_new, cnt


""" #######################################################################
    Here there is a wrong attempt.
    I tried to compute mu_new_1 as     mu_new_1 = np.prod(np.random.normal(normal_mean_1, normal_std_1))
    where i computed a one-d gaussian for every data, ie 10^4 gaussian and then the product.
    the result was clearly bigger than the desired and it had overflow problems.
    
    # be careful the arguments are the mean and the std
    normal_mean_1 = x_1 - rho*sigma_1*(x_2-mu_2)/sigma_2
    normal_std_1 = sqrt(sigma_1**2*(1-rho**2))
    # print(np.shape(normal_mean_1))
    # (N, ) 1D array
    # print(np.shape(normal_std_1))
    # () scalar

    mu_new_1 = np.random.normal(normal_mean_1, normal_std_1)
    # print(mu_new_1)
    # print(np.shape(mu_new_1))
    # (N,) 1D array

    # for the moment, for any value of the data, we have obtained one mu_1
    # now we have to make the product of all of them so we obtain only one mu_1
    # but we have overflow problems. It is obvious because we are multiplying sth 10**4 times
    # way out: apply log

    # however we can have log(negative number) which gives nan.
    # in order to solve this, we perform a mask which filters out the negative values
    # at the end, i think we can make this approximation because most of the values are positive

    check the values of mu_new_1
    for i in range(0,N):
        print(mu_new_1[i])
    

    # we perform the mask
    mask = mu_new_1 > 0
    mu_new_1 = mu_new_1[mask]

    # check the values of mu_new_1 after the mask
    for i in range(0,len(mu_new_1)):
        print(mu_new_1[i])
    

    # we perform the natural logarithm of the array element-wise
    mu_new_1 = np.log(mu_new_1)

    #then we sum each element
    mu_new_1 = np.sum(mu_new_1)



    # now we sample mu_new_2  using mu_new_1
    normal_mean_2 = x_2 - rho*sigma_2*(x_1-mu_new_1)/sigma_1
    normal_std_2 = sqrt(sigma_2**2*(1-rho**2))

    mu_new_2 = np.random.normal(normal_mean_2, normal_std_2)

    # we perform the natural logarithm of the array element-wise
    mu_new_2 = np.log(mu_new_2)

    #then we sum each element
    mu_new_2 = np.sum(mu_new_2)

    mu_new = np.array([mu_new_1, mu_new_2])
    print(mu_new)

    cnt = 1
    return mu_new, cnt
    
    """

def run_chain_gibbs(chainer, start, n, take=1):
    """
    _chainer_ is the method used: Metropolis, MH, Gibbs ...
    _start_ is the initial start of the Markov Chain, i.e. the initial value of mu_1,mu_2
    _start_ must be an array 1D (2,), ie a list.
    _n_ length of the chain. We can modify it to improve the convergance
    _take_ thinning
    """
    # we initialise the counter of the number of accepted values
    count = 0
    # samples recolect all the mu_1,mu_2 values along the chain
    # but we want to have an array (2, n)
    samples = np.array(start.reshape(2,1))
    for i in range(n): # we iterate
        start, c = chainer(start) # start will be the value of mu_1,mu_2, it's no longer the initial value
        print(i)
        count = count + c # this count only adds +1 when is mu* accepted
        if i%take is 0: # we recolect the values for each iteration, even if not accepted
            samples = np.append(samples, start.reshape(2,1), axis=1)
            # now samples will be an array (2, n+1)
            # ie two rows and n+1 columns. The two rows correspond to mu_1 and mu_2
    return samples, count


"""
start_point = np.array([0,0])
n_iterations = 30000 #number of iterations of the chain
samples, count = run_chain_gibbs(gibbs, start=start_point, n=n_iterations)

# print(samples)
print(np.shape(samples))

# we apply the burn-in period
# the conventional choice is to discard the first half (gelman pag 297)

burn_in = 15000
samples_burn_in = samples[:, burn_in : n_iterations+1]
# remember slices must be integers that's why we use //
#print(samples_burn_in)
print(np.shape(samples_burn_in))

# now we compute the sample mean and sample standard deviation
# we select the axis 1 to obtain 2 means: mean_1 mean_2
# the standard deviation is biased with ddof=0 and ddof=1
# the var is the unbiased estimator for ddof=1
sample_mean = np.mean(samples_burn_in, axis=1)
standard_deviation_mean = np.std(samples_burn_in, axis=1, ddof=1)
print("number iterations =", n_iterations)
print("burn-in =", burn_in)
print('Acceptance fraction:', count / float(n_iterations))
print("sample mean:", sample_mean)
print("sample standard deviation:", standard_deviation_mean)

# we try different values of c in order to find an appropiate acceptance fraction
# and we want the sample mean and standard deviation to fit with the true values

plt.figure(4)
plt.scatter(samples_burn_in[0,:], samples_burn_in[1,:], color = "black", s = 0.1)
plt.xticks(np.arange(1.90, 2.10, 0.02))
plt.yticks(np.arange(0.90, 1.10, 0.02))
plt.xlim(1.92,2.08)
plt.ylim(0.92,1.08)
plt.xlabel(r"$\mu_1$")
plt.ylabel(r"$\mu_2$")

np.savetxt("samples_burn_in", samples_burn_in)

plt.show()
"""


