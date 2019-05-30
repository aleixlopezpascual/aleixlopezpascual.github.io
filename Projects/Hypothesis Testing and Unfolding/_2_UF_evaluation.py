# Load Modules

# mathematical tools optimized for lists
import numpy as np
from numpy.linalg import inv
from scipy import stats
# tools for numerical integration
import scipy.integrate as pyint
# tools for numerical function minimization
import scipy.optimize as pyopt
# tools for plots
import matplotlib.pyplot as plt
# for neural network
import sklearn.preprocessing as preproc
import sklearn.neural_network as nn
from sklearn.externals import joblib
from math import factorial
from math import log
from math import sqrt
from math import exp
from math import pi
# allows major and minor ticks in plot:
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

np.random.seed(2) # we set the seed of randomness

############################################################
################## EXERCISE 1 ############################
###########################################################

#################### PART A ##############################
############################################################
# we are going to construct the (square) migration matrix R

# we consider equal range and number of bins (N=M)
# by range means the total width of the sum of widths of all the bins

# we set the bins N (measured distribution) and M (true distribtion)
N = 20
M = 20

# we set the range:
range_bins = [0.,100.]

# we compute the width of the bins:
width_bins = (range_bins[1]- range_bins[0])/N

# we define a 1d gaussian that we will use as s(x|y)):
def gaussian(x, mean, sigma):
    return (1./(sqrt(2.*pi)*sigma)*exp(-0.5*((x - mean)**2)/(sigma**2)))

# we create a 2D array for the migration matrix R
R = np.empty([N,M])
# shape(20,20)
# now we are going to fill this array
for i in range(N):
    for j in range(M):
        # a: left ege, b: right edge
        # bins i: variable x
        # bins j: variable y
        # even in the plot the variable y is shown in the x-axis
        # and the variable x in the y-axis, we take this this notation
        # which is the one in the literature
        a_i = i * width_bins
        b_i = (i + 1) * width_bins
        a_j = j * width_bins
        b_j = (j + 1) * width_bins
        y = (a_j + b_j)/2
        # since we do not integrate over y, we must fix y as the
        # center of the bin j
        # on the other hand, we integrate over x between a_i and b_i

        # we use pyint.quad(func, lim_inf, lim_sup): one variable integral
        # be careful because the integral returns two outputs
        # we are only interested in the first one
        R[i,j], error = pyint.quad(gaussian, a_i, b_i, args=(y, 2.*width_bins))
        # first index row, second index column

"""
plt.figure()
# now we are going to plot these the R matrix in a imshow
# we use this format since we want to plot an array
plt.imshow(R, cmap=plt.cm.Greys)
plt.xticks(np.arange(0,20,1), ("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
                               "16","17","18","19","20"))
plt.yticks(np.arange(0,20,1), ("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
                               "16","17","18","19","20"))
plt.xlabel("bin j")
plt.ylabel("bin i")
plt.colorbar()
plt.show()
"""


#################### PART B ##############################
############################################################
# we assume the true distribution is given by two gaussians
# with the following parameters:

mu1 = 30.
mu2 = 70.
sig1 = 10.
sig2 = 10.

# we define the sum of two 1d gaussians:
def double_gauss(x, mu1, sig1, mu2, sig2):
	g1 = 1./(np.sqrt(2.*np.pi)*sig1) * np.exp(-0.5*(x-mu1)**2/sig1**2)
	g2 = 1./(np.sqrt(2.*np.pi)*sig2) * np.exp(-0.5*(x-mu2)**2/sig2**2)
	return (g1 + g2)

"""
# we define a range of values study the true distribution
# this range is determined up to the parameters given (means)
# the resulting values are going to be the y true values
# distributed as the pdf
y_range = np.linspace(0,100,1000)
y = double_gauss(y_range, mu1, sig1, mu2, sig2)
plt.figure()
plt.plot(y_range, y, color="black")
plt.xlim(0,100)
plt.xlabel("y")
plt.show()
"""

""" this method was bad because i was using MC
# which depend on random numbers
# fine but we do not want this, we want the histogram
# in order to do so, we must apply a Monte Carlo simulation
# remember we did a lot of this in block 1 of statistics

def function_to_minimize(x, mu1, sig1, mu2, sig2):
	g1 = 1./(np.sqrt(2.*np.pi)*sig1) * np.exp(-0.5*(x-mu1)**2/sig1**2)
	g2 = 1./(np.sqrt(2.*np.pi)*sig2) * np.exp(-0.5*(x-mu2)**2/sig2**2)
	return (-1)*0.5*(g1 + g2)

def acceptace_rejection_MC(size):
    x_min = 0.
    x_max = 100.
    r = np.random.uniform(0.,1., size)
    x = x_min + r*(x_max-x_min)
    f_max = pyopt.minimize_scalar(function_to_minimize, args=(mu1, sig1, mu2, sig2))
    # we multiply the result of maximization by -1, because it gives the result
    # -0.2 when it should be 0.2. this is apart from the fact of -1*function
    # maybe the bad maximization is due to we have to provide also the derivative
    # of the function with a minus sign too. Search more inf in scipy.optimize.minimization
    # anyway, here we obtain good results after this correction
    u = np.random.uniform(0,-f_max.fun, size)
    x = x[u < double_gauss(x, mu1, sig1, mu2, sig2)]
    return x

def generate_true_distribution_2(size):
    # here i was trying to copy the method of MC to generate a histogram
    # but without using random numbers.
    # however the result is not good.
    y = np.linspace(0., 100., 100)
    # we make copies of this 1d array
    y_list = []
    for i in np.arange(size / 100.):
        for y_i in y:
            y_list.append(y_i)
    f_max = pyopt.minimize_scalar(function_to_minimize, args=(mu1, sig1, mu2, sig2))
    # we multiply the result of maximization by -1, because it gives the result
    # -0.2 when it should be 0.2. this is apart from the fact of -1*function
    # maybe the bad maximization is due to we have to provide also the derivative
    # of the function with a minus sign too. Search more inf in scipy.optimize.minimization
    # anyway, here we obtain good results after this correction
    u = np.linspace(0, -f_max.fun, 100)
    u_list = []
    for i in np.arange(size / 100.):
        for u_i in u:
            u_list.append(u_i)
    y_list = np.asarray(y_list)
    u_list = np.asarray(u_list)
    y_list = y_list[u_list < double_gauss(y_list, mu1, sig1, mu2, sig2)]
    return y_list

y = acceptace_rejection_MC(100000)
"""

# we compute the probabilities to find y in the bin j
# we need to perform an integral of the pdf over the bin
# we proceed as before:
prob_bins_j = np.empty(M)
for j in range(M):
    # a: left ege, b: right edge
    # bins j: variable y
    a_j = j * width_bins
    b_j = (j + 1) * width_bins
    prob_bins_j[j], error = pyint.quad(double_gauss, a_j, b_j, args=(mu1, sig1, mu2, sig2))

number_events = 25000
mu_tot = number_events
mu = mu_tot*prob_bins_j
mu_exercise1 = mu # we will use this in exercise 2

"""
# now i plot a histogram. but note that i don't have data,
# but i have already the counts, so we proceed as follows:
fig = plt.figure()

mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu, color="black", histtype="step", normed = False)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("y")
plt.ylabel(r"$\mu$")
plt.show()
"""

#################### PART C ##############################
############################################################

nu = np.empty(N)
# we multiply a matrix by a vector
nu = np.dot(R,mu)

"""
fig = plt.figure()

nu_center = np.linspace(0,100,20)
plt.hist(nu_center, bins=N, weights=nu, color="black", histtype="step", normed = False)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"$\nu$")
plt.show()
"""

#################### PART D ##############################
############################################################

n = np.empty(N)
# we fill this array with random numbers generated from a
# poisson distribution with lambda=nu (values computed in C)
# we use the same number of events than before
n = np.random.poisson(nu)

# Let's compute the covariance matrix in order to compute the standard deviations
# for the error bars
# the matrix V (covariance of n) is just diagonal with the values nu
# cov_n_matrix = np.diag(nu)
# we extract the diagonal terms, so we again obtain nu
# and we compute the sqrt
standard_deviation_n = np.sqrt(nu)

"""
# we plot again the previous histogram but adding n with its error bars too
fig = plt.figure()

nu_center = np.linspace(0,100,20)
plt.hist(nu_center, bins=N, weights=nu, color="black", histtype="step", normed = False)

# we want to add n with its error bars
# matplotlib.pyplot.errorbar: Plot an errorbar graph.
# Plot x versus y with error deltas in yerr and xerr.
# we use as x the center of the bins of n
n_center = np.linspace(2.5,97.5,20)
# we use as y the n values
# yerr are the y error bars, which we take to be the standard deviations
# xerr are the x error bars, which we take to be the bin width
# if fmt is none, only the error bars are plotted
plt.errorbar(n_center, n, yerr=standard_deviation_n, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"$\nu$, n")
plt.show()
"""

#################### PART E ##############################
############################################################
# first we compute the inverse of the matrix R

R_inv = np.linalg.inv(R)
#check inv is correct:
#print(np.dot(R_inv,R)) # it gives approx identity (ok)

mu_0 = np.dot(R_inv, n)
# mu_theoretical = np.dot(R_inv, nu) #it recovers the mu initial as expected (ok)

# Let's compute the covariance matrix in order to compute the standard deviations
# for the error bars
# the covariance matrix in matrix notation is computed as
# U = R**(-1)*V*(R**(-1))**T
# the matrix V (covariance of n) is just diagonal with the values nu
cov_n_matrix = np.diag(nu)
cov_mu_matrix = np.dot(R_inv, np.dot(cov_n_matrix, np.transpose(R_inv)))
# this matrix is no longer diagonal as expected
# we are interested in the diagonal terms, we extract them using
# np.diagonal(array)
variance_mu_diagonal = np.diagonal(cov_mu_matrix)
# the sqrt of these terms will be the standard deviation for each bin
standard_deviation_mu = np.sqrt(variance_mu_diagonal)

""""
fig = plt.figure()

mu_0_center = np.linspace(2.5,97.5,20)
# we use as y the mu_0 values
# yerr are the y error bars, which we take to be the standard deviations
# xerr are the x error bars, which we take to be the bin width
# if fmt is none, only the error bars are plotted
plt.errorbar(mu_0_center, mu_0, yerr=standard_deviation_mu, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
plt.yticks(np.arange(-7000*10**5,8000*10**5,1000*10**5))
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.ylim(-6.5*10**8, 6.5*10**8)
plt.xlabel("x")
plt.ylabel(r"$\hat{\mu}$")
plt.show()
"""

#################### PART F ##############################
############################################################

"""
def log_likelihood(nu_lik):
    out = 0.
    for i in range(N):
        out += log(stats.poisson.pmf(n[i], nu_lik[i]))
        # n is the data generated in part d
        # pmf(k, mu, loc=0) gives the prob mass function of a Poisson
        # ie the result given some k and lambda.
    return -out # we add a minus in order to maximize using minimize module

# now we maximize this function and return the parameters nu_lik that
# maximize our function
# we use pyopt.minimze since we minimze a function of 20 variables, which are
# the 20 mu_i

initial_guess = nu+10
minimization = pyopt.minimize(log_likelihood, x0=initial_guess, bounds=((0,None),(0,None),(0,None),(0,None),(0,None),
            (0, None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),
                (0, None),(0,None),(0,None),(0,None),(0,None)))
nu_maximization = minimization.x
"""

"""
fig = plt.figure()

n_center = np.linspace(0,100,20)
plt.hist(n_center, bins=N, weights=n, color="black", histtype="step", normed = False)
nu_center = np.linspace(2.5,97.5,20)
plt.scatter(nu_center, nu_maximization, s=10)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"$\hat{\nu}$, n")
plt.show()
"""

""" This is bad
# I was interpreting bad the problem. The solution I want to obtain
# is nu = n. Here I was trying to minimize wrt mu in order to obtain
# mu_0. but this is not what we want to do.

# we are going to compute the solution for the maximum likelihood
# and compare it to the exact solution obtained in part E
# we should obtain that maximizing the likelihood we obtain
# a vector of mu which is equal to the vector mu_0

# we define the likelihood as a function of mu
# (there is no need to define the log-likelihood, since
# we will obtain the same values mu for both cases)
# a priori the expession does not show the dependency of mu explicitely
# this dependency is in nu_i = R_ik mu_k

def likelihood_mu(mu_lik):
    out = 1.
    nu_prime = np.dot(R, mu_lik)
    for i in range(M):
        out *= stats.poisson.pmf(n[i], nu_prime[i])
        # pmf(k, mu, loc=0) gives the prob mass function of a Poisson
        # ie the result given some k and lambda.
        print(out)
    return -out # we add a minus in order to maximize using minimize module

# now we maximize this function and return the parameters mu that
# maximize our function
# we use pyopt.minimze since we minimze a function of 20 variables, which are
# the 20 mu_j

# as an initial value we set e.g. the mu from the true distribution
# we get 0 iterations (problem) the problem is not the initial guess

initial_guess = mu
# np.zeros(20)
minimization = pyopt.minimize(likelihood_mu, x0=initial_guess)
print(minimization)
mu_lik = minimization.x
print(mu)
print(mu_0)
print(mu_lik)
"""

#################### PART G ##############################
############################################################
# method of correction factors
# this is an iterative method, so we define a function as a function of
# the number of iterations, which repeats the method until the
# desired iteration is reached

def method_correction_factors(iterations):
    count = 0
    if count == 0: # first iteration
        mu_MC = n
        nu_MC = np.dot(R, mu_MC)
        C = mu_MC / nu_MC
        mu_est_C = C*n
        # C is a vector and n is a vector
        # we perform a product component by component to get a vector mu_est_C
        # otherwise we only get one value
        count += 1
    while count < iterations: # further iterations
        mu_MC = mu_est_C
        nu_MC = np.dot(R, mu_MC)
        C = mu_MC / nu_MC
        mu_est_C = C*n
        count += 1
    return C, mu_est_C

"""
plt.figure("mu")
# we plot the true distribution from part A
mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu, color="black", histtype="step", normed = False)

plt.figure("C")
#plot for 1 iteration
C, mu_est_C = method_correction_factors(1)
mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=C, color="red", histtype="step", normed = False, label="First iteration")

plt.figure("mu")
# we compute the covariance matrix for mu in order to define the error bars
cov_mu_est_C_matrix = C**2*nu # it is diagonal
standard_deviation_mu_est_C = np.sqrt(cov_mu_est_C_matrix)
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_est_C, yerr=standard_deviation_mu_est_C, xerr=width_bins/2,
             fmt="none", color = "red", elinewidth=1, label="First iteration")

# plot for 5 iterations
plt.figure("C")
C, mu_est_C = method_correction_factors(5)
mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=C, color="black", histtype="step", normed = False, label="After 5 iterations")

plt.figure("mu")
# we compute the covariance matrix for mu in order to define the error bars
cov_mu_est_C_matrix = C**2*nu # it is diagonal
standard_deviation_mu_est_C = np.sqrt(cov_mu_est_C_matrix)
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_est_C, yerr=standard_deviation_mu_est_C, xerr=width_bins/2,
             fmt="none", color = "black", elinewidth=1, label="After 5 iterations")


fig = plt.figure("C")
ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,2., 0.05), minor=True)
plt.ylim(0,1.55)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"C")
leg = plt.legend(loc="lower center")
leg.get_frame().set_edgecolor('black')

fig = plt.figure("mu")
ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5400,200), minor=True)
plt.ylim(0,5200)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"$\hat{\mu}$")
leg = plt.legend(loc="lower center")
leg.get_frame().set_edgecolor('black')

plt.show()
"""

#################### PART H ##############################
############################################################
"""
# we create the chi2 as a funciton of theta
# which is a vector of 6 parameters
# in our case (double gaussian): mu1, mu2, sigma1, sigma2, N1, N2
# we want to estimate these 6 parameters so that we can plot f_true(y)
# for y in [0,100]

# first of all, we define our function double gaussian, but now
# as a function of the 6 parameters (we don't know them)
def double_gauss_parametric(x, N1, mu1, sig1, N2, mu2, sig2):
	g1 = N1 * np.exp(-0.5*(x-mu1)**2/sig1**2)
	g2 = N2 * np.exp(-0.5*(x-mu2)**2/sig2**2)
	return (g1 + g2)
# note that N1 and N2 correspon to 1/(np.sqrt(2.*np.pi)*sig1) and 1/(np.sqrt(2.*np.pi)*sig2)

#V is the covariance matrix of the data
#V = cov[ni,nj] = nu_i delta_ij (diagonal)
V = np.diag(nu)
V_inv = np.linalg.inv(V)


# in chi2 we need mu, so first we define a function mu(theta)
# we proceed as we did in part B:
def mu_F_estimation(theta):
    prob_bins_j = np.empty(M)
    for j in range(M):
        # a: left ege, b: right edge
        # bins j: variable y
        a_j = j * width_bins
        b_j = (j + 1) * width_bins
        prob_bins_j[j], error = pyint.quad(double_gauss_parametric, a_j, b_j,
                                           args=(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]))
    mu_F_est = mu_tot*prob_bins_j
    return mu_F_est

# now create the funtion chi2:
# we use the n, nu, V and R computed from the precious exercises
K = np.zeros(N)
L = np.zeros(M)
def chi2(theta):
    mu_F_est = mu_F_estimation(theta)
    out = 0.
    for i in range(N):
        K[i] = 0.
        for k in range(M):
            K[i] += R[i,k]*mu_F_est[k]
        K[i] -= n[i]
        sum_j = 0.
        for j in range(M):
            L[j]= 0.
            for l in range(M):
                L[j] += R[j,l]*mu_F_est[l]
            L[j] -= n[j]
            sum_j += V_inv[i,j]*L[j]
        out += K[i]*sum_j
    print(out)
    return out

# once we have the chi2, we proceed to minimize
# we use pyopt.minimze since we minimze a function of 6 variables, which are
# the 6 parameters theta

initial_guess = np.array([1, 20, 7, 1, 80, 12])
minimization = pyopt.minimize(chi2, x0=initial_guess, options={"maxiter": 1000},
                              bounds= ((0,None), (0,None), (1,None), (0,None), (0,None), (1,None)))
# we redefine the number of max iterations, since with the default number is not enough
# i wanted to try if i could define the error of the parameters minimized by performing different
# initial points and methods. but i get the same result for the different initial points and methods.
print(minimization)
theta_parameters = minimization.x
print("N1: {0:.3f}".format(initial_guess[0]))
print("mu1: {0:.3f}".format(initial_guess[1]))
print("sigma1: {0:.3f}".format(initial_guess[2]))
print("N2: {0:.3f}".format(initial_guess[3]))
print("mu2: {0:.3f}".format(initial_guess[4]))
print("sigma2: {0:.3f}".format(initial_guess[5]))
print("Parameter estimations:")
print("N1: {0:.5f}".format(theta_parameters[0]))
print("mu1: {0:.3f}".format(theta_parameters[1]))
print("sigma1: {0:.3f}".format(theta_parameters[2]))
print("N2: {0:.5f}".format(theta_parameters[3]))
print("mu2: {0:.3f}".format(theta_parameters[4]))
print("sigma2: {0:.3f}".format(theta_parameters[5]))

"""
"""
# we compute the covariance matrix of the estimators theta
# cowan pag 111 eq 7.11
# this is bad, the cov_theta matrix is more complicated in our case
# i don't know how to compute it
cov_theta_matrix = np.linalg.inv(np.dot(np.transpose(R), np.dot(V_inv, R)))
print(np.shape(cov_theta_matrix))
standard_deviations_theta = np.sqrt(np.diagonal(cov_theta_matrix))
print("Standard deviations:")
print("N1: {0:.5f}".format(standard_deviations_theta[0]))
print("mu1: {0:.3f}".format(standard_deviations_theta[1]))
print("sigma1: {0:.3f}".format(standard_deviations_theta[2]))
print("N2: {0:.5f}".format(standard_deviations_theta[3]))
print("mu2: {0:.3f}".format(standard_deviations_theta[4]))
print("sigma2: {0:.3f}".format(standard_deviations_theta[5]))
"""
"""

# now we compute the mu (estimation) obtained with these parameters:
mu_F_est = mu_F_estimation(theta_parameters)

# we also compute the covariance matrix U for the error bars:
# we don't have to do it, the exercise do not ask us to plot mu
# we only have to plot the f_true given by the parameters
# futhermore it is not trivial to know how the expression
# of this covariance matrix

fig = plt.figure("histogram")

# we plot the true distribution as a histogram
mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu, color="black", histtype="step", normed = False, label= r"$\mu$")

# we plot mu_F_est
mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu_F_est, color="black", histtype="step",
         ls ="dashed", normed = False, label= r"$\hat{\mu}$")

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("y")
plt.ylabel(r"$\mu$, $\hat{\mu}$")
leg = plt.legend(loc="lower center")
leg.get_frame().set_edgecolor('black')
plt.show()


fig = plt.figure("distribution")

# we plot the true distribution (not histogram)
y_range = np.linspace(0,100,1000)
y = double_gauss(y_range, mu1, sig1, mu2, sig2)
plt.plot(y_range, y, color="black", label = "True")

# we plot the true distribution (not histogram) using the minimizing parameters
y = double_gauss_parametric(y_range, *theta_parameters)
# we use directly *args to pass the 6 parameters
plt.plot(y_range, y, color="black", ls = "dashed", label = "Estimated")

plt.xticks(np.arange(0,110,10))
plt.ylim(0)
plt.xlim(0,100)
plt.xlabel("y")
plt.ylabel(r"$f_{\mathrm{true}}(y)$")
leg = plt.legend(loc="lower center")
leg.get_frame().set_edgecolor('black')
plt.show()
"""




############################################################
################## EXERCISE 2 ############################
###########################################################

#################### PART A ##############################
############################################################
# we recover the values R and n from exercise 1
# but we assume that we don't know nu and mu

#################### PART B ##############################
############################################################
# we chose the Tykhonov regularization method

#################### PART C ##############################
############################################################
# we define the fucntion phi(mu,lambda,alpha)
# since we will want to maximize this function wrt mu and lambda,
# we must include mu and lambda in the same variable x
# alhpa is a parameter we pass as an argument

def log_likelihood_ex2(mu):
    nu = np.dot(R,mu)
    out_i = n*np.log(nu) - nu
    out = np.sum(out_i)
    return out

# we construct the G matrix for the case k=2 (can be found in literature)
# we do it out of the function of Tykhonov in order to not compute it again every time
# also we will need it in the future
# in our case M=20, it is a matrix (20,20)
G = np.zeros((20,20))
# be careful, our matrix start at index 0, the matrix from literature at index 1
for i in range(2,M-2):
    G[i,i] = 6.
    G[i,i+1] = -4.
    G[i,i-1] = -4.
    G[i+1,i] = -4.
    G[i-1,i] = -4.
    G[i,i+2] = 1.
    G[i,i-2] = 1.
    G[i+2,i] = 1.
    G[i-2,i] = 1.
G[0,0]=1.
G[19,19]=1.
G[1,1]=5.
G[18,18]=5.
G[0,1]=-2.
G[1,0]=-2.
G[19,18]=-2.
G[18,19]=-2.
#print(G) #(ok)

def Tykhonov_function(mu):
    mu_T = np.transpose(mu)
    out = -np.dot(mu_T, np.dot(G,mu))
    return out

def phi(x, alpha):
    lambda_parameter = x[20]
    mu = x[0:20] #remember slicing ends at end-1
    out = alpha*log_likelihood_ex2(mu)+Tykhonov_function(mu) + \
          lambda_parameter*(np.sum(n)-np.sum(np.dot(R,mu)))
    return -out # we minimize -function since we want to maximize

# once we have the function defined, we procced to maximize
# we use pyopt.minimze since we minimze a function of 21 variables, which are
# the 20 mu and lambda

# we create an array that stores the values of mu for a given alpha
# the second dimension is in accordance to the number of alpha values we compute
len_alpha = 500
mu_alpha = np.empty((20,len_alpha))
alpha_values = np.empty((len_alpha))

""" we do not minimize every time, we save the values
# we must set the range of values of alpha, which is between 0 and infinity
# we set the range in accordance to the values of Delta*log(L) we get
i = 0
for alpha in np.linspace(7*10**4,3*10**6,len_alpha):
    # as a initial guess we use the true parameter mu
    lambda_initial_guess = 1.
    initial_guess = np.append(mu_exercise1, lambda_initial_guess)

    minimization = pyopt.minimize(phi, x0=initial_guess, options={"maxiter": 30000}, args = alpha,
                                  method = "Nelder-Mead")
#                              bounds=((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),
#                                      (0, None),(0,None),(0,None)#,(0,None),(0,None),(0,None),(0,None),
#                                      (0, None),(0,None),(0,None),(0,None),(0,None),(0,None), (None,None)), args=alpha)
    # we do not consider bounds, because in the literature says that we can obtain estimations with negative values
    # we redefine the number of max iterations, since with the default number is not enough
#    print(minimization)
    print(minimization.x)
    print(minimization.success)
    mu_alpha[:,i] = minimization.x[0:20] # we only want to save the mu values, not the lambda
    alpha_values[i] = alpha # we also save the alpha values
    i += 1

# print(mu_exercise1)
np.save("mu_alpha", mu_alpha)
np.save("alpha_values", alpha_values)
"""

mu_alpha = np.load("mu_alpha.npy")
alpha_values = np.load("alpha_values.npy")

#################### PART D E and F ##############################
############################################################
# first the values of Delta*log(L), which is a vector with as many
# components as values of alpha in part C
# in order to compute log(L) we use the function defined in C
# log L_max is computed with the mu_0 obtained in part E

# there is sth bad with delta_log_L, it should give results between 0,15
# and for alpha very big, it should give practically 0

Delta_log_L = np.empty(len_alpha)
for i in range(0,len_alpha):
    Delta_log_L[i] = log_likelihood_ex2(mu_0)-log_likelihood_ex2(mu_alpha[:,i])
#print(Delta_log_L)
#print(alpha_values)

#####################################################
# we compute the covariance matrix of the estimator
# first we compute the matrices A and B
# it is important to note that we have a matrix A and B for every alpha
# i.e. for every set of mu
# then we compute C
# since we need to compute C for every pair alpha,mu, we create a function

def C_matrix_computation(mu,alpha):
    A_matrix = np.zeros((21,21))
    K = np.zeros((20,20))
    L = np.zeros(20)
    nu = np.dot(R,mu)
    for i in range(0,M):
        for j in range(0,M):
            K[i,j]=0.
            for k in range(N):
                K[i,j] += n[k]*R[k,i]*R[k,j]/(nu[k]**2)
            A_matrix[i,j] = - alpha*K[i,j] - 2*G[i,j]
    for i in range(M):
#        L[i] = 0.
#        for l in range(N):
#            L[i] += R[l,i]
#        A_matrix[i,19] = -L[i]
        A_matrix[i,20]=-1.
    for j in range(M):
        A_matrix[20,j]=-1.
    A_matrix[20,20]=0.


    B_matrix = np.zeros((21,20))
    for i in range(M):
        for j in range(M):
            B_matrix[i,j] = alpha*R[j,i]/nu[j]
    for j in range(N):
        B_matrix[20,j] = 1.

    A_matrix_inv = np.linalg.inv(A_matrix)
    C_matrix = -np.dot(A_matrix_inv, B_matrix)
    #print(np.shape(C_matrix)) #(21,20) (ok)
    C_matrix_submatrix = np.delete(C_matrix, M, axis=0) #we delete the row M+1
    #print(np.shape(C_matrix_submatrix)) #(20,20) (ok)
    return C_matrix_submatrix

# now we compute the covariance matrix U
def cov_U_matrix_computation(mu,alpha):
    nu = np.dot(R,mu)
    V = np.diag(nu)
    C = C_matrix_computation(mu,alpha)
    out = np.dot(C,np.dot(V,np.transpose(C)))
    return out

#now we compute the bias of the estimator:
def bias_computation(mu,alpha):
    nu = np.dot(R,mu)
    C = C_matrix_computation(mu,alpha)
    out = np.dot(C,(nu-n))
    return out

# covariance matrix for the bias:
def cov_W_matrix_computation(mu,alpha):
    I = np.identity(20) #shape 20x20
    U = cov_U_matrix_computation(mu, alpha)
    C = C_matrix_computation(mu,alpha)
    term = np.dot(C,R)-I
    out = np.dot(term,np.dot(U,np.transpose(term)))
    return out

def MSE(mu, alpha):
    U = cov_U_matrix_computation(mu, alpha)
    b = bias_computation(mu, alpha)
    #print(np.shape(U)) # 20,20 ok
    #print(np.shape(b)) # 20 ok
    sum = 0.
    for i in range(M):
        sum += U[i,i] + b[i]**2
    out = (1./M)*sum
    return out

def weigthed_MSE(mu,alpha):
    U = cov_U_matrix_computation(mu, alpha)
    b = bias_computation(mu, alpha)
    sum = 0.
    for i in range(M):
        sum += (U[i,i] + (b[i])**2)/(mu[i])
    out = (1./M)*sum
    return out

def chi2_eff(mu,alpha):
    C = C_matrix_computation(mu,alpha)
    nu = np.dot(R,mu)
    V = np.diag(nu)
    V_inv = np.linalg.inv(V)
    term = nu-n
#    term2 = np.transpose(np.dot(R,C))
#    out1 = np.dot(V_inv,np.dot(term2,term))
#    out2 = np.dot(R,np.dot(C,out1))
#    out = np.dot(np.transpose(term),out2)
    out = np.linalg.multi_dot([np.transpose(term),R,C,V_inv,np.transpose(np.dot(R,C)),term]) # (ok)
    return out

def chi2_b(mu,alpha):
    b = bias_computation(mu, alpha)
    W = cov_W_matrix_computation(mu, alpha)
    sum = 0.
    for i in range(M):
        sum += b[i]**2/W[i,i]
    out = (1./M)*sum
    return out

######################################################
# now we proceed to make the plots:

# MSE
"""
################ PART D
fig = plt.figure("MSE")

MSE_values = np.empty(len_alpha)
for i in range(0, len_alpha):
    MSE_values[i] = MSE(mu_alpha[:,i], alpha_values[i])
    print(i)

# we select the index of the minimum value of MSE
# and we look at which value of alpha and mu corresponds
index_MSE_optimal = np.argmin(MSE_values)
mu_MSE_opt = mu_alpha[:,index_MSE_optimal]
alpha_MSE_opt = alpha_values[index_MSE_optimal]
Delta_log_L_opt = Delta_log_L[index_MSE_optimal]

plt.plot(Delta_log_L , MSE_values, color="black", lw =1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(15000,33000, 500), minor=True)
ax.set_xticks(np.arange(0,20, 0.25), minor=True)
plt.yticks(np.arange(15000,35000,2500))
plt.xticks(np.arange(0,20,1))
plt.ylim(15000,32500)
plt.xlim(0,12)
plt.xlabel(r"$\Delta \log L$")
plt.ylabel(r"MSE")
#leg = plt.legend(loc="lower center")
#leg.get_frame().set_edgecolor('black')

#we plot lines:
plt.axvline(x=Delta_log_L_opt,ls="dashed",color="black",lw=0.5,)

################# PART E
fig = plt.figure()

mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu_exercise1, color="black", histtype="step", normed = False)
plt.hist(mu_center, bins=N, weights=n, color="black", histtype="step", normed = False, ls ="dashed")

# we compute with the optimal values of mu and alpha
cov_U = cov_U_matrix_computation(mu_MSE_opt,alpha_MSE_opt)
standard_deviation = np.sqrt(np.diag(cov_U))
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_MSE_opt, yerr=standard_deviation, xerr=width_bins/2,
             fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"n, $\mu$, $\hat{\mu}$")

################ PART F

fig = plt.figure()
# we compute the bias and its covariance
bias = bias_computation(mu_MSE_opt, alpha_MSE_opt)
cov_W = cov_W_matrix_computation(mu_MSE_opt,alpha_MSE_opt)
standard_deviation = np.sqrt(np.diag(cov_W))

mu_0_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_0_center, bias, yerr=standard_deviation, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(-300,400,20), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.yticks(np.arange(-300,400,100))
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.ylim(-300,300)
plt.xlabel("x")
plt.ylabel(r"$\hat{b}$")
#we plot lines:
plt.axhline(y=0,ls="dashed",color="black",lw=0.5,)

plt.show()
"""

# WMSE
"""
################ PART D
fig = plt.figure("Weighted MSE")

WMSE_values = np.empty(len_alpha)
for i in range(0, len_alpha):
    WMSE_values[i] = weigthed_MSE(mu_alpha[:,i], alpha_values[i])
    print(i)

# we select the index of the minimum value of MSE
# and we look at which value of alpha and mu corresponds
index_WMSE_optimal = np.argmin(WMSE_values)
mu_WMSE_opt = mu_alpha[:,index_WMSE_optimal]
alpha_WMSE_opt = alpha_values[index_WMSE_optimal]
Delta_log_L_opt = Delta_log_L[index_WMSE_optimal]

plt.plot(Delta_log_L , WMSE_values, color="black", lw =1)

# in the funcion of WMSE
# print(mu[i]) #problem: with Tikinhov regularisation the bins of the extreme get negative values
# this mu[i] are the ones with small values, then they have the bigger contribution in sum
# as a consequence, the sum is negative for most values of alpha
# however, although the plot is a bit different, the alpha optimal i get is the optimal one.

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(-10000,25000, 1000), minor=True)
ax.set_xticks(np.arange(0,20, 0.25), minor=True)
plt.yticks(np.arange(-10000,25000,5000))
plt.xticks(np.arange(0,20,1))
plt.ylim(-10000,20000)
plt.xlim(0,12)
plt.xlabel(r"$\Delta \log L$")
plt.ylabel(r"Weighted MSE")
#leg = plt.legend(loc="lower center")
#leg.get_frame().set_edgecolor('black')

#we plot lines:
plt.axvline(x=Delta_log_L_opt,ls="dashed",color="black",lw=0.5,)

################# PART E
fig = plt.figure()

mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu_exercise1, color="black", histtype="step", normed = False)
plt.hist(mu_center, bins=N, weights=n, color="black", histtype="step", normed = False, ls ="dashed")

# we compute with the optimal values of mu and alpha
cov_U = cov_U_matrix_computation(mu_WMSE_opt,alpha_WMSE_opt)
standard_deviation = np.sqrt(np.diag(cov_U))
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_WMSE_opt, yerr=standard_deviation, xerr=width_bins/2,
             fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"n, $\mu$, $\hat{\mu}$")

################ PART F

fig = plt.figure()
# we compute the bias and its covariance
bias = bias_computation(mu_WMSE_opt, alpha_WMSE_opt)
cov_W = cov_W_matrix_computation(mu_WMSE_opt,alpha_WMSE_opt)
standard_deviation = np.sqrt(np.diag(cov_W))

mu_0_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_0_center, bias, yerr=standard_deviation, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(-300,400,20), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.yticks(np.arange(-300,400,100))
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.ylim(-300,300)
plt.xlabel("x")
plt.ylabel(r"$\hat{b}$")
#we plot lines:
plt.axhline(y=0,ls="dashed",color="black",lw=0.5,)

plt.show()
"""

# chi2 eff
"""
################ PART D
fig = plt.figure("chi2 eff")

chi2_eff_values = np.empty(len_alpha)
for i in range(0, len_alpha):
    chi2_eff_values[i] = chi2_eff(mu_alpha[:,i], alpha_values[i])
    print(i)

# we select the index at which chi2 is equal to 1
# and we look at which value of alpha and mu corresponds
# in order to find nearest value in numpy array
# we use (np.abs(array-value)).argmin()
index_chi2_eff_optimal = np.argmin(np.abs(chi2_eff_values-1.))
mu_chi2_eff_opt = mu_alpha[:,index_chi2_eff_optimal]
alpha_chi2_eff_opt = alpha_values[index_chi2_eff_optimal]
Delta_log_L_opt = Delta_log_L[index_chi2_eff_optimal]

#print(chi2_eff_values)
plt.plot(Delta_log_L , chi2_eff_values, color="black", lw =1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,20, 0.2), minor=True)
ax.set_xticks(np.arange(0,20, 0.25), minor=True)
plt.yticks(np.arange(0,30))
plt.xticks(np.arange(0,20,1))
plt.ylim(0,10)
plt.xlim(0,12)
plt.xlabel(r"$\Delta \log L$")
plt.ylabel(r"$\chi^2_{\mathrm{eff}}$")
#leg = plt.legend(loc="lower center")
#leg.get_frame().set_edgecolor('black')

#we plot lines:
plt.axhline(y=1,ls="dashed",color="black",lw=0.5,)
plt.axvline(x=Delta_log_L_opt,ls="dashed",color="black",lw=0.5,)

################# PART E
fig = plt.figure()

mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu_exercise1, color="black", histtype="step", normed = False)
plt.hist(mu_center, bins=N, weights=n, color="black", histtype="step", normed = False, ls ="dashed")

# we compute with the optimal values of mu and alpha
cov_U = cov_U_matrix_computation(mu_chi2_eff_opt,alpha_chi2_eff_opt)
standard_deviation = np.sqrt(np.diag(cov_U))
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_chi2_eff_opt, yerr=standard_deviation, xerr=width_bins/2,
             fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"n, $\mu$, $\hat{\mu}$")

################ PART F

fig = plt.figure()
# we compute the bias and its covariance
bias = bias_computation(mu_chi2_eff_opt, alpha_chi2_eff_opt)
cov_W = cov_W_matrix_computation(mu_chi2_eff_opt,alpha_chi2_eff_opt)
standard_deviation = np.sqrt(np.diag(cov_W))

mu_0_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_0_center, bias, yerr=standard_deviation, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(-300,400,20), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.yticks(np.arange(-300,400,100))
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.ylim(-300,300)
plt.xlabel("x")
plt.ylabel(r"$\hat{b}$")
#we plot lines:
plt.axhline(y=0,ls="dashed",color="black",lw=0.5,)

plt.show()
"""

# chi2 b
"""
fig = plt.figure("chi2 b")

chi2_b_values = np.empty(len_alpha)
for i in range(0, len_alpha):
    chi2_b_values[i] = chi2_b(mu_alpha[:,i], alpha_values[i])
    print(i)

# we select the index at which chi2 is equal to 1
# and we look at which value of alpha and mu corresponds
# in order to find nearest value in numpy array
# we use (np.abs(array-value)).argmin()
index_chi2_b_optimal = np.argmin(np.abs(chi2_b_values-1.))
mu_chi2_b_opt = mu_alpha[:,index_chi2_b_optimal]
alpha_chi2_b_opt = alpha_values[index_chi2_b_optimal]
Delta_log_L_opt = Delta_log_L[index_chi2_b_optimal]

plt.plot(Delta_log_L , chi2_b_values, color="black", lw=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,20, 0.2), minor=True)
ax.set_xticks(np.arange(0,20, 0.25), minor=True)
plt.yticks(np.arange(0,30))
plt.xticks(np.arange(0,20,1))
plt.ylim(0,10)
plt.xlim(0,12)
plt.xlabel(r"$\Delta \log L$")
plt.ylabel(r"$\chi^2_{\mathrm{b}}$")
#leg = plt.legend(loc="lower center")
#leg.get_frame().set_edgecolor('black')

#we plot lines:
plt.axhline(y=1,ls="dashed",color="black",lw=0.5,)
plt.axvline(x=Delta_log_L_opt,ls="dashed",color="black",lw=0.5,)

################# PART E
fig = plt.figure()

mu_center = np.linspace(0,100,20)
plt.hist(mu_center, bins=M, weights=mu_exercise1, color="black", histtype="step", normed = False)
plt.hist(mu_center, bins=N, weights=n, color="black", histtype="step", normed = False, ls ="dashed")

# we compute with the optimal values of mu and alpha
cov_U = cov_U_matrix_computation(mu_chi2_b_opt,alpha_chi2_b_opt)
standard_deviation = np.sqrt(np.diag(cov_U))
mu_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_center, mu_chi2_b_opt, yerr=standard_deviation, xerr=width_bins/2,
             fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(0,5000,200), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.ylim(0,5000)
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.xlabel("x")
plt.ylabel(r"n, $\mu$, $\hat{\mu}$")

################ PART F

fig = plt.figure()
# we compute the bias and its covariance
bias = bias_computation(mu_chi2_b_opt, alpha_chi2_b_opt)
cov_W = cov_W_matrix_computation(mu_chi2_b_opt,alpha_chi2_b_opt)
standard_deviation = np.sqrt(np.diag(cov_W))

mu_0_center = np.linspace(2.5,97.5,20)
plt.errorbar(mu_0_center, bias, yerr=standard_deviation, xerr=width_bins/2, fmt="none", color = "black", elinewidth=1)

ax = fig.add_subplot(1, 1, 1)
ax.set_yticks(np.arange(-300,400,20), minor=True)
ax.set_xticks(np.arange(0,102,2), minor=True)
plt.yticks(np.arange(-300,400,100))
plt.xticks(np.arange(0,110,10))
plt.xlim(0,100)
plt.ylim(-300,300)
plt.xlabel("x")
plt.ylabel(r"$\hat{b}$")
#we plot lines:
plt.axhline(y=0,ls="dashed",color="black",lw=0.5,)

plt.show()
"""




