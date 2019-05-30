import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from scipy.stats import kurtosis, skew

def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)

def inv_cdf_pos(r, P_mu):
    # inverse of the cumulative density function
    # since we have positive and negative solutions for x(r), we have to consider both
    return 3./P_mu*(1.+np.sqrt(1.+2./3.*P_mu*(P_mu/6.+1.-2.*r)))

def inv_cdf_neg(r, P_mu):
    return 3./P_mu*(1.-np.sqrt(1.+2./3.*P_mu*(P_mu/6.+1.-2.*r)))

def inv_trans(N_measurements, P_mu):
    r = np.random.uniform(0.0, 1.0, size=N_measurements)
    data_pdf = inv_cdf_neg(r, P_mu)
    return data_pdf

def acc_rej(N_measurements, P_mu):
    x = np.random.uniform(-1.0, 1.0, size=N_measurements)

    x_axis = np.linspace(-1.0,1.0, 1000)
    fmax= np.amax(pdf(x_axis,P_mu)) #find the maximum of the function

    u = np.random.uniform(0, fmax, size= N_measurements)

    # we use a mask in order to reject the values we don't want
    data_pdf = x[u < pdf(x,P_mu)]
    return data_pdf


N_measurements=100
P_mu=0.5
N_experiments_max=10000
N_experiments = 0
N_bins=50
number_of_bin=25

#changing the value of the bin we obtain the mean in a different point of the pdf (not x), ie the y axis.

single_bin = np.array([])

while N_experiments < N_experiments_max:
    data_pdf = inv_trans(N_measurements, P_mu)
    counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=True)
    single_bin=np.append(single_bin, counts_bins[number_of_bin])
    N_experiments += 1

center_bin = (edges_bins[number_of_bin] + edges_bins[number_of_bin+1])/2.
parameter = pdf(center_bin, P_mu)
print(parameter)

plt.hist(single_bin, 12, normed=True, histtype="step", color="black", linewidth = "1")
#x_axis=np.linspace(0.4, 0.6, 100)
#plt.plot(x_axis, stats.poisson.pmf(x_axis, parameter), color="orange", linewidth="1", label ="$Poisson distribution$")
#plt.plot(x_axis, stats.norm.pdf(x_axis, loc=parameter, scale=sqrt(parameter)), color="red", linewidth="1", label = r"$\mathcal{N} \ (N,\sqrt{2N})$")
plt.xlabel(r'$n_i$')
plt.ylabel(r'$N(n_i)$')
plt.xlim(0, 2)

plt.show()

