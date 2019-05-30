import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

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

def chi_square_variable_i(N_measurements, counts_bins, mean_i, norm):
    return ((counts_bins-norm*mean_i)**2)/(norm*mean_i*(1-norm*mean_i/N_measurements))

def normalization_histogram(counts_bins, edges_bins):
    # this is not a normalization constant, it is the integral over the histogram
    # note that it is also different from N_measurements.
	out=0.
	spacing = edges_bins[1]-edges_bins[0]
	for i in range(0,len(counts_bins)-1):
		out += counts_bins[i] * spacing
	return out


N_measurements=10000
P_mu=0.5
N_experiments_max=10000
N_experiments = 0
N_bins=50

#First we do an experiment to compute mu_i ---------------------------------------
#we use a great number of measurements because we want the maximum ajust to the theory
data_pdf = inv_trans(1000000, P_mu)

#since we do not want to display the histogram, but only compute the results we use numpy
counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=False)

#array with the center of bins. this array is equal for each monte carlo with bins fixed
center_bins=np.array([])
for i in np.arange(0,len(edges_bins)-1,1):
    center_bins=np.append(center_bins, (edges_bins[i] + edges_bins[i+1])/2.)

#array with the means_i from theory
mean_i = pdf(center_bins, P_mu)
#print(mean_i)

# --------------------------------------------------------------------------------
chi_square_variable = np.array([])

# we compute the chi^2 variable for each experiment
while N_experiments < N_experiments_max:
    sum_chi_square = 0
    data_pdf = inv_trans(N_measurements, P_mu)
    counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=False)
    norm = normalization_histogram(counts_bins,edges_bins)
    for i in range (0, N_bins):
        sum_chi_square += chi_square_variable_i(N_measurements, counts_bins[i], mean_i[i], norm) #we sum for all the bins
    chi_square_variable = np.append(chi_square_variable, sum_chi_square) #we save the final chi_square_variable
    N_experiments += 1
    print(N_experiments)

#plot the chi histogram
plt.hist(chi_square_variable, bins=N_bins, normed=True, histtype="step", color="black", linewidth = "1")
plt.xlabel(r'$\chi^2$')
plt.ylabel(r'$N(\chi^2)$')

#plot the theoretical pdfs
x_axis=np.linspace(0, np.max(chi_square_variable), 10000)
plt.plot(x_axis, stats.chi2.pdf(x_axis, N_bins-1), color="orange", linewidth="1", label ="$\chi^2-distribution$")
plt.plot(x_axis, stats.norm.pdf(x_axis, loc=N_bins, scale=sqrt(2*N_bins)), color="red", linewidth="1", label = r"$\mathcal{N} \ (N,\sqrt{2N})$")

plt.xlim(20,80)
plt.text(22, 0.030, r"$N_{bins}=$" + str(N_bins))
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')



plt.show()