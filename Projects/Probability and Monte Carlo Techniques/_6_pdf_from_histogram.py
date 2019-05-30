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


N_measurements=10000
P_mu=0.5

data_pdf = inv_trans(N_measurements, P_mu)
#print(len(data_pdf))
counts_bins, edges_bins, patches = plt.hist(data_pdf, bins=20, normed=True, histtype="step", color="black", linewidth = "1")


#print(counts_bins)
#print(edges_bins)

bin_width=abs(edges_bins[7]-edges_bins[8])
#print(bin_width)

#Note that the normalization is not to 1.
normalisation_bins=np.sum(counts_bins)
#print(normalisation_bins)

pdf_estimator = counts_bins/(normalisation_bins*bin_width)
#print(pdf_estimator)

center_bins=np.array([])
for i in np.arange(0,len(edges_bins)-1,1):
    center_bins=np.append(center_bins, (edges_bins[i] + edges_bins[i+1])/2.)
#print(center_bins)

plt.plot(center_bins, pdf_estimator, color = "blue", linewidth = 1, label = "pdf estimation")

x_axis = np.linspace(-1.0,1.0, 1000)
plt.plot(x_axis, pdf(x_axis,P_mu), color="red", linestyle="--", linewidth=1, label ="pdf theory")

plt.xlim(-1.,1.)
plt.ylim(0.4,0.6)
plt.xlabel(r'$cos \ \theta$')
plt.ylabel(r'$N(\cos \ \theta)$')
plt.text(-0.9, 0.41, r"$N_{events}=$" + str(N_measurements))
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()