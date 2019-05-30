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

def t_variable(N_events, sample_mean, mean, unbiased_sample_mean):
    return sqrt(N_events)*(sample_mean-mean)/sqrt(unbiased_sample_mean)

def g_variable(N_events, sample_mean, mean, biased_sample_mean):
    return sqrt(N_events)*(sample_mean-mean)/sqrt(biased_sample_mean)

t = np.array([])
g = np.array([])
N_experiments = 0
N_experiments_max = 10000
N_measurements = 1000
P_mu=0.5
mean = -P_mu/9.

while N_experiments < N_experiments_max:
    data_pdf = inv_trans(N_measurements, P_mu) #the inv_trans methods works quite better because takes all the points
    sample_mean = np.mean(data_pdf)
    unbiased_sample_mean = np.var(data_pdf, ddof=1)
    biased_sample_mean = np.var(data_pdf, ddof=0)
    t = np.append(t, t_variable(N_measurements, sample_mean, mean, unbiased_sample_mean))
#    g = np.append(g, g_variable(N_measurements, sample_mean, mean, biased_sample_mean))
    N_experiments += 1
    print(N_experiments)

plt.hist(t, bins=30, normed=True, histtype="step", color="black", linewidth="1")
plt.xlabel(r't')
plt.ylabel(r'$N(t)$')
plt.xlim(-4,4)

#plt.hist(g, bins=30, normed=True, histtype="step", color="black", linewidth="1")
#plt.xlabel(r'g')
#plt.ylabel(r'$N(g)$')

x_axis=np.linspace(-20, 20, 1000000)
plt.plot(x_axis, stats.t.pdf(x_axis, N_measurements-1), color="orange", linewidth="1", label ="t-distribution")
plt.plot(x_axis, stats.norm.pdf(x_axis, loc=0, scale=1), color="red", linewidth="1", label = r"$\mathcal{N} \ (0,1)$")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
#plt.text(-3.5, 0.38, r"$N_{events}=$" + str(N_measurements))

mean_t= np.mean(t)
var_t= np.var(t)
skew_t = skew(t)
kurt_t = kurtosis(t, fisher=True)

print("MC:", mean_t, var_t, skew_t, kurt_t)

mean_g= np.mean(g)
var_g= np.var(g)
skew_g = skew(g)
kurt_g = kurtosis(g, fisher=True)

print("MC:", mean_g, var_g, skew_g, kurt_g)
#print("Theory:", 0,(N_measurements-1)/(N_measurements-3), 0, 6/(N_measurements-5))

plt.show()

