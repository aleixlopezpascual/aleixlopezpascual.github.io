# Inverse method using Monte Carlo

import numpy as np
import matplotlib.pyplot as plt

P_mu=1.0
N_measures=10000000

def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)

def inv_cdf_pos(r, P_mu):
    # inverse of the cumulative density function
    # since we have positive and negative solutions for x(r), we have to consider both
    return 3./P_mu*(1.+np.sqrt(1.+2./3.*P_mu*(P_mu/6.+1.-2.*r)))

def inv_cdf_neg(r, P_mu):
    return 3./P_mu*(1.-np.sqrt(1.+2./3.*P_mu*(P_mu/6.+1.-2.*r)))

# generate random float numbers between [0,1] uniformly distributed
# note that np.random.random() generates [0,1)
r = np.random.uniform(0.0,1.0, size=N_measures)

# plot an histogram, since we want to know how many values there are for each output in order to reconstruct the pdf
# we had two possible solutions, however when we plot them, we find that only one of them gives us the correct solution
data_pdf=inv_cdf_neg(r,P_mu)
plt.hist(data_pdf, bins=100, normed=True, histtype="step", color="black")

#we also plot the original pdf
x_axis = np.linspace(-1.0,1.0, 1000)
plt.plot(x_axis,pdf(x_axis,P_mu), color="orange", linewidth="1")

plt.xlabel(r'$cos \ \theta$')
plt.ylabel('f')
plt.xlim(-1.,1.)
plt.ylim(0.3,0.7)

print(len(data_pdf))
# plt.savefig(r"C:\Users\Aleix López\Desktop\inv_trans.jpg")
plt.show()

