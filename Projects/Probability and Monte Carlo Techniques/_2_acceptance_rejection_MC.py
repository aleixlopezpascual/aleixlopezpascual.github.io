# Acceptance-Rejection method using Monte Carlo

import numpy as np
import matplotlib.pyplot as plt

def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)


N_measures=10000000
P_mu=1.0

x = np.random.uniform(-1.0, 1.0, size=N_measures)

x_axis = np.linspace(-1.0,1.0, 1000)
fmax= np.amax(pdf(x_axis,P_mu)) #find the maximum of the function

u = np.random.uniform(0, fmax, size= N_measures)

# we use a mask in order to reject the values we don't want
data_pdf = x[u < pdf(x,P_mu)]

# look out, the number of bins cannot be very large, otherwise the histogram doesn't fit the function.
plt.hist(data_pdf, bins=100, normed=True, histtype="step", color="black")
plt.plot(x_axis,pdf(x_axis,P_mu), color="orange", linewidth="1")
plt.xlabel(r'$cos \ \theta$')
plt.ylabel('f')
plt.xlim(-1.,1.)
plt.ylim(0.3,0.7)

print(len(data_pdf))
# plt.savefig(r"C:\Users\Aleix López\Desktop\acc_rej.jpg")
plt.show()




