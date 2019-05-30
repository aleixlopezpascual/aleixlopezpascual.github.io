import numpy as np
import matplotlib.pyplot as plt

def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)

def acc_rej(N_measurements, P_mu):
    x = np.random.uniform(-1.0, 1.0, size=N_measurements)

    x_axis = np.linspace(-1.0,1.0, 1000)
    fmax= np.amax(pdf(x_axis,P_mu)) #find the maximum of the function

    u = np.random.uniform(0, fmax, size= N_measurements)

    # we use a mask in order to reject the values we don't want
    data_pdf = x[u < pdf(x,P_mu)]
    return data_pdf


N_measurements=100000

first_experiment=acc_rej(N_measurements,0.5)
mean = np.mean(first_experiment)
P_mu_first=-9.*mean
print(P_mu_first)

P_mu_est = np.array([])
N_experiments=0
while N_experiments < 1000:
    data_pdf = acc_rej(N_measurements, P_mu_first)
    mean = np.mean(data_pdf)
    P_mu_est = np.append(P_mu_est,-9. * mean)
    N_experiments += 1

variance = np.var(P_mu_est)
sigma = np.sqrt(variance)
print(variance)
print(sigma)

plt.hist(P_mu_est, bins=30, histtype="step", color="black", linewidth="1")
plt.xlabel(r'$\hat{P}_\mu$')
plt.ylabel(r'$N(\hat{P}_\mu)$')

# plt.savefig(r"C:\Users\Aleix López\Desktop\acc_rej.jpg")
plt.show()