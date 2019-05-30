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


N_measurements=10000000

for P_mu in np.arange(-1.,1.2,0.2):
    data_pdf = acc_rej(N_measurements,P_mu)
    mean = np.mean(data_pdf)
    P_mu_est=-9.*mean
    print(P_mu, abs(P_mu-P_mu_est))
    plt.plot(P_mu, abs(P_mu-P_mu_est), ".")


plt.xlabel(r"$P_\mu$")
plt.ylabel(r'$|P_\mu - \hat{P_\mu}|$')

# plt.savefig(r"C:\Users\Aleix López\Desktop\acc_rej.jpg")
plt.show()