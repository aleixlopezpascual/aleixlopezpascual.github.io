import numpy as np
import matplotlib.pyplot as plt


def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)


def acc_rej(N_measurements, P_mu):
    x = np.random.uniform(-1.0, 1.0, size=N_measurements)

    x_axis = np.linspace(-1.0, 1.0, 1000)
    fmax = np.amax(pdf(x_axis, P_mu))  # find the maximum of the function

    u = np.random.uniform(0, fmax, size=N_measurements)

    # we use a mask in order to reject the values we don't want
    data_pdf = x[u < pdf(x, P_mu)]
    return data_pdf


P_mu = 0.5
N_measurements_max = 10000000

P_mu_est=np.array([])
data_N_measurements=np.array([])

for N_measurements in np.arange(1, N_measurements_max+10000, 10000):
    data_pdf = acc_rej(N_measurements, P_mu)
    mean = np.mean(data_pdf)
    P_mu_est = np.append(P_mu_est,-9. * mean)
    data_N_measurements = np.append(data_N_measurements, N_measurements)
    print(N_measurements)

np.save("P_mu_est", P_mu_est)
np.save("data_N_measurements", data_N_measurements)
P_mu_est = np.load("P_mu_est.npy")
data_N_measurements = np.load("data_N_measurements.npy")

plt.plot(data_N_measurements, P_mu_est, "k", linewidth=0.5, label=r"$\hat{P}_\mu$")
plt.plot([1, N_measurements_max], [P_mu, P_mu], "orange", linestyle='-', linewidth=1.0, label=r"$P_\mu$")
plt.xlabel(r"$N_{events}$")
plt.ylabel(r'$\hat{P}_\mu}$')
plt.xlim(1, N_measurements_max)
plt.ylim(0.46, 0.54)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

# plt.savefig(r"C:\Users\Aleix López\Desktop\acc_rej.jpg")
plt.show()
