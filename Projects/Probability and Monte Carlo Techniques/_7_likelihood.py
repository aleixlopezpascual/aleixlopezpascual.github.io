import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy import stats

def pdf(costh, P_mu):
    # define our probability density function
    return 0.5 * (1.0 - 1.0 / 3.0 * P_mu * costh)

def pdf_integral (costh, P_mu):
    return 0.5*costh - (P_mu*costh**2)/6

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

def chi_square_variable_i_2(counts_bins, mean_i):
    return ((counts_bins-mean_i)**2)/(mean_i)

def normalization_histogram(counts_bins, edges_bins):
    # this is not a normalization constant, it is the integral over the histogram
    # note that it is also different from N_measurements.
	out=0.
	spacing = edges_bins[1]-edges_bins[0]
	for i in range(0,len(counts_bins)-1):
		out += counts_bins[i] * spacing
	return out


N_measurements=10000000 #this number must be high
P_mu=0.5
N_bins= 50 #this number doesn't matter, 50 is fine
normed = False # set histograms normed true or false. better use normed false.

method = 3

#First we do an experiment with a true value P_mu to compute the expected values mu_i and sigma_i per bin
#we use a great number of measurements because we want the maximum ajust to the theory
data_pdf = inv_trans(N_measurements, P_mu)

#since we do not want to display the histogram, but only compute the results we use numpy
counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=normed)

#these mean_i and var_i are the values of the expected histogram (the one we want)
mean_i = counts_bins
var_i = counts_bins

#array with the center of bins. this array is equal for each monte carlo with bins fixed
center_bins=np.array([])
for i in np.arange(0,len(edges_bins)-1,1):
    center_bins=np.append(center_bins, (edges_bins[i] + edges_bins[i+1])/2.)

#we will use norm in method 4
norm = normalization_histogram(counts_bins, edges_bins)

#we distinguish differents methods

#######################################################################################################################
if method == 2: #this is the method that works better for me, however it is not exactly the correct way to do it
    # using n_i from different MC with differents P_mu in multivariate gaussian

    P_mu_axis = np.array([])
    multi_gaussian_axis = np.array([])

    for P_mu_i in np.arange(0.01, 1.01, 0.005): # the step sets the precision from which we get p_mu
        P_mu_axis = np.append(P_mu_axis, P_mu_i)
        data_pdf = inv_trans(N_measurements, P_mu_i)
        counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=normed) # remember counts_bins is an array not a number
        multi_gaussian = stats.multivariate_normal.pdf(counts_bins, mean=mean_i, cov=var_i)
        multi_gaussian_axis = np.append(multi_gaussian_axis, multi_gaussian)
        print(P_mu_i)

    plt.figure(1)
    plt.plot(P_mu_axis, multi_gaussian_axis, color = "black", linewidth = 1)
    plt.ylim(0)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.figure(2)
    plt.plot(P_mu_axis, multi_gaussian_axis, color="black", linewidth=1)
    plt.ylim(0)
    plt.xlim(0.4, 0.6)
    plt.xticks(np.arange(0.4,0.62, 0.02))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    # find the maximum
    index_max = np.argmax(multi_gaussian_axis)
    maximum = P_mu_axis[index_max]
    print(maximum)

    plt.show()



#######################################################################################################################
if method == 3:
# ------- computing the chi^2 then we compute the likelihood from here ----------------------------------------------------
    P_mu_axis = np.array([])
    chi_square_variable = np.array([])

    for P_mu_i in np.arange(0.01, 1.01, 0.01):
        sum_chi_square = 0
        P_mu_axis = np.append(P_mu_axis, P_mu_i)
        data_pdf = inv_trans(N_measurements, P_mu_i)
        counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=normed)
        for i in range (0, N_bins):
            sum_chi_square += chi_square_variable_i_2(counts_bins[i], mean_i[i]) #we sum for all the bins
        chi_square_variable = np.append(chi_square_variable, sum_chi_square) #we save the final chi_square_variable
        print(P_mu_i)

    plt.figure(1)
    plt.plot(P_mu_axis, chi_square_variable, color = "black", linewidth = 1)
    plt.ylim(0)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$\chi^2(P_\mu)$")

    plt.figure(2)
    likelihood = np.exp(-chi_square_variable/2.)
    plt.plot(P_mu_axis, likelihood)
    plt.ylim(0)
    plt.xlim(0.4, 0.6)
    plt.xticks(np.arange(0.4,0.62, 0.02))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.show()


#######################################################################################################################
if method == 4: # doesn't work properly
    # similar to method 2 using the multivariate gaussian but using only one histogram.
    # from the pdf formula we obtain mu_i and sigma_i (expected values) using x as the
    # central value in the bin i and multiplying over the histogram
    # y_i are obtained from the histogram, are the counts

    P_mu_axis = np.array([])
    multi_gaussian_axis = np.array([])
    y_i = mean_i #what we denoted as mean_i for the histogram, now we denote it as y_i. It's an array

    for P_mu_i in np.arange(0.01, 1.001, 0.01):
        P_mu_axis = np.append(P_mu_axis, P_mu_i)
        mean_i = pdf(center_bins, P_mu_i)*norm #mean_i is already an array since center_bins is an array
        # remember form exercise 6 that we have to integrate this values over the histogram (multiply by norm)
        print(y_i)
        print(mean_i)
        multi_gaussian = stats.multivariate_normal.pdf(y_i, mean=mean_i, cov=mean_i)
        print(multi_gaussian)
        multi_gaussian_axis = np.append(multi_gaussian_axis, multi_gaussian)
        print(P_mu_i)

    plt.figure(1)
    plt.plot(P_mu_axis, multi_gaussian_axis, color = "black", linewidth = 1)
    plt.ylim(0)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.figure(2)
    plt.plot(P_mu_axis, multi_gaussian_axis, color="black", linewidth=1)
    plt.ylim(0)
    plt.xlim(0.4, 0.6)
    plt.xticks(np.arange(0.4,0.62, 0.02))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.show()

#######################################################################################################################
if method == 5:
    # similar to method 2 but instead of using a multivariate gaussian distribution, we compute the likelihood
    # without using this function, since actually the product of unvariate normals is not a multivaraite normal.
    # we obtain the same plot than method 2. so it was already ok.

    P_mu_axis = np.array([])
    likelihood_axis = np.array([]) #we have a value for each P_mu

    for P_mu_i in np.arange(0.01, 1.01, 0.005): # the step sets the precision from which we get p_mu
        P_mu_axis = np.append(P_mu_axis, P_mu_i)
        data_pdf = inv_trans(N_measurements, P_mu_i)
        counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=normed) # remember counts_bins is an array not a number

        # we compute the likelihood for this P_mu_i

        likelihood_product = 1/(np.sqrt(2*pi*mean_i))*np.exp((-(counts_bins-mean_i)**2)/(2*mean_i)) #contains the elements we want to multiply (product operator)
        likelihood=np.prod(likelihood_product) #now we multiply all the elements of the array (product operator). the result is the likelihood for this P_mu_i

        likelihood_axis = np.append(likelihood_axis, likelihood)
        print(P_mu_i)

    plt.figure(1)
    plt.plot(P_mu_axis, likelihood_axis, color = "black", linewidth = 1)
    plt.ylim(0)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.figure(2)
    plt.plot(P_mu_axis, likelihood_axis, color="black", linewidth=1)
    plt.ylim(0)
    plt.xlim(0.4, 0.6)
    plt.xticks(np.arange(0.4,0.62, 0.02))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    # find the maximum
    index_max = np.argmax(likelihood_axis)
    maximum = P_mu_axis[index_max]
    print(maximum)

    plt.show()



if method ==6:

    P_mu_axis = np.array([])
    likelihood_axis = np.array([])  # we have a value for each P_mu


    for P_mu_i in np.arange(0.01, 1.01, 0.005):  # the step sets the precision from which we get p_mu
        P_mu_axis = np.append(P_mu_axis, P_mu_i)
        data_pdf = inv_trans(N_measurements, P_mu_i)
        counts_bins, edges_bins = np.histogram(data_pdf, bins=N_bins, density=normed)  # remember counts_bins is an array not a number

        mean_i = np.array([])
        for i in np.arange(0, N_bins):
            mean_i_term = pdf_integral(edges_bins[i], P_mu_i) - pdf_integral(edges_bins[i + 1], P_mu_i) * N_measurements
            mean_i = np.append(mean_i, mean_i_term)

        # we compute the likelihood for this P_mu_i

        likelihood_product = 1 / (np.sqrt(2 * pi * mean_i)) * np.exp((-(counts_bins - mean_i) ** 2) / (2 * mean_i))  # contains the elements we want to multiply (product operator)
        likelihood = np.prod(likelihood_product)  # now we multiply all the elements of the array (product operator). the result is the likelihood for this P_mu_i

        likelihood_axis = np.append(likelihood_axis, likelihood)
        print(P_mu_i)

    plt.figure(1)
    plt.plot(P_mu_axis, likelihood_axis, color="black", linewidth=1)
    plt.ylim(0)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    plt.figure(2)
    plt.plot(P_mu_axis, likelihood_axis, color="black", linewidth=1)
    plt.ylim(0)
    plt.xlim(0.4, 0.6)
    plt.xticks(np.arange(0.4, 0.62, 0.02))
    plt.xlabel(r"$P_\mu$")
    plt.ylabel(r"$L(P_\mu)$")

    # find the maximum
    index_max = np.argmax(likelihood_axis)
    maximum = P_mu_axis[index_max]
    print(maximum)

    plt.show()

