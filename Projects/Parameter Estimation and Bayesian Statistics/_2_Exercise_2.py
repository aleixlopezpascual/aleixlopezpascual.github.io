import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt
from math import sqrt, factorial

exercise = 15

#################################################################
# exercise 2.2

if exercise == 2:
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 150, sqrt(150)), color="black")

    plt.xlabel(r"$\hat{\nu}$")
    plt.ylabel(r"f($\hat{\nu}$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.show()

#################################################################
# exercise 2.3

if exercise == 3:
    q = 150 # estimator from the first experiment, we use it as "true" value for the following MC's

    # we now genetare 10^6 MC experiments from a Poisson pdf. There is
    # a module, so-called numpy.random, which implements pseudo-random number generators for various distributions.
    # since we want to generate numbers Poisson distributed, we can directly use this module.
    poisson_data = np.random.poisson(q, 10**6)
    print(poisson_data)
    print(len(poisson_data))
    # the obtained values are directly N_i.
    # Now we compute the estimator for each one, which in fact corresponds to q_i = N_i
    # so we don't have to make any change
    plt.hist(poisson_data, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 150, sqrt(150)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (150,150)$")

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(poisson_data)
    variance_estimator = np.var(poisson_data, ddof=1)
    print(mean_estimator, variance_estimator)

    plt.xlabel(r"$\hat{\nu}$")
    plt.ylabel(r"N($\hat{\nu}$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

#################################################################
# exercise 2.4

if exercise == 4:
    N = 150
    nu = np.linspace(100,200,100000)
#   posterior_pdf = ((nu**N)*np.exp(-nu))/special.gamma(N+1)
#   be careful here. This last expresion overflows when n=113 since result exp(308)
#   Remember that Max normal number with 64 bit floating point is 1.7x10308
#   so we have to redefine the calculations
    posterior_pdf = ((nu**(N/2))*(np.exp(-nu))*(nu**(N/2)))/special.gamma(N+1)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|150)")
    plt.plot(nu, stats.norm.pdf(nu, 150, sqrt(150)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (150,150)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|150)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()


#################################################################
# exercise 2.5

if exercise == 5:
    N_obs=150
    nu = np.random.uniform(0.0, N_obs + 10.0*sqrt(N_obs), size=10**7) # we generate nu that satisfies our prior
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 150, sqrt(150)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (150,150)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))
# mean = 150.91 variance 150.56
    # 151.06,151.09
    # 150.95,152.57
    # 151.04,151.19
    # revise exercise 2.4: find mode and verify if its 150 o 151


    plt.figure(2)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (151,151)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()

#################################################################
# exercise 2.6 (if 6 -> repeat ex 2.4. if 7 ex -> 7 repeat ex 2.5)

if exercise == 6:
    N = 10
    nu = np.linspace(0,50,100000)
#   posterior_pdf = ((nu**N)*np.exp(-nu))/special.gamma(N+1)
#   be careful here. This last expresion overflows when n=113 since result exp(308)
#   Remember that Max normal number with 64 bit floating point is 1.7x10308
#   so we have to redefine the calculations
    posterior_pdf = ((nu**(N/2))*(np.exp(-nu))*(nu**(N/2)))/special.gamma(N+1)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|10)")
    plt.plot(nu, stats.norm.pdf(nu, 10, sqrt(10)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (10,10)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|10)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

if exercise == 7:
    N_obs=10
    nu = np.random.uniform(0.0, N_obs + 10.0*sqrt(N_obs), size=10**7) # we generate nu that satisfies our prior
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 10, sqrt(10)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (10,10)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (11,11)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()

#################################################################
# exercise 2.7 (if 8 -> repeat ex 2.4. if 9 ex -> 7 repeat ex 2.5)

if exercise == 8:
    N = 1
    nu = np.linspace(0,10,10000)
#   posterior_pdf = ((nu**N)*np.exp(-nu))/special.gamma(N+1)
#   be careful here. This last expresion overflows when n=113 since result exp(308)
#   Remember that Max normal number with 64 bit floating point is 1.7x10308
#   so we have to redefine the calculations
    posterior_pdf = ((nu**(N/2))*(np.exp(-nu))*(nu**(N/2)))/special.gamma(N+1)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|1)")
    plt.plot(nu, stats.norm.pdf(nu, 1, sqrt(1)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (1,1)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|1)")
    plt.xlim(0,5)
    plt.ylim(0, 0.45)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

if exercise == 9:
    N_obs=1
    nu = np.random.uniform(0.0, N_obs + 10.0*sqrt(N_obs), size=10**7) # we generate nu that satisfies our prior
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 1, sqrt(1)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (1,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 0.45)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (1,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 0.45)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()


#################################################################
#################################################################
#################################################################
# exercise 2.8

if exercise == 10:
    N = 150
    nu = np.linspace(100,200,100000)
    posterior_pdf = ((nu**((N-1)/2))*(np.exp(-nu))*(nu**((N-1)/2)))/special.gamma(N)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|150)")
    plt.plot(nu, stats.norm.pdf(nu, 150, sqrt(150)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (150,150)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|150)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()


#################################################################

if exercise == 11:
    N_obs=150
    # we generate nu that satisfies our prior.
    # we have used the inv transformation MC method
    # remember that we always use r = U[0,1] in this method
    nu = np.exp(np.random.uniform(0.0, np.log(N_obs + 10.0*sqrt(N_obs)), size=10**7))
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count, nu_i)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 150, sqrt(150)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (150,150)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (151,151)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(100,200)
    plt.ylim(0, 0.035)
    plt.xticks(np.arange(100,210,10))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()

#################################################################
# exercise 2.6 (if 6 -> repeat ex 2.4. if 7 ex -> 7 repeat ex 2.5)

if exercise == 12:
    N = 10
    nu = np.linspace(0,50,100000)
#   posterior_pdf = ((nu**N)*np.exp(-nu))/special.gamma(N+1)
#   be careful here. This last expresion overflows when n=113 since result exp(308)
#   Remember that Max normal number with 64 bit floating point is 1.7x10308
#   so we have to redefine the calculations
    posterior_pdf = ((nu**((N-1)/2))*(np.exp(-nu))*(nu**((N-1)/2)))/special.gamma(N)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|10)")
    plt.plot(nu, stats.norm.pdf(nu, 10, sqrt(10)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (10,10)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|10)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

if exercise == 13:
    N_obs=10
    nu = np.exp(np.random.uniform(0.0, np.log(N_obs + 10.0*sqrt(N_obs)), size=10**7)) # we generate nu that satisfies our prior
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 10, sqrt(10)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (10,10)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 30, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (11,11)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,20)
    plt.ylim(0, 0.14)
    plt.xticks(np.arange(0,22,2))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()

#################################################################
# exercise 2.7 (if 8 -> repeat ex 2.4. if 9 ex -> 7 repeat ex 2.5)

if exercise == 14:
    N = 1
    nu = np.linspace(0,10,10000)
#   posterior_pdf = ((nu**N)*np.exp(-nu))/special.gamma(N+1)
#   be careful here. This last expresion overflows when n=113 since result exp(308)
#   Remember that Max normal number with 64 bit floating point is 1.7x10308
#   so we have to redefine the calculations
    posterior_pdf = ((nu**((N-1)/2))*(np.exp(-nu))*(nu**((N-1)/2)))/special.gamma(N)
    plt.plot(nu, posterior_pdf, color = "black", linewidth= "1", label = r"P($\nu$|1)")
    plt.plot(nu, stats.norm.pdf(nu, 1, sqrt(1)), color="red", linewidth= "1",
             label=r"$\mathcal{N} \ (1,1)$")
    print(np.argmax(posterior_pdf))
    print(nu[np.argmax(posterior_pdf)])

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"P($\nu$|1)")
    plt.xlim(0,5)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

if exercise == 15:
    N_obs=1
    nu = np.exp(np.random.uniform(-100., np.log(N_obs + 10.0*sqrt(N_obs)), size=10**7)) # we generate nu that satisfies our prior
#    N = np.array([])
    count = 0
    nu_accepted = np.array([])
    for nu_i in nu:
        N = np.random.poisson(nu_i, 1)  # we generate one N for each value of nu
        if N == N_obs:
            nu_accepted = np.append(nu_accepted, nu_i)
        count += 1
        print(count)

    plt.figure(1)
    plt.hist(nu_accepted, bins= 40, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 1, sqrt(1)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (1,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 0.60)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 50, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (1,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    plt.show()

    np.savetxt("nu_accepted", nu_accepted)


if exercise == 16:
    nu_accepted=np.loadtxt("nu_accepted")

    plt.figure(1)
    plt.hist(nu_accepted, bins= 40, normed=True, histtype="step", color="black")

    # we compare it with the result in exercise 2.2
    x_axis = np.linspace(0,300, 10000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, 1, sqrt(1)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (1,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 0.60)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')

    # we compute the estimators of the mean and variance from poisson_data
    mean_estimator = np.mean(nu_accepted)
    variance_estimator = np.var(nu_accepted, ddof=1)
    print(mean_estimator, variance_estimator)
    print("{0:.2f},{1:.2f}".format(mean_estimator, variance_estimator))

    plt.figure(2)
    plt.hist(nu_accepted, bins= 50, normed=True, histtype="step", color="black")
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean_estimator, sqrt(variance_estimator)), color="red", linewidth = "1",
             label= r"$\mathcal{N} \ (2,1)$")

    plt.xlabel(r"$\nu$")
    plt.ylabel(r"N($\nu$)")
    plt.xlim(0,5)
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0,5.5,0.5))

    plt.legend(loc="best")
    leg = plt.legend()
    leg.get_frame().set_edgecolor('black')


    plt.show()