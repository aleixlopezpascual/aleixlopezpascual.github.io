import numpy as np
from scipy.stats import moment, kurtosis, skew
# from _2_acceptance_rejection_MC import data_pdf
from _2_inverse_transf_MC import data_pdf

mean = np.mean(data_pdf)
variance = np.var(data_pdf)
#variance_2 = moment(data_pdf, moment=2) it's equivalent to the previous function
skewness = skew(data_pdf)
kurtosis = kurtosis(data_pdf, fisher=True)

print(mean, variance, skewness, kurtosis)