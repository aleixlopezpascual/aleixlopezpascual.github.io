# Load Modules

# mathematical tools optimized for lists
import numpy as np
from numpy.linalg import inv
from scipy import stats
# tools for numerical integration
import scipy.integrate as pyint
# tools for numerical function minimization
import scipy.optimize as pyopt
# tools for plots
import matplotlib.pyplot as plt
# for neural network
import sklearn.preprocessing as preproc
import sklearn.neural_network as nn
from sklearn.externals import joblib
from math import factorial
from math import log
from math import sqrt

############################################################
# Read input files

trainSignalFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\train_signal.txt"
trainSignal = np.loadtxt(trainSignalFileName, skiprows=2)

# Extract x,y columns
trainSignalX = trainSignal[:,0]
trainSignalY = trainSignal[:,1]

#print("trainSignalX = ",trainSignalX,", with ",trainSignalX.size," elements")
#print("trainSignalY = ",trainSignalY,", with ",trainSignalY.size," elements")
#print("-----------------------")


trainBkgFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\train_bkg.txt"
trainBkg = np.loadtxt(trainBkgFileName, skiprows=2)

# Extract x,y columns
trainBkgX = trainBkg[:,0]
trainBkgY = trainBkg[:,1]


#print("trainBkgX = ",trainBkgX,", with ",trainBkgX.size," elements")
#print("trainBkgY = ",trainBkgY,", with ",trainBkgY.size," elements")
#print("-----------------------")


testSignalFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\test_signal.txt"
testSignal = np.loadtxt(testSignalFileName, skiprows=2)

# Extract x,y columns
testSignalX = testSignal[:,0]
testSignalY = testSignal[:,1]

#print("testSignalX = ",testSignalX,", with ",testSignalX.size," elements")
#print("testSignalY = ",testSignalY,", with ",testSignalY.size," elements")
#print("-----------------------")

testBkgFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\test_Bkg.txt"
testBkg = np.loadtxt(testBkgFileName, skiprows=2)

# Extract x,y columns
testBkgX = testBkg[:,0]
testBkgY = testBkg[:,1]

#print("testBkgX = ",testBkgX,", with ",testBkgX.size," elements")
#print("testBkgY = ",testBkgY,", with ",testBkgY.size," elements")
#print("-----------------------")



##################################################################################
#################################################################################
############################     PART A     ######################################
##################################################################################
# we construct the test statistics
# we use functions since we will use them more than once

def radial_distance(x,y):
    return np.sqrt(x**2 + y**2)

########################################################################################
# input: Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY)
def Fisher_discriminant(x_s, y_s, x_b, y_b, x, y):
    """
    x_s, y_s, x_b, y_b are the ones used to estimate the parameters, i.e.
    trainSignalX ,trainSignalY, trainBkgX, trainBkgY
    x,y are the data for which we want to compute the test statistic,
    can be signal, bkg, train or test.
    """

    # first we transform the data (x,y) to polar coordinates:
    r_s = np.sqrt(x_s**2 + y_s**2)
    theta_s = np.arctan(y_s/x_s) # despite the warning about dividing by zero, the division is computed
    r_b = np.sqrt(x_b**2 + y_b**2)
    theta_b = np.arctan(y_b/x_b)

    # then we estimate the free parameters:
    mu_0 = [np.mean(r_s), np.mean(theta_s)]
    mu_1 = [np.mean(r_b), np.mean(theta_b)]
    mu_0_x = [np.mean(x_s), np.mean(y_s)]
    mu_1_x = [np.mean(x_b), np.mean(y_b)]
    cov_0_x = np.cov(x_s, y_s)
    cov_0 = np.cov(r_s, theta_s)
    cov_1_x = np.cov(x_b, y_b)
    cov_1 = np.cov(r_b, theta_b)

    """
    print(mu_0, mu_1)
    print(mu_0_x, mu_1_x)
    print(cov_0)
    print(cov_0_x)
    print(cov_1)
    print(cov_1_x)
    """

    # we compute the difference
    mu_diff = np.array(mu_1) - np.array(mu_0)
    w = cov_0 + cov_1
    #print(mu_diff)
    #print(w)
    w_inv = inv(w)

    # we compute alpha
    alpha = np.dot(w_inv, mu_diff)
    #print(alpha)

    ##########
    # once we have the parameters, we compute the distribution of T:

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return alpha[0] * r + alpha[1] * theta

#############################################################################
def exact_likelihood_ratio(x_s, y_s, x_b, y_b):

    # Define and normalize function (pdf) with parameters
    def pdfSignalXY(x,y,N,a,b,c) :
        return 1./N*np.exp(-(a*x**2+b*y**2+2*c*x*y))

    # get normalization factors of pdfs of 2D:
    a, b, c = 6., 6., -5.

    # remember that pyint = scipy.integrate
    # scipy.integrate.dblquad Compute a double integral.
    # we integrate from x: -1 to 1 and from y: -1 to 1
    # a normalized pdf should give int = 1. If not int = N
    # Therefore, we compute N.
    # Then we just have to divide the pdf by this N (we already did in the def pdfSignalXY)

    NpdfSignalXY, Nerr = pyint.dblquad(pdfSignalXY,-1,1,
                    lambda x: -1,lambda x: 1,args=(1,a,b,c))

    #print("The normalization value of pdf(x,y) for signal is: {:.3f} (error is {:.3e})".format(NpdfSignalXY,Nerr))

    # Once we have the pdf complete, we can compute its values
    # use the normalization to compute pdf values at (x,y)=(xtest,ytest)
    #xtest, ytest = -0.3, 0.2
    #print("The value of pdf(x,y={0},{1}) for signal is = ".format(xtest,ytest),
    #      pdfSignalXY(xtest,ytest,NpdfSignalXY,a,b,c))

    # check that the normalization is properly computed
    # The integral should be 1

    #Norm, Error = pyint.dblquad(pdfSignalXY,-1,1,
    #            lambda x: -1,lambda x: 1,
    #            args=(NpdfSignalXY,a,b,c))
    #print("Once normalized the pdf normalization is: {:.3f} (error is {:.3e})".format(Norm,Error))
    #print("-----------------------")

    def pdfBkgXY(x,y,N,r_0, sigma_r):
        return 1./N*np.exp(-1./2.*((np.sqrt(x**2+y**2)-r_0)/sigma_r)**2)

    # get normalization factors of pdfs of 2D:
    r_0 = 0.6
    sigma_r = 0.4

    # we integrate from x: -1 to 1 and from y: -1 to 1

    NpdfBkgXY, Nerr = pyint.dblquad(pdfBkgXY,-1,1,
                    lambda x: -1,lambda x: 1,args=(1,r_0, sigma_r))

    #print("The normalization value of pdf(x,y) for bkg is: {:.3f} (error is {:.3e})".format(NpdfBkgXY,Nerr))

    # Once we have the pdf complete, we can compute its values
    # use the normalization to compute pdf values at (x,y)=(xtest,ytest)
    #xtest, ytest = -0.3, 0.2
    #print("The value of pdf(x,y={0},{1}) for bkg is = ".format(xtest,ytest),
    #      pdfBkgXY(xtest,ytest,NpdfBkgXY,r_0, sigma_r))

    # check that the normalization is properly computed
    # The integral should be 1

    #Norm, Error = pyint.dblquad(pdfBkgXY,-1,1,
    #            lambda x: -1,lambda x: 1,
    #            args=(NpdfBkgXY,r_0, sigma_r))
    #print("Once normalized the pdf normalization is: {:.3f} (error is {:.3e})".format(Norm,Error))
    #print("-----------------------")


    return pdfBkgXY(x_b,y_b,NpdfBkgXY,r_0, sigma_r)/pdfSignalXY(x_s,y_s,NpdfSignalXY,a,b,c)

#################################################################################################

def estimated_likelihood_ratio(data):

    # Define some constants
    nbinsx, nbinsy = 20, 20 #number of bins
    minx, maxx, miny, maxy = -1., 1., -1., 1. #domain x,y

    #########################################
    # 2D histogram of X and Y coordinates for signal:
    # we indicate the x and y datas, the bins and the ranges
    htrainSignalXY = np.histogram2d(trainSignalX, trainSignalY, [nbinsx, nbinsy],
                                    [[minx, maxx], [minx, maxx]])

    # this returns an array of shape (nx, ny), i.e. nx*ny bins
    """
    print("Entries in bins of X-Y histogram: ")
    print(htrainSignalXY[0])
    print("(", htrainSignalXY[0].size, " entries)")
    print("-----------------------")
    print("Bin edges of X in X-Y histogram:")
    print(htrainSignalXY[1])
    print("(", htrainSignalXY[1].size, " entries)")
    print("-----------------------")
    print("Bin edges of Y in X-Y histogram:")
    print(htrainSignalXY[2])
    print("(", htrainSignalXY[2].size, " entries)")
    print("-----------------------")
    """

    # now we are going to plots the 2D histogram:
    # we generate an array with the bin edges of both variables X,Y
    xx, yy = np.meshgrid(htrainSignalXY[1][:-1], htrainSignalXY[2][:-1])


    # flatten: Return a copy of the array collapsed into one dimension
    #plt.figure(1)
    #plt.hist2d(xx.flatten(), yy.flatten(), weights=htrainSignalXY[0].flatten(),
    #           bins=(htrainSignalXY[1], htrainSignalXY[2]))
    # note that in the bins parameter we do not discard the last value

    ##############
    # 2D histogram of X and Y coordinates for Background:
    # we indicate the x and y data, the bins and the ranges
    htrainBkgXY = np.histogram2d(trainBkgX, trainBkgY, [nbinsx, nbinsy],
                                    [[minx, maxx], [minx, maxx]])

    # this returns an array of shape (nx, ny), i.e. nx*ny bins
    """
    print("Entries in bins of X-Y histogram: ")
    print(htrainBkgXY[0])
    print("(", htrainBkgXY[0].size, " entries)")
    print("-----------------------")
    print("Bin edges of X in X-Y histogram:")
    print(htrainBkgXY[1])
    print("(", htrainBkgXY[1].size, " entries)")
    print("-----------------------")
    print("Bin edges of Y in X-Y histogram:")
    print(htrainBkgXY[2])
    print("(", htrainBkgXY[2].size, " entries)")
    print("-----------------------")
    """

    # now we are going to plot the 2D histogram:
    # we generate an array with the bin edges of both variables X,Y
    xx, yy = np.meshgrid(htrainBkgXY[1][:-1], htrainBkgXY[2][:-1])

    # flatten: Return a copy of the array collapsed into one dimension
    #plt.figure(2)
    #plt.hist2d(xx.flatten(), yy.flatten(), weights=htrainBkgXY[0].flatten(),
    #          bins=(htrainBkgXY[1], htrainBkgXY[2]))
    # note that in the bins parameter we do not discard the last value


    #######
    # once we have estimated a pdf as a histogram for the signal train data
    # and the bkg train data, we proceed to compute the statistic:

    def discrete_pdf(H, loc):
        # we have some 2d Histogram H which represents our trained pdf
        # we want to know which values will take some data loc
        # when fitted to this trained pdf
        entries = H[0]
        binx = H[1] #contains xedges
        biny = H[2] #contains yedges
        Nx = len(binx) # len will be nbins +1
        Ny = len(biny)
        out = 12 # because is outside the range of interest, but it can be anything, even 0, the result doesnt change.
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                # what we are doing here is to find inside what edges are the x and y considered in loc
                # once we have found this, we accept the counts there are in that bins.
                if loc[0] >= binx[i] and loc[0] <= binx[i + 1] and loc[1] >= biny[j] and loc[1] <= biny[j + 1]:
                    out = entries[i, j]
                    break
                    # elif (loc[0] > binx[Nx-1] and loc[1] > biny[Ny-1]) or (loc[0] < binx[0] and loc[1] < biny[0]):
                    # 	out = entries[Nx,Ny]
        if out < 1e-4:
            # print i,j, '\t', entries[i,j], '\t', loc
            out = 1e-5
        return out

    t_est = []
    # print(len((trainSignal[:,0]))) #10000
    for i in range(len(trainSignal[:,0])):
        t = discrete_pdf(htrainBkgXY, data[i,:]) / discrete_pdf(htrainSignalXY, data[i,:])
        t_est.append(t)

    return t_est
    # print(np.shape(t_est)) #(10000,)

    # Now we are going to plot the distribution of interest,
    # which is the joint pdf of Bkg divided by the joint pdf of Signal
    # we already have the counts and the bins, we only need to modify the weights,
    # which now will be the division of counts
    # Note that we can use the bins of Bkg or Signal, are equal

    #plt.figure(3)
    #plt.hist2d(xx.flatten(), yy.flatten(), weights=htrainBkgXY[0].flatten()/htrainSignalXY[0].flatten(),
    #           bins=(htrainBkgXY[1], htrainBkgXY[2]))

    # return(htrainBkgXY[0].flatten()/htrainSignalXY[0].flatten())
    # print(np.shape(htrainBkgXY[0].flatten()/htrainSignalXY[0].flatten()))
    # shape (400,) since 20*20

    # there are a lot of inf in this result. Maybe we need to filter them out with a mask.

    # plt.show()

################################################################################################################

def neural_network(train_sig, train_bkg, data):
    # train_sig, train_bkg are train samples to define the test statistic
    # data is the sample that we want to evaluate with the test statisic

    # Simple Neural Network
    # Refer to scikit-learn.com for more information

    # Remember
    # import sklearn.preprocessing as preproc

    # Scale the data (recommended for NN stability)

    # Standardization of datasets is a common requirement for many machine learning estimators
    # implemented in scikit-learn; they might behave badly if the individual features
    # do not more or less look like standard normally distributed data: Gaussian with zero
    # mean and unit variance.
    # The function scale provides a quick and easy way to perform this operation
    # on a single array-like dataset

    # define the scale
    # sklearn.preprocessing.StandardScaler is a class, ie it has attributes and methods.
    # Standardize features by removing the mean and scaling to unit variance

    # It also implements the Transformer API to compute the mean and standard deviation
    # on a training set so as to be able to later reapply the same transformation on the
    # testing set
    scaler = preproc.StandardScaler()
    # method fit(X[, y])	Compute the mean and std to be used for later scaling.
    scaler.fit(train_bkg)

    """
    # method transform(X[, y, copy]): Perform standardization by centering and scaling
    # ie applies the scale
    sc_train_sig = scaler.transform(train_sig)
    sc_train_bkg = scaler.transform(train_bkg)

    # Once we have the datasets standarized
    # Define and train the NN
    # here we will use import sklearn.neural_network as nn
    # reference search for neural networks models in sckikit-learn


    # put it all in one simple train data sample:
    # first we append signal and background in the same array
    sc_train_all = np.append(sc_train_sig, sc_train_bkg, axis=0)
    # print(sc_train_all)
    # print(np.shape(sc_train_all))
    # shape : (20000,2) 20000 rows 2 columns
    # since the given signal and brackground were (10000,2) dim datasets

    # size : Number of elements in the array, i.e. the product of the array’s dimensions.
    # train_sig.size = 10000*2 = 20000
    # train_sig[0].size = 2
    # so implies = 10000 zeros.
    type_all = np.append(np.zeros(int(train_sig.size / train_sig[0].size)),
                         np.ones(int(train_bkg.size / train_bkg[0].size)))
    # print(type_all)

    # create the NN and train it with the training data
    # We will use the class sklearn.neural_network.MLPRegressor
    # Multi-layer Perceptron regressor.
    # This model optimizes the squared-loss using LBFGS or stochastic gradient descent.
    # the input layer is the train data, which corresponds to an array (20000,2)
    # Each neuron in the hidden layer transforms the values from the previous layer
    # The output layer receives the values from the last hidden layer and transforms them
    # into output values.
    # hidden_layer_sizes=(13333,6667) this implies that we are considering two hidden layers
    # the first with 13333 neurons and the second with 6667 neurons. The last layer transforms to
    # the output layer, which has 1 neuron.
    # ref: https://stats.stackexchange.com/questions/181/
    # how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # alpha: L2 penalty (regularization term) parameter.
    # which helps in avoiding overfitting by penalizing weights with large magnitudes.
    # si if there is overfitting we must modify this
    # random_state different from none if we want to fix a seed of random numbers

    clf = nn.MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(1000, 200),
                          random_state=1)

    # once we have the nn defined, we apply fit
    # fit(X, y): Fit (train) the model to data matrix X and target(s) y.
    # returns a trained MLP model.
    # the fit must receive a list of the data train, and a list indicating if
    # they are signal (0) or background (1).
    # H0 es signal H1 es background
    clf.fit(sc_train_all, type_all)
    # print(clf.fit(sc_train_all,type_all))
    # for the moment what we have done here is to learn a function with its weights.
    # ie we have done the training, and we have defined the the test statistic
    # now it remains to enter some data (input layer), and return the predicted output
    # using this trained Model


    # since performing the model is time demandanding. We do not want to repeat the process again.
    # We can save the model using joblib.dump & joblib.load
    # >>> from sklearn.externals import joblib
    # >>> joblib.dump(clf, 'filename.pkl')
    # clf = joblib.load('filename.pkl')
    # clf.predict(X[0:1])

    joblib.dump(clf, "MLPR_model.pkl")
    """
    # once we have the file created, we do not compile the model again

    clf = joblib.load("MLPR_model.pkl")

    ###################################
    # evaluate the NN test statistic with some data
    # the data variable is an argument of the function
    # first we scale the data as we did before
    sc_data = scaler.transform(data)

    # Predict using the multi-layer perceptron model.
    # Returns the predicted values, ie t(x), which is a number
    clf_data = clf.predict(sc_data)


    """
    # now we return the final results
    # on the left the data evaluated (input layer)
    # on the right the result t(x), which is a number
    print("{:^20}  {:>15}".format("   data", "NN classifier"))
    print("{:^20}  {:>15}".format("  ********", "*************"))
    for dataval, clfval in zip(data, clf_data):
        print("{:>20}  {:^15.2f}".format(str(dataval), clfval))


    # ara mateix la funcio col·lapsa i el pc es penja
    # 1st hypothesis: el input data es una array 10000,2
    # en canvi la primera hidden te 13333 neurons
    # de manera que hauria de considerar una initial data
    # amb signal + background

    """

    # return the 1D array with the values of T(x)
    return clf_data


####################################################################################
##########################      PART B    ########################################
#################################################################################
# we plot de distributions of the test statistics for the given files
# the goal is to compare the distributions obtained with the
# train files and with the test files.
# They should give very similar results.
# we must do this for every test statistic created before

##################################################################################
# radial distance

"""
plt.figure(1)
plt.hist(radial_distance(trainSignalX, trainSignalY), bins = 20, histtype="step", color ="black", label= "train")
plt.hist(radial_distance(testSignalX, testSignalY), bins= 20, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(0,1.44)
plt.xticks(np.arange(0,1.5,0.1))
plt.ylim(0,1100)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.figure(2)
plt.hist(radial_distance(trainBkgX, trainBkgY), bins = 20, histtype="step", color ="black", label = "train")
plt.hist(radial_distance(testBkgX, testBkgY), bins=20, histtype="step", color ="black", ls= "dashed", label = "test")
plt.xlim(0,1.44)
plt.xticks(np.arange(0,1.5,0.1))
plt.ylim(0,1100)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

##################################################################################
# Fisher_discriminant

"""
plt.figure(1)
plt.hist(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, trainSignalX, trainSignalY),
         bins = 20, histtype="step", color ="black", label= "train")
plt.hist(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, testSignalX, testSignalY),
         bins= 20, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(-0.7,2.5)
# plt.xticks(np.arange(0,1.6,0.1))
plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.figure(2)
plt.hist(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, trainBkgX, trainBkgY),
         bins = 20, histtype="step", color ="black", label = "train")
plt.hist(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, testBkgX, testBkgY),
         bins=20, histtype="step", color ="black", ls= "dashed", label = "test")
plt.xlim(-0.7,2.5)
#plt.xticks(np.arange(0,1.6,0.1))
plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

##################################################################################
# exact_likelihood_ratio(x,y)

"""
plt.figure(1)
plt.hist(exact_likelihood_ratio(trainSignalX, trainSignalY, trainSignalX, trainSignalY), range= [-0,2],
         bins = 80, histtype="step", color ="black", label= "train")
plt.hist(exact_likelihood_ratio(testSignalX, testSignalY, testSignalX, testSignalY), range= [-0,2],
         bins= 80, histtype="step", color ="black", ls= "dashed", label="test")
#plt.xticks(np.arange(0,60,5))
plt.xlim(0,1.2)
#plt.yticks(np.arange(0,12000,1000))
#plt.ylim(0,10000)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.figure(2)
plt.hist(exact_likelihood_ratio(trainBkgX, trainBkgY, trainBkgX, trainBkgY), range= [-0,2],
         bins = 80, histtype="step", color ="black", label = "train")
plt.hist(exact_likelihood_ratio(testBkgX, testBkgY, testBkgX, testBkgY), range= [-0,2],
         bins= 80, histtype="step", color ="black", ls= "dashed", label = "test")
#plt.xticks(np.arange(0,60,5))
plt.xlim(0,1.2)
#plt.yticks(np.arange(0,12000,1000))
#plt.ylim(0,10000)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

"""
plt.figure(3)
# we must apply range, because we have elements which are inf
plt.hist(exact_likelihood_ratio(trainSignalX, trainSignalY, trainBkgX, trainBkgY), range= [-0,10],
         bins = 100, histtype="step", color ="black", label= "train")
plt.hist(exact_likelihood_ratio(testSignalX, testSignalY, testBkgX, testBkgY), range= [-0,10],
         bins= 100, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(0,4)
# plt.xticks(np.arange(0,1.6,0.1))
#plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

##################################################################################
# estimated_likelihood_ratio(data)

"""
plt.figure(1)
plt.hist(estimated_likelihood_ratio(trainSignal), range= [-0,2],
         bins = 10, histtype="step", color ="black", label= "train")
plt.hist(estimated_likelihood_ratio(testSignal), range= [-0,2],
         bins= 10, histtype="step", color ="black", ls= "dashed", label="test")
plt.xticks(np.arange(0,4,0.2))
plt.xlim(0,2)
#plt.yticks(np.arange(0,12000,1000))
#plt.ylim(0,10000)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.figure(2)
plt.hist(estimated_likelihood_ratio(trainBkg), range= [-0,2],
         bins = 10, histtype="step", color ="black", label = "train")
plt.hist(estimated_likelihood_ratio(testBkg), range= [-0,2],
         bins= 10, histtype="step", color ="black", ls= "dashed", label = "test")
plt.xticks(np.arange(0,4,0.2))
plt.xlim(0,2)
#plt.yticks(np.arange(0,12000,1000))
#plt.ylim(0,10000)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

"""
# the estimated LR makes use of both samples, signal and bkg, since it computes
# the test statistic dividing both histograms. So, in this case we cannot
# study both cases separately

z_train = estimated_likelihood_ratio(trainSignalX ,trainSignalY, trainBkgX, trainBkgY)
z_train = np.ma.masked_invalid(z_train) # we apply a mask to erradicate the inf values
# however the histogram is the same, applying or not the mask

z_test = estimated_likelihood_ratio(testSignalX ,testSignalY, testBkgX, testBkgY)
z_test = np.ma.masked_invalid(z_test)


plt.figure(4)
# we must apply range, because we have elements which are inf
plt.hist(z_train, range= [-0,50], bins = 300, histtype="step", color ="black", label= "train")
plt.hist(z_test, range= [-0,50], bins= 300, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(0,4)
# plt.xticks(np.arange(0,1.6,0.1))
#plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

##################################################################################
# neural_network(trainSignal , trainBkg , data)

"""
plt.figure(1)
plt.hist(neural_network(trainSignal , trainBkg , trainSignal),
         bins = 20, histtype="step", color ="black", label= "train")
plt.hist(neural_network(trainSignal , trainBkg , testSignal),
         bins= 20, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(0,1.2)
# plt.xticks(np.arange(0,1.6,0.1))
# plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')


plt.figure(2)
plt.hist(neural_network(trainSignal , trainBkg , trainBkg),
         bins = 20, histtype="step", color ="black", label= "train")
plt.hist(neural_network(trainSignal , trainBkg , testBkg),
         bins= 20, histtype="step", color ="black", ls= "dashed", label="test")
plt.xlim(0,1.2)
# plt.xticks(np.arange(0,1.6,0.1))
#plt.ylim(0,1200)
plt.xlabel("T")
plt.ylabel("N(T)")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()
"""

###################################################
### Kolmogorov test
# we use scipy.stats.ks_2samp(data1, data2)

# radial distance
"""
kol_radial_signal = stats.ks_2samp(radial_distance(trainSignalX, trainSignalY), radial_distance(testSignalX, testSignalY))
kol_radial_bkg = stats.ks_2samp(radial_distance(trainBkgX, trainBkgY),radial_distance(testBkgX, testBkgY))
print(kol_radial_signal)
print(kol_radial_bkg)
"""

# Fisher_discriminant
"""
kol_fisher_signal = stats.ks_2samp(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, trainSignalX, trainSignalY),
                                   Fisher_discriminant(trainSignalX, trainSignalY, trainBkgX, trainBkgY, testSignalX, testSignalY))
kol_fisher_bkg = stats.ks_2samp(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, trainBkgX, trainBkgY),
                                   Fisher_discriminant(trainSignalX, trainSignalY, trainBkgX, trainBkgY, testBkgX, testBkgY))
print(kol_fisher_signal)
print(kol_fisher_bkg)
"""

# exact_likelihood_ratio(x,y)
"""
kol_exact_lik_signal = stats.ks_2samp(exact_likelihood_ratio(trainSignalX, trainSignalY, trainSignalX, trainSignalY),
                                      exact_likelihood_ratio(testSignalX, testSignalY, testSignalX, testSignalY))
kol_exact_lik_bkg = stats.ks_2samp(exact_likelihood_ratio(trainBkgX, trainBkgY, trainBkgX, trainBkgY),
                                   exact_likelihood_ratio(testBkgX, testBkgY, testBkgX, testBkgY))
print(kol_exact_lik_signal)
print(kol_exact_lik_bkg)
"""

# estimated_likelihood_ratio(trainSignalX ,trainSignalY, trainBkgX, trainBkgY)

"""
kol_est_likelihood_signal = stats.ks_2samp(estimated_likelihood_ratio(trainSignal),
                                    estimated_likelihood_ratio(testSignal))
kol_est_likelihood_bkg = stats.ks_2samp(estimated_likelihood_ratio(trainBkg),
                                    estimated_likelihood_ratio(testBkg))
print(kol_est_likelihood_signal)
print(kol_est_likelihood_bkg)
"""

# neural_network(trainSignal , trainBkg , data)

#if we want to compute again the file model, we call the nn only once.
# once we have the file we compute the kolmogorov test without computing the model again every time.
#neural_network(trainSignal , trainBkg , trainSignal)

"""
kol_nn_signal = stats.ks_2samp(neural_network(trainSignal, trainBkg , trainSignal),
                                      neural_network(trainSignal, trainBkg, testSignal))
kol_nn_bkg = stats.ks_2samp(neural_network(trainSignal, trainBkg , trainBkg),
                                   neural_network(trainSignal, trainBkg, testBkg))
print(kol_nn_signal)
print(kol_nn_bkg)
"""

###################################################################################
#########################      PART C & D & E & F     ##############################
##################################################################################
# now that we have already verified that the train data and the test data gave
# similar distributions, we could use one of the other.
# However, we will use the test data that we know it does not have fluctuations.
# for each test we compute alpha and beta for different values fo tcut
# and we plot the required functions
# In part D, we also compute the T_cut as the value that maximizes
# the signal-to-noise ratio
# In part E we plot the signal-to-noise ratio vs (1-alpha) for all the test statistics
# in the same figure.
# In part F we draw the boundaries of the critical region defined by T_cut in the
# (x,y) plane

# what is the pdf?
# we don't know it, we use an estimator, ie an histogram.
# so for each statistic the pdf are the histogram of the values of T obtained.

# how do we compute the integrals alpha and beta?
# we use the following functions:

def alpha(T_cut, pdf_s):
    # the integral is just the total area underneath the curve.
    # an integral of a histogram is computes as:
    # sum([bin_width[i] * bin_height[i] for i in bin_indexes_to_integrate])
    counts = pdf_s[0] #len Nbins
    bins = pdf_s[1] #len Nbins+1
    # we get the width of each bin (all widths are equal)
    bins_width = bins[1] - bins[0]
    # first we separate the cases if T_cut < any of our values
    # or > any of our values
    sum_counts = 0
    if T_cut < bins[0]:
        return 1.0
    elif T_cut > bins[len(bins)-1]:
        return 0.0
    else:
        # if we have this case we identify the bins we want to integrate
        # then we sum all the counts of them
        for i in range(0, len(bins)):
            if T_cut >= bins[i] and T_cut < bins[i+1]:
                for j in range(i,len(counts)):
                    sum_counts += counts[j]
        return sum_counts*bins_width

def beta(T_cut, pdf_b):
    counts = pdf_b[0] #len Nbins
    bins = pdf_b[1] #len Nbins+1
    bins_width = bins[1] - bins[0]
    sum_counts = 0
    if T_cut < bins[0]:
        return 0.0
    elif T_cut > bins[len(bins)-1]:
        return 1.0
    else:
        for i in range(0,len(bins)):
            if T_cut > bins[i] and T_cut <= bins[i+1]: # note the condition is changed
                for j in range(0,i):
                    sum_counts += counts[j]
        return sum_counts*bins_width

############################################################
# radial distance

"""
pdf_s = np.histogram(radial_distance(testSignalX, testSignalY), normed=True, bins=1000)
pdf_b = np.histogram(radial_distance(testBkgX, testBkgY), normed=True, bins=1000)
# note that we use normalized histograms in order to obtain values of
# alpha and beta between 0 and 1

# we propose a range of values of T_cut. This range must be in accordance
# to the values of T obtained, because the values out of the range
# give directly 0 or 1 (not relevant)
# this range must be in accordance to the bins of the histogram
# if we use 1000 bins, we provide 1000 values of T_cut
# more values imply more precision
T_cut = np.linspace(-0.5,2.0,1000)

# we compute alpha integral for each T_cut
# and we save each value
alpha_list =[]
for T_cut_i in T_cut:
    alpha_list.append(alpha(T_cut_i, pdf_s))

# we compute beta integral for each T_cut
# and we save each value
beta_list = []
for T_cut_i in T_cut:
    beta_list.append(beta(T_cut_i, pdf_b))

# now we plot the functions of interest
plt.figure()
plt.plot(T_cut, 1-np.array(alpha_list), color = "black", label = r"$1-\alpha$")
plt.plot(T_cut, np.array(beta_list), color = "red", label = r"$\beta$")

# we compute STNR before plotting, since we need to avoid an error:
# (dividing by beta = 0):
STNR = []
for i in range(0,len(T_cut)):
    if beta_list[i] != 0:
        STNR.append((1-alpha_list[i])/np.sqrt(beta_list[i]))
    else:
        STNR.append(0)

plt.plot(T_cut, STNR, color = "orange", label = r"$(1-\alpha)/\sqrt{\beta}$")
plt.xlabel(r"$T_{cut}$")
plt.xticks(np.arange(0,2,0.1))
plt.yticks(np.arange(0,2.0,0.1))
plt.xlim(-0,1.4)
plt.ylim(0,1.2)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')

plt.show()

# Part D:
# we find the max value of the signal to noise ratio (1-alpha)/sqrt(beta)
# and its corresponding T_cut value
print("Signal-to-noise max: ", np.amax(STNR))
print(r"T_{cut}: ", T_cut[np.argmax(STNR)])
T_cut_max = T_cut[np.argmax(STNR)]

# Part E:
# we plot the signal-to-noise ratio vs (1-alpha)
plt.figure("STNR vs 1-alpha")
plt.plot(1-np.array(alpha_list), STNR, color = "black", label = "Radial distance")
plt.xticks(np.arange(0,2,0.1))
plt.yticks(np.arange(0,2.0,0.1))
plt.xlim(0,1)
plt.ylim(0,1.5)
plt.xlabel(r"$1-\alpha$")
plt.ylabel(r"$(1-\alpha)/\sqrt{\beta}$")
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
# since we want to compute all the results in the same figure,
# we must plot all the test statistics together
# we don't call plt.show()
# plt.show()

# Part F
# we draw the boundaries of the critical region defined by T_cut in the (x,y) plane
# first, we generate values of x,y between the defined range
x_plane, y_plane = np.linspace(-1., 1., 1000), np.linspace(-1., 1., 1000)

# we use meshgrid to create a 2-D array from two 1-D vectors
x_grid,y_grid = np.meshgrid(x_plane, y_plane)
# for each pair (x,y) of the array we compute the value of the t statistic.
t_grid = radial_distance(x_grid, y_grid)
# this will be a 2D array
# now we define a mask to select the values above T_cut and below T_cut
mask = t_grid > T_cut_max
# the mask is a 2D array which returns two possible outputs
# true for above T_cut (reject-red) and false for below t_cut (accept-blue)
plt.figure()
# the best way to show the boundaries is with a filled contour plot.
cs = plt.contourf(x_grid, y_grid, mask, levels=[-0.5,0.5,1.5], colors=("blue", "red"))
# we define that we want the levels to be coulored as red blue in such order
# rejected-red and accepted-blue
# we define the level curves in increasing order and in accordante to our values
# we have the values false = 0 and true = 1.

plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")

# now we generate the legend for our contourf plot:
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]
plt.legend(proxy, ["Acceptance region", "Critical region"])
plt.show()

#This method is not optimal:
# requires a lot of time and the boundaries are not clear
# for each pair (x,y) we compute the value of the t statistic
# and we compare with T_cut, if the value is above T_cut we
# plot them in one colour, otherwise in other color
#plt.figure()
#for x_i in x_plane:
#    for y_i in y_plane:
#        if radial_distance(x_i, y_i) > T_cut_max:
#            plt.scatter(x_i, y_i, c="red") #red for reject
#        else:
#            plt.scatter(x_i, y_i, c="blue")  #blue for accept
plt.show()

"""

############################################################
# Fisher_discriminant

"""
pdf_s = np.histogram(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, testSignalX, testSignalY),
                     normed=True, bins=1000)
pdf_b = np.histogram(Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, testBkgX, testBkgY),
                     normed=True, bins=1000)

T_cut = np.linspace(-1,2.5,1000)

alpha_list =[]
for T_cut_i in T_cut:
    alpha_list.append(alpha(T_cut_i, pdf_s))

beta_list = []
for T_cut_i in T_cut:
    beta_list.append(beta(T_cut_i, pdf_b))

plt.figure()
plt.plot(T_cut, 1-np.array(alpha_list), color = "black", label = r"$1-\alpha$")
plt.plot(T_cut, np.array(beta_list), color = "red", label = r"$\beta$")

STNR = []
for i in range(0,len(T_cut)):
    if beta_list[i] != 0:
        STNR.append((1-alpha_list[i])/np.sqrt(beta_list[i]))
    else:
        STNR.append(0)

plt.plot(T_cut, STNR, color = "orange", label = r"$(1-\alpha)/\sqrt{\beta}$")
plt.xlabel(r"$T_{cut}$")
plt.xticks(np.arange(-0.6, 2.4, 0.2))
plt.xlim(-0.6,2.2)
plt.yticks(np.arange(0,2.0,0.1))
plt.ylim(0,1.3)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.show()

# Part D:
# we find the max value of the signal to noise ratio (1-alpha)/sqrt(beta)
# and its corresponding T_cut value
print("Signal-to-noise max: ", np.amax(STNR))
print(r"T_{cut}: ", T_cut[np.argmax(STNR)])
T_cut_max = T_cut[np.argmax(STNR)]

# Part E:
plt.figure("STNR vs 1-alpha")
plt.plot(1-np.array(alpha_list), STNR, color = "red", label = "Fisher discriminant")

# Part F
x_plane, y_plane = np.linspace(-1., 1., 1000), np.linspace(-1., 1., 1000)
x_grid,y_grid = np.meshgrid(x_plane, y_plane)
#print(np.array_equal(x_grid,y_grid))
# they are not equal
t_grid = Fisher_discriminant(trainSignalX ,trainSignalY, trainBkgX, trainBkgY, x_grid, y_grid)
#print(np.shape(t_grid)) (1000, 1000)
mask = t_grid > T_cut_max
plt.figure()
cs = plt.contourf(x_grid, y_grid, mask, levels=[-0.5,0.5,1.5], colors=("blue", "red"))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]
plt.legend(proxy, ["Acceptance region", "Critical region"], loc="upper left")
plt.show()
"""

############################################################
# exact_likelihood_ratio(x,y)

"""
pdf_s = np.histogram(exact_likelihood_ratio(testSignalX, testSignalY, testSignalX, testSignalY),
                     range = [0,100], normed=True, bins=10000)
pdf_b = np.histogram(exact_likelihood_ratio(testBkgX, testBkgY, testBkgX, testBkgY),
                     range = [0,100], normed=True, bins=10000)

T_cut = np.linspace(-0,100.00001,10000)

alpha_list =[]
for T_cut_i in T_cut:
    alpha_list.append(alpha(T_cut_i, pdf_s))

beta_list = []
for T_cut_i in T_cut:
    beta_list.append(beta(T_cut_i, pdf_b))

plt.figure()
plt.plot(T_cut, 1-np.array(alpha_list), color = "black", label = r"$1-\alpha$")
plt.plot(T_cut, np.array(beta_list), color = "red", label = r"$\beta$")


STNR = []
for i in range(0,len(T_cut)):
    if beta_list[i] != 0:
        STNR.append((1-alpha_list[i])/np.sqrt(beta_list[i]))
    else:
        STNR.append(0)

plt.plot(T_cut, STNR, color = "orange", label = r"$(1-\alpha)/\sqrt{\beta}$")
plt.xlabel(r"$T_{cut}$")
plt.xticks(np.arange(-0.6, 2.4, 0.2))
plt.xlim(-0,2.2)
plt.yticks(np.arange(0,2.0,0.1))
plt.ylim(0,1.3)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.show()

# Part D:
# we find the max value of the signal to noise ratio (1-alpha)/sqrt(beta)
# and its corresponding T_cut value
print("Signal-to-noise max: ", np.amax(STNR))
print(r"T_{cut}: ", T_cut[np.argmax(STNR)])
T_cut_max = T_cut[np.argmax(STNR)]
# Signal-to-noise max:  1.28411059882
# T_{cut}:  0.850085093509
#print(beta(T_cut_max, pdf_b))
# beta(T_cut_max, pdf_b)) = 0.415502534306

# Part E:
plt.figure("STNR vs 1-alpha")
plt.plot(1-np.array(alpha_list), STNR, color = "blue", label = "Exact likelihood ratio")

# Part F
x_plane, y_plane = np.linspace(-1., 1., 1000), np.linspace(-1., 1., 1000)
x_grid,y_grid = np.meshgrid(x_plane, y_plane)
t_grid = exact_likelihood_ratio(x_grid, y_grid, x_grid, y_grid)
mask = t_grid > T_cut_max
plt.figure()
cs = plt.contourf(x_grid, y_grid, mask, levels=[-0.5,0.5,1.5], colors=("blue", "red"))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]
plt.legend(proxy, ["Acceptance region", "Critical region"], loc="upper left")
plt.show()
"""

############################################################
# estimated_likelihood_ratio

"""
pdf_s = np.histogram(estimated_likelihood_ratio(testSignal), range = [0,100], normed=True, bins=10000)
pdf_b = np.histogram(estimated_likelihood_ratio(testBkg), range = [0,100], normed=True, bins=10000)

T_cut = np.linspace(-0,100.00001,10000)

alpha_list =[]
for T_cut_i in T_cut:
    alpha_list.append(alpha(T_cut_i, pdf_s))

beta_list = []
for T_cut_i in T_cut:
    beta_list.append(beta(T_cut_i, pdf_b))

plt.figure()
plt.plot(T_cut, 1-np.array(alpha_list), color = "black", label = r"$1-\alpha$")
plt.plot(T_cut, np.array(beta_list), color = "red", label = r"$\beta$")


STNR = []
for i in range(0,len(T_cut)):
    if beta_list[i] != 0:
        STNR.append((1-alpha_list[i])/np.sqrt(beta_list[i]))
    else:
        STNR.append(0)

plt.plot(T_cut, STNR, color = "orange", label = r"$(1-\alpha)/\sqrt{\beta}$")
plt.xlabel(r"$T_{cut}$")
plt.xticks(np.arange(-0.6, 2.4, 0.2))
plt.xlim(-0,2.2)
plt.yticks(np.arange(0,2.0,0.1))
plt.ylim(0,1.3)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.show()

# Part D:
# we find the max value of the signal to noise ratio (1-alpha)/sqrt(beta)
# and its corresponding T_cut value
print("Signal-to-noise max: ", np.amax(STNR))
print(r"T_{cut}: ", T_cut[np.argmax(STNR)])
T_cut_max = T_cut[np.argmax(STNR)]
#Signal-to-noise max:  1.2497021548
#T_{cut}:  0.94009410341

# Part E:
plt.figure("STNR vs 1-alpha")
plt.plot(1-np.array(alpha_list), STNR, color = "green", label = "Estimated likelihood ratio")
"""

"""
# Part F
T_cut_max = 0.94009410341
# there is a problem when computing Part F for Estimated likelihood and NN
# opposite to the other test, in these two cases we defined the function
# taking as an argument the whole array data of (x,y)
# instead of single pairs (x,y).
# We need the value of T for each x,y to plot the contourf
# we must modify the function:

def estimated_likelihood_ratio_modified(x,y):
    # the principal difference is that now data, will be a list of two values
    # [x,y], but only two.
    # we compute one t for each pair [x,y]
    # in the function we had before, we introduced the whole data of (x,y)
    # and we computed t for all the pairs so we obtained a list of t.

    nbinsx, nbinsy = 20, 20 #number of bins
    minx, maxx, miny, maxy = -1., 1., -1., 1. #domain x,y

    htrainSignalXY = np.histogram2d(trainSignalX, trainSignalY, [nbinsx, nbinsy],
                                    [[minx, maxx], [minx, maxx]])

    htrainBkgXY = np.histogram2d(trainBkgX, trainBkgY, [nbinsx, nbinsy],
                                    [[minx, maxx], [minx, maxx]])

    def discrete_pdf(H, x, y):
        # we have some 2d Histogram H which represents our trained pdf
        # we want to know which values will take some data loc
        # when fitted to this trained pdf
        entries = H[0]
        binx = H[1] #contains xedges
        biny = H[2] #contains yedges
        Nx = len(binx) # len will be nbins +1
        Ny = len(biny)
        out = 12 # because is outside the range of interest, but it can be anything, even 0, the result doesnt change.
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                # what we are doing here is to find inside what edges are the x and y considered in loc
                # once we have found this, we accept the counts there are in that bins.
                if x >= binx[i] and x <= binx[i + 1] and y >= biny[j] and y <= biny[j + 1]:
                    out = entries[i, j]
                    break
                    # elif (loc[0] > binx[Nx-1] and loc[1] > biny[Ny-1]) or (loc[0] < binx[0] and loc[1] < biny[0]):
                    # 	out = entries[Nx,Ny]
        if out < 1e-4:
            # print i,j, '\t', entries[i,j], '\t', loc
            out = 1e-5
        return out

    t_est = discrete_pdf(htrainBkgXY, x, y) / discrete_pdf(htrainSignalXY, x, y)
    return t_est
    # now the function only returns one value of t for each x,y
    # there is still a problem: estimated_likelihood_ratio_modified(x_grid, y_grid)
    # is a function, not an array.
    # then interprets x_grid, y_grid as the whole array
    # we need to deal work with x[i] and y[i]

x_plane, y_plane = np.linspace(-1., 1., 100), np.linspace(-1., 1., 100)
# in this case we cannot use to many points, because the computation
# is slower
x_grid,y_grid = np.meshgrid(x_plane, y_plane)
# x_grid and y_grid are 2d arrays (100,100)

t_grid = np.zeros((100,100)) # we create an array of the shape we want
# if we use np.array([]), we have problems
for j in range(0,len(x_plane)):
    x_j = x_plane[j] #the x values start from -1 and goes to 1
    for i in range(0,len(y_plane)):
        y_i = y_plane[len(y_plane)-1-i] # the y values start from 1 and goes to -1
        print((i,j), x_j,y_i)
        t_grid[i,j] = estimated_likelihood_ratio_modified(x_j, y_i)

mask = t_grid > T_cut_max
# i don't know why, but the contourf plot shows the mirror image of the mask
# it didn't happen in the other cases, but we are doing the same and
# the mask is correct.
# so we need to reverse the order of the elements along x
mask = np.flip(mask, axis=1)
plt.figure()
cs = plt.contourf(x_grid, y_grid, mask, levels=[-0.5,0.5,1.5], colors=("blue", "red"))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]
plt.legend(proxy, ["Acceptance region", "Critical region"], loc="upper left")
plt.show()
"""

############################################################
# neural_network(trainSignal , trainBkg , data)

"""
pdf_s = np.histogram(neural_network(trainSignal, trainBkg, testSignal),
                     normed=True, bins=1000)
pdf_b = np.histogram(neural_network(trainSignal, trainBkg, testBkg),
                     normed=True, bins=1000)

T_cut = np.linspace(0,1.2,1000)

alpha_list =[]
for T_cut_i in T_cut:
    alpha_list.append(alpha(T_cut_i, pdf_s))

beta_list = []
for T_cut_i in T_cut:
    beta_list.append(beta(T_cut_i, pdf_b))

plt.figure()
plt.plot(T_cut, 1-np.array(alpha_list), color = "black", label = r"$1-\alpha$")
plt.plot(T_cut, np.array(beta_list), color = "red", label = r"$\beta$")

STNR = []
for i in range(0,len(T_cut)):
    if beta_list[i] != 0:
        STNR.append((1-alpha_list[i])/np.sqrt(beta_list[i]))
    else:
        STNR.append(0)

plt.plot(T_cut, STNR, color = "orange", label = r"$(1-\alpha)/\sqrt{\beta}$")
plt.xlabel(r"$T_{cut}$")
plt.xticks(np.arange(-0.6, 2.4, 0.1))
plt.xlim(-0,1.1)
plt.yticks(np.arange(0,2.0,0.1))
plt.ylim(0,1.5)
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
#plt.show()

# Part D:
# we find the max value of the signal to noise ratio (1-alpha)/sqrt(beta)
# and its corresponding T_cut value
print("Signal-to-noise max: ", np.amax(STNR))
print(r"T_{cut}: ", T_cut[np.argmax(STNR)])
T_cut_max = T_cut[np.argmax(STNR)]

# Part E:
plt.figure("STNR vs 1-alpha")
plt.plot(1-np.array(alpha_list), STNR, color = "orange", label = "Neural network")
# once we have all the plots labeled, we call the legend and plt.show()
plt.legend(loc="best")
leg = plt.legend()
leg.get_frame().set_edgecolor('black')
plt.show()
"""

"""
# Part F
# Again, we need the value of T for each x,y to plot the contourf
# in this case, we don't have the need to modify the function,
# but we must proceed as in the estimated case

x_plane, y_plane = np.linspace(-1., 1., 1000), np.linspace(-1., 1., 1000)
x_grid,y_grid = np.meshgrid(x_plane, y_plane)

t_grid = np.zeros((1000,1000)) # we create an array of the shape we want
# if we use np.array([]), we have problems
for j in range(0,len(x_plane)):
    x_j = x_plane[j] #the x values start from -1 and goes to 1
    for i in range(0,len(y_plane)):
        y_i = y_plane[len(y_plane)-1-i] # the y values start from 1 and goes to -1
        print((i,j), x_j,y_i)
        t_grid[i,j] = neural_network(trainSignal, trainBkg, np.array([x_j, y_i]).reshape(1,-1))
        # we must reshape. Otherwise there is a warning:
        # C:\Program Files\WinPython-64bit-3.6.2.0Qt5\python-3.6.2.amd64\lib\site-packages\sklearn\
        # preprocessing\data.py:649: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17
        # and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data
        # has a single feature or X.reshape(1, -1) if it contains a single sample.

mask = t_grid > T_cut_max
# again, i don't know why, but the contourf plot shows the mirror image of the mask
# it didn't happen in the other cases, but we are doing the same and
# the mask is correct.
# so we need to reverse the order of the elements along x
mask = np.flip(mask, axis=1)
plt.figure()
cs = plt.contourf(x_grid, y_grid, mask, levels=[-0.5,0.5,1.5], colors=("blue", "red"))
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
    for pc in cs.collections]
plt.legend(proxy, ["Acceptance region", "Critical region"], loc="upper left")
plt.show()
"""

############################################################
# contourf plot for the pdf of signal
'''
x_plane, y_plane = np.linspace(-1., 1., 1000), np.linspace(-1., 1., 1000)
x_grid,y_grid = np.meshgrid(x_plane, y_plane)


def pdfSignalXY(x, y, N, a, b, c):
    return 1. / N * np.exp(-(a * x ** 2 + b * y ** 2 + 2 * c * x * y))
a, b, c = 6., 6., -5.
NpdfSignalXY, Nerr = pyint.dblquad(pdfSignalXY, -1, 1,
                                   lambda x: -1, lambda x: 1, args=(1, a, b, c))

pdf_grid = pdfSignalXY(x_grid, y_grid, NpdfSignalXY, a, b, c)

#mask = pdf_grid > T_cut_max
plt.figure()
#cs = plt.contourf(x_grid, y_grid, pdf_grid)
plt.contourf(x_grid, y_grid, pdf_grid, 8, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(x_grid, y_grid, pdf_grid, 8, colors='black', linewidth=.5)
plt.clabel(C, inline=1, fontsize=10)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("x")
plt.ylabel("y")
#proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0])
#    for pc in cs.collections]
#plt.legend(proxy, ["Acceptance region", "Critical region"], loc="upper left")
plt.show()
'''



###################################################################################
#############################      PART G    ####################################
##################################################################################
# we are going to use the optimal test from the ones computed before to select
# signal subsamples out of the data_on.txt and data_off.txt

"""
# first, we read the input files:
dataOnFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_On.txt"
dataOn = np.loadtxt(dataOnFileName, skiprows=2)

# Extract x,y columns
dataOnX = dataOn[:,0]
dataOnY = dataOn[:,1]

dataOffFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_Off.txt"
dataOff = np.loadtxt(dataOffFileName, skiprows=2)

# Extract x,y columns
dataOffX = dataOff[:,0]
dataOffY = dataOff[:,1]

# as an optimal test we are going to use the exact likelihood ratio
# we use the T_cut_max computed before
T_cut_max = 0.850085093509
# T_cut_max = 0.93 is the one for which I get better results
# 2.48895890453
# in theory i should get 2.57 but i don't want to change all again
# the problem must be in the T_cut, but i think it is correct


############# DataOn ####################
t_values = exact_likelihood_ratio(dataOnX, dataOnY, dataOnX, dataOnY)
# we get a list of t values for each pair (x,y)
print(np.shape(t_values))

# we want to select those with t < T_cut as signal
# and those with t > T_cut as bkg
# we use masks
t_signal_on = t_values[t_values < T_cut_max]
t_bkg_on = t_values[t_values > T_cut_max]
print("Number of events in enriched-signal subsample in On:", len(t_signal_on))
print("Number of events in bkg subsample in On:", len(t_bkg_on))

############# DataOff ####################
t_values = exact_likelihood_ratio(dataOffX, dataOffY, dataOffX, dataOffY)
# we get a list of t values for each pair (x,y)
print(np.shape(t_values))

# we want to select those with t < T_cut as signal
# and those with t > T_cut as bkg
# we use masks
t_signal_off = t_values[t_values < T_cut_max]
t_bkg_off = t_values[t_values > T_cut_max]
print("Number in enriched-signal subsample in Off:", len(t_signal_off))
print("Number of events in bkg subsample in Off:", len(t_bkg_off))

# we verify that in the dataOff we obtain 3 times more bkg than in dataOn:
print("signal relation", len(t_signal_off)/len(t_signal_on))
print("bkg relation", len(t_bkg_off)/len(t_bkg_on))

############################################
# now we are going to compute the number of signal and bkg events in the subsample
# we need to compute beta (type ii error) with the T_cut_max.
# the question is, which distribution to use
# the testbkg we used before, ie we go back and compute beta for this cut.
# or the bakcground control sample (data_off), which in principle
# only contains bkg events, so its values x,y should be distributed as
# the hypothesis H1, which is what we want.
# Let's try both cases:

# using the testbkg sample:
beta_testbkg = 0.415502534306

# using the background control sample (dataOff):
pdf_b_dataOff = np.histogram(exact_likelihood_ratio(dataOffX, dataOffY, dataOffX, dataOffY),
                     range = [0,100], normed=True, bins=10000)

beta_dataOff = beta(T_cut_max, pdf_b_dataOff)
# 0.412417763158

# in both cases we obtain very similar results as expected
# since both samples are distributed as H1.
# we are going to use beta_dataOff

bkg_subsample_On = len(t_signal_on)*beta_dataOff
signal_subsample_On = len(t_signal_on) - bkg_subsample_On

print("bkg_subsample_On", bkg_subsample_On)
print("signal_subsample_On", signal_subsample_On)

bkg_subsample_Off = len(t_signal_off)*beta_dataOff
signal_subsample_Off = len(t_signal_off) - bkg_subsample_Off

print("bkg_subsample_Off", bkg_subsample_Off)
print("signal_subsample_Off", signal_subsample_Off)


############################################
# now we are going to compute the significance in units of sigma
# the formula is in the pdf 3.2 pag 32

N_on = len(t_signal_on)
N_off = len(t_signal_off)
# tau is the ratio between Off and On exposures
# how much more time we have been measuring the Off data wrt the On data
# the statement of the exercise tell us that this is 3.
tau = 3.
#tau = len(t_bkg_off)/len(t_bkg_on)
#tau = len(t_signal_off)/len(t_signal_on)


significance = np.sqrt(2)*np.sqrt(N_on*np.log((tau+1)*(N_on/(N_on+N_off)))
                                  + N_off*np.log(((1+tau)/tau)*(N_off/(N_on+N_off))))
print(significance)
# 2.17973671972 using tau = len(t_bkg_off)/len(t_bkg_on)
# nan using tau = len(t_signal_off)/len(t_signal_on)
# 2.32843463566 using tau = 3.
"""

"""
#################################################
######################### nn ######################
print("###### Neural Network")
# using neural network
# in the function of the nn, the argument data must have only two columns: x,y
# therefore we must eliminate the third column (E)
dataOn_xy = np.delete(dataOn, 2, 1)
dataOff_xy = np.delete(dataOff, 2, 1)

T_cut_max = 0.433633633634

############# DataOn ####################
t_values = neural_network(trainSignal, trainBkg, dataOn_xy)
print(np.shape(t_values))

t_signal_on = t_values[t_values < T_cut_max]
t_bkg_on = t_values[t_values > T_cut_max]
print("Number of signal in On:", len(t_signal_on))
print("Number of bkg in On:", len(t_bkg_on))

############# DataOff ####################
t_values = neural_network(trainSignal, trainBkg, dataOff_xy)
print(np.shape(t_values))

t_signal_off = t_values[t_values < T_cut_max]
t_bkg_off = t_values[t_values > T_cut_max]
print("Number of signal in Off:", len(t_signal_off))
print("Number of bkg in Off:", len(t_bkg_off))

print("signal relation", len(t_signal_off)/len(t_signal_on))
print("bkg relation", len(t_bkg_off)/len(t_bkg_on))

############################################

N_on = len(t_signal_on)
N_off = len(t_signal_off)
tau = 3.

significance = np.sqrt(2)*np.sqrt(N_on*np.log((tau+1)*(N_on/(N_on+N_off)))
                                  + N_off*np.log(((1+tau)/tau)*(N_off/(N_on+N_off))))
print(significance)
# 2.39067540395 using tau = len(t_bkg_off)/len(t_bkg_on)
# nan using tau = len(t_signal_off)/len(t_signal_on)
# 2.49073101281 using tau = 3.
"""

"""
######################### p-value ######################
# this is wrong, can be deleted
# now we compute the p-value (statistical significance) for signal
# we need the pdf f(T|H0), which we computed before as an histogram
# also remember that we already defined this integrals
# as the functions alpha beta
# in fact the formula for computing alpha is the same as the formula
# for computing p. So we just copy the function.
# In the previous exercises, we computed alpha for different values of
# T_cut. We still didn't know the appropiate T_cut.
# Now that we know this T_cut_max, we compute T using this value.
# As a pdf we define a histogram as we did before, but we have to
# change the data to the desired case

def p_value(T_cut, pdf_s):
    # the integral is just the total area underneath the curve.
    # an integral of a histogram is computes as:
    # sum([bin_width[i] * bin_height[i] for i in bin_indexes_to_integrate])
    counts = pdf_s[0] #len Nbins
    bins = pdf_s[1] #len Nbins+1
    # we get the width of each bin (all widths are equal)
    bins_width = bins[1] - bins[0]
    # first we separate the cases if T_cut < any of our values
    # or > any of our values
    sum_counts = 0
    if T_cut < bins[0]:
        return 1.0
    elif T_cut > bins[len(bins)-1]:
        return 0.0
    else:
        # if we have this case we identify the bins we want to integrate
        # then we sum all the counts of them
        for i in range(0, len(bins)):
            if T_cut >= bins[i] and T_cut < bins[i+1]:
                for j in range(i,len(counts)):
                    sum_counts += counts[j]
        return sum_counts*bins_width


pdf_s_on = np.histogram(exact_likelihood_ratio(dataOnX, dataOnY, dataOnX, dataOnY),
                     range = [0,100], normed=True, bins=10000)

p_value_dataOn = p_value(T_cut_max, pdf_s_on)

pdf_s_off = np.histogram(exact_likelihood_ratio(dataOffX, dataOffY, dataOffX, dataOffY),
                     range = [0,100], normed=True, bins=10000)

p_value_dataOff = p_value(T_cut_max, pdf_s_off)

print("p value data_On Exact Likelihood:", p_value_dataOn)
print("p value data_Off Exact Likelihood:", p_value_dataOff)


##########################################################33
# p-value for nn

# in the function of the nn, the argument data must have only two columns: x,y
# therefore we must eliminate the third column (E)
dataOn_xy = np.delete(dataOn, 2, 1)
dataOff_xy = np.delete(dataOff, 2, 1)

T_cut_max = 0.433633633634

pdf_s_on = np.histogram(neural_network(trainSignal, trainBkg, dataOn_xy),
                     normed=True, bins=1000)
pdf_s_off = np.histogram(neural_network(trainSignal, trainBkg, dataOff_xy),
                     normed=True, bins=1000)

p_value_dataOn = p_value(T_cut_max, pdf_s_on)

p_value_dataOff = p_value(T_cut_max, pdf_s_off)

print("p value data_On NN:", p_value_dataOn)
print("p value data_Off NN:", p_value_dataOff)
"""





###################################################################################
#############################      EXERCISE 2    ####################################
##################################################################################

############################# PART A #######################################
# We are going to compute the normalization constants of the energy pdfs
# well, we don't have to do it.

def pdf_h_E(E,E_0,N, sigma_E):
    return 1./N*np.exp(-1/2*((E-E_0)/sigma_E)**2)

"""
sigma_E = 1.

# remember that pyint = scipy.integrate
# scipy.integrate.dblquad Compute a double integral.
# we integrate from E: 0 to 10 and from E_0: 0 to 10
# a normalized pdf should give int = 1. If not int = N
# Therefore, we compute N.
# Then we just have to divide the pdf by this N (we already did in the def pdf_h_E)

Npdf_h_E, Nerr = pyint.dblquad(pdf_h_E, 0, 10,
                lambda E: 0,lambda E: 10,args=(1,sigma_E))

print("The normalization value of pdf(x,y) for signal is: {:.3f} (error is {:.3e})".format(Npdf_h_E,Nerr))

# check that the normalization is properly computed
# The integral should be 1

Norm, Error = pyint.dblquad(pdf_h_E, 0, 10,
                lambda E: 0,lambda E: 10,args=(Npdf_h_E,sigma_E))
print("Once normalized the pdf normalization is: {:.3f} (error is {:.3e})".format(Norm,Error))
"""

def pdf_k_E(E,gamma,N):
    return 1./N*(2+gamma*E)

"""
Npdf_k_E, Nerr = pyint.dblquad(pdf_k_E, 0, 10,
                               lambda E: -1./5, lambda E: 10**8, args=(1,))

print("The normalization value of pdf(x,y) for signal is: {:.3f} (error is {:.3e})".format(Npdf_k_E,Nerr))

# check that the normalization is properly computed
# The integral should be 1

Norm, Error = pyint.dblquad(pdf_k_E, 0, 10,
                lambda E: -1./5, lambda E: 10**8, args=(Npdf_k_E,))
print("Once normalized the pdf normalization is: {:.3f} (error is {:.3e})".format(Norm,Error))
"""

"""
# this code is not exactly correct, the following block "" contains the correct one
# however for this code we obtaina a nice plot
# the difference is that here we minimize, and there we maximize
############################# PART B #######################################
# we are going to compute the profile likelihood ratio test

# First of all, note that the likelihood function only takes the values of
# energy which are below the Tcut. We are going to filter these values:

# since we want to improve the latter values, we use the Tcut used before
# obtained from the exact likelihood ratio
T_cut_max = 0.850085093509
N_on = 385
N_off = 1003
tau = 3

# we read the input files:
dataOnFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_On.txt"
dataOn = np.loadtxt(dataOnFileName, skiprows=2)

# Extract x,y columns
dataOnX = dataOn[:,0]
dataOnY = dataOn[:,1]
dataOnE = dataOn[:,2]

dataOffFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_Off.txt"
dataOff = np.loadtxt(dataOffFileName, skiprows=2)

# Extract x,y columns
dataOffX = dataOff[:,0]
dataOffY = dataOff[:,1]
dataOffE = dataOff[:,2]

# we filter the values of energy:
E_on = []
t_values = exact_likelihood_ratio(dataOnX, dataOnY, dataOnX, dataOnY)
# we get a list of t values for each pair (x,y)
for i in range(0, len(t_values)):
    if t_values[i] < T_cut_max:
        E_on.append(dataOnE[i])

E_off = []
t_values = exact_likelihood_ratio(dataOffX, dataOffY, dataOffX, dataOffY)
for i in range(0, len(t_values)):
    if t_values[i] < T_cut_max:
        E_off.append(dataOffE[i])

# now we define the likelihood funciton:
# it has 4 arguments (s,b,E_0,gamma)

def likelihood(x):
    s = x[0]
    b = x[1]
    E_0 = x[2]
    gamma = x[3]
    sigma_E = 1.
    tau = 3.
    E_on_term = 1.
    for E_i in E_on:
        E_on_term*=(1/(s+b))*(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))
    print(E_on_term)
    E_off_term = 1.
    for E_j in E_off:
        E_off_term*=(2.+gamma*E_j)
    print(E_off_term)
    print(np.exp(-tau*b))
    return ((s+b)**N_on)/factorial(N_on)*np.exp(-(s+b))*((tau*b)**N_off)/factorial(N_off)*\
           np.exp(-tau*b)*(E_on_term*E_off_term)

# we find problems minimizing the likelihood, we best use the log likelihood
# expression in order to avoid overflow numerical problems
# we compute the log_likelihood analytically
# and we simplify the expression in order to avoid overflow terms

def log_likelihood(x):
    s = x[0]
    b = x[1]
    E_0 = x[2]
    gamma = x[3]
    sigma_E = 1.
    tau = 3.
    E_on_term = 0.
    for E_i in E_on:
        E_on_term+= log(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))

    E_off_term = 0.
    for E_j in E_off:
        E_off_term+= log(2.+gamma*E_j)

    #print("s:", s)
    #print("b:",b)
    #print("E_0:", E_0)
    #print("gamma:", gamma)
    #print(E_on_term)
    #print(E_off_term)
    #print(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)
    return -log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term

#######################################################################
# now we are going to maximize the likelihodd (4 variables)
# first of all, there are no modules for maximize
# to maximize we use minimization and then multiply by -1
# since we are going to make the ratio between two maximizations+
# we will not need to multiply by -1
# we use scipy.optimize.minimize:
# Minimization of scalar function of one or more variables

initial_guess = [N_on-1/tau*N_off, 1/tau*N_off, 5., -1/5]
minimize_denominator = pyopt.minimize(log_likelihood, initial_guess,
                                        bounds=((0,None),(1,None),(0,10),(-1/5, None)))
# we must indicate range of b (1,None) because the minimum tends to be in b=0
# and for b=0 there is log(0) math problems
# even if we use b=0.0000001, then the distribution looks bad

print(minimize_denominator)
print(minimize_denominator.x) # we print the 1d array with the parameters that minimize the function
log_likelihood_denominator = minimize_denominator.fun
print(log_likelihood_denominator) # the value of the denominator of the test, as a log L.

####################################
# now we are going to minimize the likelihood numerator,
# ie the likelihood as a funciton of three variables given s
# and simultaneously we compute the profile distribution


def log_likelihood_fixed_s(x,s):
    b = x[0]
    E_0 = x[1]
    gamma = x[2]
    sigma_E = 1.
    tau = 3.
    E_on_term = 0.
    for E_i in E_on:
        E_on_term+= log(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))

    E_off_term = 0.
    for E_j in E_off:
        E_off_term+= log(2.+gamma*E_j)

    #print("s:", s)
    #print("b:",b)
    #print("E_0:", E_0)
    #print("gamma:", gamma)
    #print(E_on_term)
    #print(E_off_term)
    #print(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)
    return -log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term

profile_distribution = [] # here we will store the values of -2*log(lambda)
s_list = [] # store the s values used to compute the profile test
initial_guess = [1/tau*N_off, 5., -1/5]
# we define a range of values of s
for s in np.arange(0,840,20):
    minimize_numerator = pyopt.minimize(log_likelihood_fixed_s, initial_guess, args=s,
                                        bounds=((1,None),(0,10),(-1/5, None)))
    log_likelihood_numerator = minimize_numerator.fun
    print(log_likelihood_numerator) # the value of the numerator of the test, as a log L.
    # note that we have applied a minus sign before the results obtained from the
    # minimization, since we wantes the maximum results instead of the minimum results
    # maximize x is equal to minimize -x
    profile_distribution.append(-2 * (-log_likelihood_numerator + log_likelihood_denominator))
    s_list.append(s)

#np.savetxt("profile_distribution.txt",profile_distribution)
#profile_distribution = np.loadtxt("profile_distribution.txt")

plt.figure()
plt.plot(s_list, profile_distribution, color = "black", label=r"$-2\log \lambda_p(s)$")
#plt.plot(s_list, stats.chi2.pdf(s_list, df=1), color="black", linestyle="dashed", label=r"$\chi^2$")
plt.xlabel("s")
plt.ylabel(r"$-2\log \lambda_p(s)$")
plt.xticks(np.arange(0,900,100))
plt.xlim(0,800)
plt.ylim(2.75533e8, 800+2.75533e8)
#plt.legend(loc="best")
#leg = plt.legend()
#leg.get_frame().set_edgecolor('black')
plt.show()

print("s estimation:", s_list[np.argmax(profile_distribution)])

# what if we plot as a histogram:
# we get a bad plot, which is obvious, since here we want to
# plot a function of s and we have both list of values
# then there is no need to make an histogram

#plt.figure()
#plt.hist(profile_distribution, bins=20, color = "black", label=r"$-2\log \lambda_p(s)$")
#plt.plot(s_list, stats.chi2.pdf(s_list, df=1), color="black", linestyle="dashed", label=r"$\chi^2$")
#plt.show()


# now we compute the statistical significance with the formula of pag 32:
print(profile_distribution[0])
#print(profile_distribution[-1])
statistical_significance = sqrt(profile_distribution[0])
print(statistical_significance)
# S = 16599.1898835218 in the case b range 1,none
"""

"""
# this is the code corrected, 
# here we do the maximizations instead of minimizations
# however the plot is shit
############################# PART B #######################################
# we are going to compute the profile likelihood ratio test

# First of all, note that the likelihood function only takes the values of
# energy which are below the Tcut. We are going to filter these values:

# since we want to improve the latter values, we use the Tcut used before
# obtained from the exact likelihood ratio
T_cut_max = 0.850085093509
N_on = 385
N_off = 1003
tau = 3

# we read the input files:
dataOnFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_On.txt"
dataOn = np.loadtxt(dataOnFileName, skiprows=2)

# Extract x,y columns
dataOnX = dataOn[:,0]
dataOnY = dataOn[:,1]
dataOnE = dataOn[:,2]

dataOffFileName = "C:\\Users\\Aleix López\\Desktop\\Third Exercise" + \
    "\\Scripts\\data_Off.txt"
dataOff = np.loadtxt(dataOffFileName, skiprows=2)

# Extract x,y columns
dataOffX = dataOff[:,0]
dataOffY = dataOff[:,1]
dataOffE = dataOff[:,2]

# we filter the values of energy:
E_on = []
t_values = exact_likelihood_ratio(dataOnX, dataOnY, dataOnX, dataOnY)
# we get a list of t values for each pair (x,y)
for i in range(0, len(t_values)):
    if t_values[i] < T_cut_max:
        E_on.append(dataOnE[i])

E_off = []
t_values = exact_likelihood_ratio(dataOffX, dataOffY, dataOffX, dataOffY)
for i in range(0, len(t_values)):
    if t_values[i] < T_cut_max:
        E_off.append(dataOffE[i])

# now we define the likelihood funciton:
# it has 4 arguments (s,b,E_0,gamma)

def likelihood(x):
    s = x[0]
    b = x[1]
    E_0 = x[2]
    gamma = x[3]
    sigma_E = 1.
    tau = 3.
    E_on_term = 1.
    for E_i in E_on:
        E_on_term*=(1/(s+b))*(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))
    print(E_on_term)
    E_off_term = 1.
    for E_j in E_off:
        E_off_term*=(2.+gamma*E_j)
    print(E_off_term)
    print(np.exp(-tau*b))
    return ((s+b)**N_on)/factorial(N_on)*np.exp(-(s+b))*((tau*b)**N_off)/factorial(N_off)*\
           np.exp(-tau*b)*(E_on_term*E_off_term)

# we find problems minimizing the likelihood, we best use the log likelihood
# expression in order to avoid overflow numerical problems
# we compute the log_likelihood analytically
# and we simplify the expression in order to avoid overflow terms

def log_likelihood(x):
    s = x[0]
    b = x[1]
    E_0 = x[2]
    gamma = x[3]
    sigma_E = 1.
    tau = 3.
    E_on_term = 0.
    for E_i in E_on:
        E_on_term+= log(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))

    E_off_term = 0.
    for E_j in E_off:
        E_off_term+= log(2.+gamma*E_j)

    #print("s:", s)
    #print("b:",b)
    #print("E_0:", E_0)
    #print("gamma:", gamma)
    #print(E_on_term)
    #print(E_off_term)
    #print(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)
    return (-1.)*(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)

#######################################################################
# now we are going to maximize the likelihodd (4 variables)
# first of all, there are no modules for maximize
# to maximize we use minimization of -1*function
# we use scipy.optimize.minimize:
# Minimization of scalar function of one or more variables

initial_guess = [N_on-1/tau*N_off, 1/tau*N_off, 5., -1/5]
minimize_denominator = pyopt.minimize(log_likelihood, initial_guess, method="L-BFGS-B",
                                        bounds=((0,None),(0.0001,None),(0,10),(-1/5, None)))
# be careful with the range of b
# for b=0 there is log(0) math problems
# we should use b=0.00001
# but note that depending on this value, the dis

print(minimize_denominator)
print(minimize_denominator.x) # we print the 1d array with the parameters that minimize the function
log_likelihood_denominator = minimize_denominator.fun
print(log_likelihood_denominator) # the value of the denominator of the test, as a log L.

####################################
# now we are going to minimize the likelihood numerator,
# ie the likelihood as a funciton of three variables given s
# and simultaneously we compute the profile distribution


def log_likelihood_fixed_s(x,s):
    b = x[0]
    E_0 = x[1]
    gamma = x[2]
    sigma_E = 1.
    tau = 3.
    E_on_term = 0.
    for E_i in E_on:
        E_on_term+= log(s*np.exp(-1/2*((E_i-E_0)/sigma_E)**2)+b*(2.+gamma*E_i))

    E_off_term = 0.
    for E_j in E_off:
        E_off_term+= log(2.+gamma*E_j)

    #print("s:", s)
    #print("b:",b)
    #print("E_0:", E_0)
    #print("gamma:", gamma)
    #print(E_on_term)
    #print(E_off_term)
    #print(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)
    return (-1.)*(-log(factorial(N_on))-(s+b)+N_off*log(tau*b)-log(factorial(N_off))-tau*b + E_on_term + E_off_term)

profile_distribution = [] # here we will store the values of -2*log(lambda)
s_list = [] # store the s values used to compute the profile test
initial_guess = [1/tau*N_off, 5., -1/5]
# we define a range of values of s
for s in np.arange(0,300,10):
    minimize_numerator = pyopt.minimize(log_likelihood_fixed_s, initial_guess, args=s, options={"maxiter": 30000},
                                        method= "L-BFGS-B",
                                        bounds=((0.0001,None),(0,10),(-1/5, None)))
    log_likelihood_numerator = minimize_numerator.fun
    print(minimize_numerator.x)
    print(minimize_numerator.success)
    print(log_likelihood_numerator) # the value of the numerator of the test, as a log L.

    profile_distribution.append(-2 * (log_likelihood_numerator - log_likelihood_denominator))
    s_list.append(s)

#np.savetxt("profile_distribution.txt",profile_distribution)
#profile_distribution = np.loadtxt("profile_distribution.txt")

plt.figure()
plt.plot(s_list, profile_distribution, color = "black", label=r"$-2\log \lambda_p(s)$")
#plt.plot(s_list, stats.chi2.pdf(s_list, df=1), color="black", linestyle="dashed", label=r"$\chi^2$")
plt.xlabel("s")
plt.ylabel(r"$-2\log \lambda_p(s)$")
plt.xticks(np.arange(0,900,100))
#plt.xlim(0,800)
#plt.ylim(2.75533e8, 800+2.75533e8)
#plt.legend(loc="best")
#leg = plt.legend()
#leg.get_frame().set_edgecolor('black')
plt.show()

# what if we plot as a histogram:
# we get a bad plot, which is obvious, since here we want to
# plot a function of s and we have both list of values
# then there is no need to make an histogram

#plt.figure()
#plt.hist(profile_distribution, bins=20, color = "black", label=r"$-2\log \lambda_p(s)$")
#plt.plot(s_list, stats.chi2.pdf(s_list, df=1), color="black", linestyle="dashed", label=r"$\chi^2$")
#plt.show()

print("s estimation:", s_list[np.argmax(profile_distribution)])

# now we compute the statistical significance with the formula of pag 32:
print(profile_distribution[0])
#print(profile_distribution[-1])
statistical_significance = sqrt(profile_distribution[0])
print(statistical_significance)
# S = 48 by 0.01
# S = 19.143589163605963 by b=0.0001 and s_estimation = 20
# S = 19.14 by 0.00001
# S = nan by b = 0.000001 or less
# S = 71.44402948078967 by b=1 bound
# S = 19.143589163605963 by b=0.0001

##################
# the s that maximized -2 log(test) is
s_max = 142

minimize_numerator = pyopt.minimize(log_likelihood_fixed_s, initial_guess, args=s_max,
                                    bounds=((0.0001, None), (0, 10), (-1 / 5, None)))
print(minimize_numerator)

s_max = 38

minimize_numerator = pyopt.minimize(log_likelihood_fixed_s, initial_guess, args=s_max,
                                    bounds=((0.0001, None), (0, 10), (-1 / 5, None)))
print(minimize_numerator)

"""









