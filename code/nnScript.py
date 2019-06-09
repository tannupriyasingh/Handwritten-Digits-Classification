import numpy as np
import timeit
import csv
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # your code here

    sig = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sig


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.


    # Feature selection
    # Your code here.

    train_size = 50000
    feature_size = 784
    test_size = 10000
    preprocess_train = np.zeros(shape=(train_size, feature_size))
    preprocess_validation = np.zeros(shape=(test_size, feature_size))
    preprocess_test = np.zeros(shape=(test_size, feature_size))
    preprocess_train_label = np.zeros(shape=(train_size,))
    preprocess_validation_label = np.zeros(shape=(test_size,))
    preprocess_test_label = np.zeros(shape=(test_size,))

    len_train = 0
    len_validation = 0
    len_test = 0
    len_train_label = 0
    len_validation_label = 0
    reduceBy = 1000

    for i in mat:
        data = mat.get(i)
        length = len(data)
        if "train" in i:
            adjust = length - reduceBy

            preprocess_train, len_train, train_label, len_train_label = data_add(preprocess_train, len_train, adjust,
                                                                                 data[
                                                                                 np.random.permutation(range(length))[
                                                                                 1000:], :], i, preprocess_train_label,
                                                                                 len_train_label)

            preprocess_validation, len_validation, preprocess_validation_label, len_validation_label = data_add(
                preprocess_validation, len_validation, 1000, data[np.random.permutation(range(length))[0:1000], :], i,
                preprocess_validation_label, len_validation_label)


        elif "test" in i:
            preprocess_test_label[len_test:len_test + length] = i[len(i) - 1]
            preprocess_test[len_test:len_test + length] = data[np.random.permutation(range(length))]
            len_test += length

    global features
    features = []
    removed = []
    deviation = np.std(preprocess_train,axis=0)
    for i in range(len(deviation)):
        if deviation[i] > 1.0:
            features.append(i)
        else:
            removed.append(i)
    # print(len(removed)) <- Features removed
    preprocess_train = preprocess_train[:, features]
    preprocess_validation = preprocess_validation[:, features]
    preprocess_test = preprocess_test[:, features]

    train_size = range(preprocess_train.shape[0])
    train_data, train_label = dsn(train_size, preprocess_train, preprocess_train_label)

    # print(train_preprocess[49999])
    validation_size = range(preprocess_validation.shape[0])
    validation_data, validation_label = dsn(validation_size, preprocess_validation, preprocess_validation_label)

    test_size = range(preprocess_test.shape[0])
    test_data, test_label = dsn(test_size, preprocess_test, preprocess_test_label)

    #print(train_data.shape)
    # print(test_data.shape)
    # print(validation_data.shape)

    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def data_add(newData, datalen, adjust, data, key, datalabel, datalabellen):
    newData[datalen:datalen + adjust] = data
    datalen += adjust

    datalabel[datalabellen:datalabellen + adjust] = key[len(key) - 1]
    datalabellen += adjust
    return newData, datalen, datalabel, datalabellen


def dsn(size, pre_process, label_preprocess):  # Double, Shuffle, Normalize

    train_perm = np.random.permutation(size)
    train_data = pre_process[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = label_preprocess[train_perm]
    return train_data, train_label

def one_of_k(tl, nc):
    result = np.eye(nc)[np.array(tl).reshape(-1)]
    return result.reshape(list(tl.shape)+[nc])

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    y = one_of_k(training_label.astype(int),n_class)
    bias_training = np.ones(training_label.shape[0])
    new_training_data = np.column_stack((training_data,bias_training))
    sigmoid_input1 = np.dot(new_training_data,w1.T)
    first_hidden_output = sigmoid(np.dot(new_training_data, w1.T))
    bias_add = np.ones(first_hidden_output.shape[0])
    new_bias_data = np.column_stack((first_hidden_output,bias_add))
    sigmoid_input2 = np.dot(new_bias_data,w2.T)
    last_hidden_output = sigmoid(sigmoid_input2)
    error = last_hidden_output - y
    gradient_w1 = np.dot(((1 - new_bias_data) * new_bias_data * (np.dot(error, w2))).T, new_training_data)
    gradient_w2 = np.dot(error.T, new_bias_data)
    obj_val = (np.sum(-1 * (y * np.log(last_hidden_output) + (1 - y) * np.log(1 - last_hidden_output)))) / new_training_data.shape[0] + (
                (lambdaval / (2 * new_training_data.shape[0])) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    gradient_w1 = np.delete(gradient_w1, n_hidden,0)
    gradient_w1 = (gradient_w1 + (lambdaval * w1)) / new_training_data.shape[0]
    gradient_w2 = (gradient_w2 + (lambdaval * w2)) / new_training_data.shape[0]

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    bias = np.ones(data.shape[0]) * 1
    data = np.column_stack((data,bias))
    z = sigmoid(np.dot(data,w1.T))
    z = np.column_stack((z,bias.T))
    o = sigmoid(np.dot(z,w2.T))
    labels = o.argmax(axis=1)
    # print(labels.shape)
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5  # <--- need to tweak

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

# start = timeit.default_timer()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj_dump = [features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj_dump, open("params.pickle", "wb"))

# stop = timeit.default_timer()
# total_time = stop - start
# print(total_time)
