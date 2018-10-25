"""
knn_yk.py
Created on Sun Aug 10 12:51:35 2014

@author: Dongwook Shin, Yash Kanoria
User input (in the main function):
    - 'knn_train_data.csv'
        Data samples in rows, outcome variable in first col
    - 'knn_test_data.csv'  Data samples in rows
    - k: the number of nearest neighbors desired

Output:
    - [x_train, y_train]: train data read from the file
    - [x_test, y_test]: test data matrix, x_test,
                        and corresponding predicted values, y_test
"""

import numpy as np
import scipy.spatial.distance as ssd
import time

def knn_predict(x_train, y_train, x_test, k):
    """ function for out-of-sample prediction using k-NN algorithm
        Inputs:
            x_train: 2D array containing training samples in rows
            y_train: 1D array containing outcomes for training samples
            x_test:  2D array containing test samples in rows
            k:       Number of neighbors to use

        Output:
            List containing predictions for test samples
    """
    # initialize list to store predicted class
    y_test = []
    # for each instance in data testing,
    # calculate distance in respect to data training
    for i, di in enumerate(x_test):
        distances = []  # initialize list to store distance
        for j, dj in enumerate(x_train):
            # calculate distances
            distances.append((dist_euclidean(di, dj), y_train[j]))
        # k-neighbors
        sorted_distances = sorted(distances)[:k]

        # predict the outcome for the instance
        y_test.append(np.mean(sorted_distances, axis=0)[1])
        # or do np.mean([y for (dist, y) in temp])

    # return predicted outcome
    return y_test


def dist_euclidean(di, dj):
    """ Distance calculation between di and dj"""
    return ssd.euclidean(di, dj)  # built-in Euclidean fn


# initialize runtime
start = time.clock()

# read data from .csv files
train_data = np.loadtxt('knn_train_data.csv', delimiter=',', dtype="float")
test_data = np.loadtxt('knn_test_data.csv', delimiter=',', dtype="float")

# translate data to x_train, y_train, and x_test array
y_train = train_data[:, 0]
x_train = train_data[:, 1:]
x_test = test_data[:, :]

# set k, the number of nearest neighbors desired
k = 3
# run k-NN algorithm to predict y_test values for x_test data
y_test = knn_predict(x_train, y_train, x_test, k)

# print out results
print("\nk-NN train data:")
print("x_train \t\t | y_train")
for i, di in enumerate(x_train):
    a = ','.join('{:5.2f}'.format(k) for ik, k in enumerate(di))
    strTrainRow = a + "\t\t | " + '{:5.1f}'.format(y_train[i])
    print(strTrainRow)

print("\nk-NN prediction results for test data with k = %d" % k)
print("x_test \t\t\t | y_test")
for i, di in enumerate(x_test):
    a = ','.join('{:5.1f}'.format(k) for ik, k in enumerate(di))
    strTestRow = a + "\t\t | " + '{:5.2f}'.format(y_test[i])
    print(strTestRow)
# # Find runtime
run_time = time.clock() - start
print("\nRuntime:", '%.4f' % run_time, "seconds")