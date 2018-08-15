# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("E:/mldl/cs231n/assignment1/cs231n/")
#from cs231n.data_utils import load_CIFAR10
from data_utils import load_CIFAR10

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

            # Load the raw CIFAR-10 data.
cifar10_dir = 'E:/mldl/cs231n/assignment1/cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape #5w个32*32*3的图像
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape #1w个32*32*3的图像
print 'Test labels shape: ', y_test.shape

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#num_classes = len(classes)
#samples_per_class = 7
#for y, cls in enumerate(classes):
#    #print y,cls
#    idxs = np.flatnonzero(y_train == y)#y应该是对应物品的标号，比如[5,6,7,0,3,2,...]
#    #print idxs
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    #print idxs
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx].astype('uint8'))#x对应的是真正的物品图片信息
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()

# Subsample the data for more efficient code execution in this exercise
#挑前5000个数据
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]#对应的是结果

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print X_train.shape[0],X_train.shape
print y_train.shape[0],y_train.shape
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))#把图像数据转为5000 * 3072
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

sys.path.append("E:/mldl/cs231n/assignment1/cs231n/classifiers/");
from classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print dists.shape

# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
#plt.imshow(dists, interpolation='none')
#plt.show()

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=5)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'
  
  # Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'
  
  # Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation