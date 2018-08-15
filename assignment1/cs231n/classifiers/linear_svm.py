# -*- coding: utf-8 -*-

import numpy as np
from random import shuffle

#W (3073, 10) X (500L, 3073)
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #print num_classes,num_train
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # 1x3073 dot 3073x10 = 1x10
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #dW 3073*10 求偏导
        dW[:,j] += X[i].T
        dW[:,y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  #注意这里多了0.5，所以求导的w^2的2就没有了
  loss += 0.5 * reg * np.sum(W * W)
  dW +=reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)        # N by C 500x10
  num_train = X.shape[0]
  #num_classes = W.shape[1]
  #print 'scores=',scores.shape
  scores_correct = scores[np.arange(num_train), y]  # 根据坐标拿出对应的正确label的分数
  scores_correct = np.reshape(scores_correct, (num_train, 1))  # 500 by 1
  #print 'scores_correct=',scores_correct.shape
  margins = scores - scores_correct + 1.0     # 500x10
  margins[np.arange(num_train), y] = 0.0
  margins[margins <= 0] = 0.0
  loss += np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  margins[margins > 0] = 1.0
  #print margins
  # 下面就是两条偏导公式的转化
  row_sum = np.sum(margins, axis=1) 
  #这里统计了需要减X.T的次数，加和减都统一在下式解决
  margins[np.arange(num_train), y] = -row_sum        
  dW += np.dot(X.T, margins)/num_train + reg * W     # 3073 x 10

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
