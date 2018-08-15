# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_calss = W.shape[1]
  num_train = X.shape[0]
  #print dW.shape,num_calss,num_train,X.shape
  #(3073L, 10L) 10 500 (500L, 3073L)
  
  buf_e = np.zeros(num_calss)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train) :
     for j in xrange(num_calss) :
       #1*3073 * 3073*1 = 1 >>>10
       buf_e[j] = np.dot(X[i,:],W[:,j])#每个样本在num_calss个种类中的分数
     buf_e -= np.max(buf_e)
     buf_e = np.exp(buf_e)
     buf_sum = np.sum(buf_e)    
     buf = buf_e/ buf_sum
     loss -= np.log(buf[y[i]] )
     for j in xrange(num_calss):
         dW[:,j] +=( buf[j] - (j ==y[i]) )*X[i,:].T 
   #regularization with elementwise production
  loss /= num_train
  dW /= num_train
 
  loss += 0.5 * reg * np.sum(W * W)
  dW +=reg*W
      

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #num_calss = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #500*3073 3073*10 >>>500*10
  buf_e  = np.dot(X,W)
  # 10 * 500 - 1*500   T ->(500L, 10L)
  buf_e = np.subtract( buf_e.T , np.max(buf_e , axis = 1) ).T
  buf_e = np.exp(buf_e)
  #10*500 / 1*500 T ->(500L, 10L)
  buf_e = np.divide( buf_e.T , np.sum(buf_e , axis = 1) ).T
  #get loss
  #print buf_e[np.arange(num_train),y]
  loss = - np.sum(np.log ( buf_e[np.arange(num_train),y] ) ) 
  #get grad 
  buf_e[np.arange(num_train),y]  -= 1   
  # 3073 * 500 * 500*10
  loss /=num_train  + 0.5 * reg * np.sum(W * W)
  dW = np.dot(X.T,buf_e)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW






