# -*- coding: utf-8 -*-
# A bit of setup

import time, os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image, preprocess_image

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)
model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')

#def create_class_visualization(target_y, model, **kwargs):
#  """
#  Perform optimization over the image to generate class visualizations.
#  
#  Inputs:
#  - target_y: Integer in the range [0, 100) giving the target class
#  - model: A PretrainedCNN that will be used for generation
#  
#  Keyword arguments:
#  - learning_rate: Floating point number giving the learning rate
#  - blur_every: An integer; how often to blur the image as a regularizer
#  - l2_reg: Floating point number giving L2 regularization strength on the image;
#    this is lambda in the equation above.
#  - max_jitter: How much random jitter to add to the image as regularization
#  - num_iterations: How many iterations to run for
#  - show_every: How often to show the image
#  """
#  
#  learning_rate = kwargs.pop('learning_rate', 10000)
#  blur_every = kwargs.pop('blur_every', 1)
#  l2_reg = kwargs.pop('l2_reg', 1e-6)
#  max_jitter = kwargs.pop('max_jitter', 4)
#  num_iterations = kwargs.pop('num_iterations', 100)
#  show_every = kwargs.pop('show_every', 25)
#  
#  X = np.random.randn(1, 3, 64, 64)
#  for t in range(num_iterations):
#    # As a regularizer, add random jitter to the image
#    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
#    X = np.roll(np.roll(X, ox, -1), oy, -2)
#
#    dX = None
#    ############################################################################
#    # TODO: Compute the image gradient dX of the image with respect to the     #
#    # target_y class score. This should be similar to the fooling images. Also #
#    # add L2 regularization to dX and update the image X using the image       #
#    # gradient and the learning rate.                                          #
#    ############################################################################
#    # Compute the score and gradient
#    scores, cache = model.forward(X, mode='test')
#    # loss = scores[0, target_y] - l2_reg*np.sum(X**2)
#    dscores = np.zeros_like(scores)
#    dscores[0, target_y] = 1.0
#    dX, grads = model.backward(dscores, cache)
#    dX -= 2*l2_reg*X
#
#    X += learning_rate*dX
#    ############################################################################
#    #                             END OF YOUR CODE                             #
#    ############################################################################
#    
#    # Undo the jitter
#    X = np.roll(np.roll(X, -ox, -1), -oy, -2)
#    
#    # As a regularizer, clip the image
#    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
#    
#    # As a regularizer, periodically blur the image
#    if t % blur_every == 0:
#      X = blur_image(X)
#    
#    # Periodically show the image
#    if t % show_every == 0:
#      plt.imshow(deprocess_image(X, data['mean_image']))
#      plt.gcf().set_size_inches(3, 3)
#      plt.axis('off')
#      plt.show()
#  return X
#
#target_y = 43 # Tarantula
#print( data['class_names'][target_y])
#X = create_class_visualization(target_y, model, show_every=25)


def invert_features(target_feats, layer, model, **kwargs):
  """
  Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
  L2 regularization and periodic blurring.
  
  Inputs:
  - target_feats: Image features of the target image, of shape (1, C, H, W);
    we will try to generate an image that matches these features
  - layer: The index of the layer from which the features were extracted
  - model: A PretrainedCNN that was used to extract features
  
  Keyword arguments:
  - learning_rate: The learning rate to use for gradient descent
  - num_iterations: The number of iterations to use for gradient descent
  - l2_reg: The strength of L2 regularization to use; this is lambda in the
    equation above.
  - blur_every: How often to blur the image as implicit regularization; set
    to 0 to disable blurring.
  - show_every: How often to show the generated image; set to 0 to disable
    showing intermediate reuslts.
    
  Returns:
  - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
  """
  learning_rate = kwargs.pop('learning_rate', 10000)
  num_iterations = kwargs.pop('num_iterations', 500)
  l2_reg = kwargs.pop('l2_reg', 1e-7)
  blur_every = kwargs.pop('blur_every', 1)
  show_every = kwargs.pop('show_every', 50)
  
  X = np.random.randn(1, 3, 64, 64)
  for t in range(num_iterations):
    ############################################################################
    # TODO: Compute the image gradient dX of the reconstruction loss with      #
    # respect to the image. You should include L2 regularization penalizing    #
    # large pixel values in the generated image using the l2_reg parameter;    #
    # then update the generated image using the learning_rate from above.      #
    ############################################################################
    # Forward until target layer
    feats, cache = model.forward(X, end=layer, mode='test')

    # Compute the loss
    loss = np.sum((feats-target_feats)**2) + l2_reg*np.sum(X**2)

    # Compute the gradient of the loss with respect to the activation
    dfeats = 2*(feats-target_feats)
    dX, grads = model.backward(dfeats, cache)
    dX += 2*l2_reg*X

    X -= learning_rate*dX

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    # As a regularizer, clip the image
    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
    
    # As a regularizer, periodically blur the image
    if (blur_every > 0) and t % blur_every == 0:
      X = blur_image(X)

    if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
      plt.imshow(deprocess_image(X, data['mean_image']))
      plt.gcf().set_size_inches(3, 3)
      plt.axis('off')
      plt.title('t = %d' % t)
      plt.show()

filename = 'kitten.jpg'
layer = 3 # layers start from 0 so these are features after 4 convolutions
img = imresize(imread(filename), (64, 64))

plt.imshow(img)
plt.gcf().set_size_inches(3, 3)
plt.title('Original image')
plt.axis('off')
plt.show()

# Preprocess the image before passing it to the network:
# subtract the mean, add a dimension, etc
img_pre = preprocess_image(img, data['mean_image'])


# Extract features from the image
feats, _ = model.forward(img_pre, end=layer)

# Invert the features
kwargs = {
  'num_iterations': 400,
  'learning_rate': 5000,
  'l2_reg': 1e-8,
  'show_every': 100,
  'blur_every': 10,
}
X = invert_features(feats, layer, model, **kwargs)