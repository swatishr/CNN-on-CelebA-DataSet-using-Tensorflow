# -*- coding: utf-8 -*-
"""
**************************************************************************
*                  
*                  ================================
*  This software is intended to teach image processing concepts
*	
*  UBIT Name: swatishr, jayantso
*  Person Number: 550246994, 0246821
*  MODULE: Project-4
*  Filename: main.py
*  Date: Dec 6, 2017
*  
*  Author: Swati Nair, Jayant Solanki, CSE-574 Project-2, Department of Computer Science
*  and Engineering, University at Buffalo.
*  
*  Software released under Creative Commons CC BY-NC-SA
*
*  For legal information refer to:
*        http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode 
*     
*
*  This software is made available on an “AS IS WHERE IS BASIS”. 
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*
**************************************************************************
"""
#Description: To implement a convolutional neural network to determine 
#whether the person in a portrait image is wearing glasses or not

#########################################################################
#***********************Necessary libraries***************************#
# import tensorflow as tf
import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
from libs import *
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
global dropOut
global layer1Nodes
global layer2Nodes
############Path to the image and the labels file##################
filename = '../data/CelebA/Anno/list_attr_celeba.txt'
imagepath = '../data/CelebA/Img/img_align_celeba/'
celebData = 0
(celebData, labels, imageNames) = dataloader2(imagepath, filename)#load the 70000 dataset

#(celebData, labels, imageNames) = dataloader(imagepath, filename)
# celebData = celebData[0:100000,:,:,:]
# celebData = celebData[0:100000,:,:]
# labels = labels[0:100000,:]
# imageNames = imageNames[0:100000]
# print("Celebdata dimension is ", celebData.shape)
# print("labels dimension is ", labels.shape)
# print("imageNames dimension is ", imageNames.shape)
# # view_image(normalized_X[9999,:,:], train_images_label[9999])
# trains_images = celebData[0:20000,:,:,:]#getting the training sets
# test_images = celebData[20000:25000,:,:,:]#getting the training sets
# train_images_labels = labels[0:20000,:]
# test_images_labels = labels[20000:25000,:]
# # trains_images = trains_images.reshape([20000,2475*3])#flattening the input array
# test_images = test_images.reshape([5000,2475*3])

# view_image(normalized_X[9999,:,:], train_images_label[9999])
#working on the 70000 dataset
train_num = 50400 #80% of 70000
val_num = 5600
test_num = 14000

trains_images = celebData[0:train_num,:,:]#getting the training sets
train_images_labels = labels[0:train_num,:]

val_images = celebData[train_num:train_num+val_num,:,:]#getting the validation sets
val_images_labels = labels[train_num:train_num+val_num,:]

test_images = celebData[train_num+val_num:train_num+val_num+test_num,:,:]#getting the training sets
test_images_labels = labels[train_num+val_num:train_num+val_num+test_num,:]

#flattening the input array and reshaping the labels as per requirement of the tnesorflow algo
trains_images = trains_images.reshape([train_num,784])
val_images = val_images.reshape([val_num,784])
test_images = test_images.reshape([test_num,784])
train_images_labels = train_images_labels.reshape([train_num,])
val_images_labels = val_images_labels.reshape([val_num,])
test_images_labels = test_images_labels.reshape([test_num,])

#standardizing the image data set with zero mean and unit standard deviation
trains_images = preprocessing.scale(trains_images)
val_images = preprocessing.scale(val_images)
test_images = preprocessing.scale(test_images)

#creating one-hot vectors for labels
# train_images_labels_mat = np.zeros((train_num, 2), dtype=np.uint8)
# train_images_labels_mat[np.arange(train_num), train_images_labels.T] = 1
# # train_images_labels = train_images_labels_mat
# # print(train_images_labels[51,:])

# val_images_labels_mat = np.zeros((val_num, 2), dtype=np.uint8)
# val_images_labels_mat[np.arange(val_num), val_images_labels.T] = 1
# # val_images.;/._labels = val_images_labels_mat

# test_images_labels_mat = np.zeros((test_num, 2), dtype=np.uint8)
# test_images_labels_mat[np.arange(test_num), test_images_labels.T] = 1
# test_images_labels = test_images_labels_mat

# print("Train images shape: ", trains_images.shape)
# print("Train labels shape: ", train_images_labels.shape)
# print("Test images shape: ", val_images.shape)
# print("Test labels shape: ", val_images_labels.shape)
# print("Test images shape: ", test_images.shape)
# print("Test labels shape: ", test_images_labels.shape)


###########################################################
#function for building the model of the CNN
#function cnn_model_fn(features, labels, mode)
#input : fearures are the 784 input features of each image, labels nd mode in which the CNN model is run
#output : Estimator for the model
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  global dropOut#globally declaring it be used as shared vairable
  global layer1Nodes
  global layer2Nodes

  # Input Layer
  # Celeb images are 28x28 pixels, and have one color channel
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=layer1Nodes, #32
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=layer2Nodes, #64
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * layer2Nodes])


  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.4 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=dropOut, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Calculate Loss (for both TRAIN and EVAL modes)
  # onehot_labels = labels
  # loss = tf.losses.softmax_cross_entropy(
  #     onehot_labels=onehot_labels, logits=logits)
  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
  	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


###########################################################
#function for running CNN on the Celeb data
#function main(dropout, layer1nodes, layer2nodes)
#input : dropout value (default value: 0.4), layer1nodes is the number of units in first hidden layer (default: 32), layer2nodes: nodes in 2nd layer(default:64) 
def main(dropout=0.4, layer1nodes=32, layer2nodes=64):
	global dropOut
	global layer1Nodes
	global layer2Nodes
	dropOut = dropout
	layer1Nodes = layer1nodes
	layer2Nodes = layer2nodes

	#saving the trained model on this path
	modelName = "tfc/tfcache/celeb_convnet_model"+str(dropOut)+str(layer1Nodes)+str(layer2Nodes)

	# Load training and eval data
	train_data = np.asarray(trains_images, dtype=np.float32)  # Returns np.array
	train_labels = train_images_labels
	eval_data = np.asarray(val_images, dtype=np.float32)  # Returns np.array 
	eval_labels = val_images_labels
	test_data = np.asarray(test_images, dtype=np.float32)  # Returns np.array 
	test_labels = test_images_labels

	# Create the Estimator
	celeb_classifier = tf.estimator.Estimator(
	  model_fn=cnn_model_fn, model_dir=modelName)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=1000)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": train_data},
	  y=train_labels,
	  batch_size=100,
	  num_epochs=None,
	  shuffle=True)
	celeb_classifier.train(
	  input_fn=train_input_fn,
	  steps=1000)#epoch count

	# Evaluate the training set and print results
	Train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": train_data},
	  y=train_labels,
	  num_epochs=1,
	  shuffle=False)
	train_results = celeb_classifier.evaluate(input_fn=Train_input_fn)
	print("Training set accuracy" ,train_results)

	# Evaluate the validation set and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": eval_data},
	  y=eval_labels,
	  num_epochs=1,
	  shuffle=False)
	eval_results = celeb_classifier.evaluate(input_fn=eval_input_fn)
	print("validation set accuracy" ,eval_results)

	# Evaluate the Test set and print results
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": test_data},
	  y=test_labels,
	  num_epochs=1,
	  shuffle=False)
	test_results = celeb_classifier.evaluate(input_fn=test_input_fn)
	print("Test accuracy" ,test_results)


if __name__ == "__main__":
    main()
