#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      chris
#
# Created:     26/04/2018
# Copyright:   (c) chris 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import getsizeof
import numpy as np
import tensorflow as tf
import os
import sys
from sklearn.model_selection import train_test_split
import secrets

tf.logging.set_verbosity(tf.logging.INFO)

def random_create_training_test_samples(pos_file_path,neg_file_path,pos_file_list,neg_file_list,cv=5,dim=580):
    #Given file pathes, random sample from positive and negative groups as traning set and test set.
    #Return dict objects of numarrays training and test samples.
    output_dic={}
    total_pos=len(pos_file_list)
    total_neg=len(neg_file_list)

    test_pos_no=int(total_pos/cv)
    train_pos_no=total_pos-test_pos_no
    test_neg_no=int(total_neg/cv)
    train_neg_no=total_neg-test_neg_no

    #Load data into numpy objects
    pos_sample_mat=np.zeros(shape=[total_pos,dim*dim])#flattened images
    neg_sample_mat=np.zeros(shape=[total_neg,dim*dim])#each row is a sample, each column is pixel value
    pos_idx_all=np.arange(total_pos)#index for all positive samples
    neg_idx_all=np.arange(total_neg)#index for all negative samples
    for i in range(total_pos):
        tmp=np.genfromtxt(pos_file_path+"/"+pos_file_list[i])#read the image (580*580) as np.array
        tmp=tmp.reshape(dim*dim)#flatten
        pos_sample_mat[i,:]=tmp
    for i in range(total_neg):
        tmp=np.genfromtxt(neg_file_path+"/"+neg_file_list[i])#read the image (580*580) as np.array
        tmp=tmp.reshape(dim*dim)#flatten
        neg_sample_mat[i,:]=tmp
    #Random split training and test samples
    #Positive samples
    train_pos_idx=np.random.choice(total_pos,train_pos_no,replace=False)#generate random index
    mask_pos=np.zeros(total_pos,dtype=bool)
    mask_pos[train_pos_idx]=True
    test_pos_idx=pos_idx_all[~mask_pos]
    #Negative samples
    train_neg_idx=np.random.choice(total_neg,train_neg_no,replace=False)
    mask_neg=np.zeros(total_neg,dtype=bool)
    mask_neg[train_neg_idx]=True
    test_neg_idx=neg_idx_all[~mask_neg]
    #Extract the data
    train_pos_data=pos_sample_mat[train_pos_idx,:]
    train_neg_data=neg_sample_mat[train_neg_idx,:]
    test_pos_data=pos_sample_mat[test_pos_idx,:]
    test_neg_data=neg_sample_mat[test_neg_idx,:]
    output_dic["train_pos"]=train_pos_data
    output_dic["train_neg"]=train_neg_data
    output_dic["test_pos"]=test_pos_data
    output_dic["test_neg"]=test_neg_data
    output_dic["train_pos_idx"]=train_pos_idx
    output_dic["train_neg_idx"]=train_neg_idx
    return(output_dic)
#The following function defines how tensorflow treats every sample
#Feature and label, define the loss function.
#Mode defines how to train (optimize), prediction and evaluation metrics (accuracy, AUC false positive etc.)
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # model receives input data from input_fn argument, which is a function
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 580, 580, 1])#extract input from "features" argument

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 580, 580, 1]
  # Output Tensor Shape: [batch_size, 580, 580, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)#padding is keep the dimension

  # Pooling Layer #1
  # First max pooling layer with a 4x4 filter and stride of 4
  # Input Tensor Shape: [batch_size, 580, 580, 32]
  # Output Tensor Shape: [batch_size, 145, 145, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)

  # Convolutional Layer #2
  # Since this is a 3D matrix, the filter would become 3D matrix as well
  # This is a 3D filter process. Details check andrew Ng's youtube video.
  # https://www.youtube.com/watch?v=Uolm8oLPocA&index=1&list=FL0hMlUUfOamwgsjojFEXKXg&t=28s
  # Computes 64 feature maps using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 145, 145, 32]
  # Output Tensor Shape: [batch_size, 145, 145, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 5x5 filter and stride of 5
  # Input Tensor Shape: [batch_size, 145, 145,64]
  # Output Tensor Shape: [batch_size, 29, 29, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 29, 29, 64]
  # Output Tensor Shape: [batch_size, 29 * 29 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 29 * 29 * 64])#Flatten all features

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 29 * 29 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer (Last layer)
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  # predictions defines how to make prediction based on input data
  logits = tf.layers.dense(inputs=dropout, units=2)#binary classification for pri-microRNA and non-pri-microRNA
  #How do we assign prediction in the model
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
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)#labels is extracted from input_fn

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)#initialize gradient descent optimizer class
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
def main(unused_argv):
  # Training and test data set up, both numpy.arrays

##  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
##
##  train_data = mnist.train.images  # Returns np.array， 55000 by 784, 784=28*28 flattened pixels
##  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
##  eval_data = mnist.test.images  # Returns np.array
##  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  #Create training and test data sets
  data_path="D:/Research_data/Jun_lu/"
  positive_data_path=data_path+"positive_contact_matrix"
  negative_data_path=data_path+"negative_contact_matrix"
  miRNA_positive_list_all=os.listdir(positive_data_path)
  miRNA_negative_list_all=os.listdir(negative_data_path)
  total_sample=1000
  train_sample=int(total_sample*0.8)
  test_sample=total_sample-train_sample
  miRNA_positive_rand_idx=list(np.random.choice(len(miRNA_positive_list_all),total_sample,False))
  miRNA_positive_list=[miRNA_positive_list_all[each] for each in miRNA_positive_rand_idx]
  miRNA_negative_rand_idx=list(np.random.choice(len(miRNA_negative_list_all),total_sample,False))#sample negative samples without replacement
  miRNA_negative_list=[miRNA_negative_list_all[each] for each in miRNA_negative_rand_idx]
  all_samples_numpy=random_create_training_test_samples(positive_data_path,negative_data_path,miRNA_positive_list,miRNA_negative_list)#read the data and flatten it, return dicts of numpy array
##  train_data=np.vstack((all_samples_numpy["train_pos"],all_samples_numpy["train_neg"]))
##  train_data=train_data.astype("float32")
##  train_pos_label=np.zeros(train_sample)#np.hstack((np.ones(train_sample),np.zeros(train_sample)))
##  train_neg_label=np.ones(train_sample)#np.hstack((np.zeros(train_sample),np.ones(train_sample)))
##  train_labels=np.hstack((train_pos_label,train_neg_label))#np.transpose(np.vstack((train_pos_label,train_neg_label)))
##  train_labels=train_labels.astype("int32")
  # Create the Estimator Class defined in Tensorflow, cnn_model_fn is defined as a fucntion which describes how to process each sample
  miRNA_cnn_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:/Research_data/Jun_lu/model_output")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model with input image data, numpy_input_fu(nction) is a function that convert input numpy array into an Estimator class compatible format

##  train_input_fn = tf.estimator.inputs.numpy_input_fn(
##      x={"x": train_data},
##      y=train_labels,
##      batch_size=10,
##      num_epochs=None,
##      shuffle=True)
##  #Optimization (Training) to obtain the parameter
##  miRNA_cnn_classifier.train(
##      input_fn=train_input_fn,#input_fn takes a function as input
##      steps=5000,
##      hooks=[logging_hook])
##  eval_data=np.vstack((all_samples_numpy["test_pos"],all_samples_numpy["test_neg"]))
##  eval_data=eval_data.astype("float32")
##  eval_pos_label=np.zeros(test_sample)#np.hstack((np.ones(test_sample),np.zeros(test_sample)))
##  eval_neg_label=np.ones(test_sample)#np.hstack((np.zeros(test_sample),np.ones(test_sample)))
##  eval_labels=np.hstack((eval_pos_label,eval_neg_label))#np.transpose(np.vstack((eval_pos_label,eval_neg_label)))
##  eval_labels=eval_labels.astype("int32")
  eval_data=np.vstack((all_samples_numpy["train_pos"],all_samples_numpy["test_pos"],all_samples_numpy["train_neg"],all_samples_numpy["test_neg"]))
  eval_data=eval_data.astype("float32")
  eval_pos_label=np.zeros(total_sample)
  eval_neg_label=np.ones(total_sample)
  eval_labels=np.hstack((eval_pos_label,eval_neg_label))
  eval_labels=eval_labels.astype("int32")
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      batch_size=20,
      num_epochs=1,
      shuffle=False)
  eval_results = miRNA_cnn_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()