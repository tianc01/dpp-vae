from __future__ import print_function

# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import input_data_ratio10to1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdb import set_trace as st
from utils import *
import os
import sys
from PIL import Image
from scipy.misc import imsave as ims
from scipy.misc import imshow

restore_sess = True
# img_name = 'dppvae_10to1.jpg'
img_name = sys.argv[1]
num_random_samples = 900
im_w, im_h = 30, 30

images = []
all_images = np.array(Image.open('results/'+img_name))
for i in range(im_w):
  for j in range(im_h):
    images.append(all_images[i*28:(i+1)*28, j*28:(j+1)*28])
images = np.array(images[:num_random_samples])
test_images = np.reshape(images, (images.shape[0], 784))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 2],  name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  saver = tf.train.Saver()
  sess = tf.Session()
  if restore_sess:
    saver = tf.train.import_meta_graph("cnn_mnist01.meta")
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("Model restored.")
    max_iters = 2
  else:
    sess.run(tf.global_variables_initializer())
    mnist = input_data_ratio10to1.read_data_sets("MNIST_data/", one_hot=True)
    max_steps = 1000
  
    for step in range(max_steps):
      batch_xs, batch_ys = mnist.balance01.next_batch(50)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    save_path = saver.save(sess, "./cnn_mnist01")
    print(max_steps, sess.run(accuracy, feed_dict={x: mnist.balance01_test.images, y_: mnist.balance01_test.labels, keep_prob: 1.0}))

  # Predictions
  y_pred = np.zeros((1,2))
  for idx in range(num_random_samples//100):
    cur_y_pred = sess.run(y, feed_dict={x: test_images[idx*100:(idx+1)*100], keep_prob: 1.0})
    y_pred = np.concatenate((y_pred, cur_y_pred), axis=0)
  y_pred = y_pred[1:]

  # Order images and save
  ordered_class0_images = []
  ordered_class1_images = []

  for pred, img in zip(y_pred, images):
    if np.argmax(pred) == 0:
      ordered_class0_images.append(img)
    else:
      ordered_class1_images.append(img)

  minor_class_p = len(ordered_class1_images) / (len(ordered_class1_images) + len(ordered_class0_images))
  print('Class 0: {} Class 1: {} Minor Class Percent: {}'.format(len(ordered_class0_images), len(ordered_class1_images), minor_class_p))
  
  ordered_all_images = np.array(ordered_class1_images+ordered_class0_images)
  ims('results/ordered_'+img_name,merge(ordered_all_images,[im_w, im_h]))