import tensorflow as tf
import numpy as np
import input_data_ratio10to1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, math
from utils import *
from ops import *
from pdb import set_trace as st
from lr_trainer import LRTrainer
import pickle
import time
import sys
from scipy.misc import imsave as ims
import random

class LatentAttention():
    def __init__(self):
        self.mnist = input_data_ratio10to1.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.unbalance01.num_examples

        self.n_hidden = 500
        self.n_z = 20
        # self.batchsize = 100
        self.batchsize = tf.placeholder(tf.int32)

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        self.guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(self.guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            w_mean = lrelu(dense(h2_flat, 7*7*32, self.n_z, "w_mean"))
            w_stddev = lrelu(dense(h2_flat, 7*7*32, self.n_z, "w_stddev"))

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        batchsize = 100       
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            for epoch in range(501):
                for idx in range(int(self.n_samples / batchsize)):
                    batch = self.mnist.unbalance01.next_batch(batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch, self.batchsize: batchsize})
                    # print("epoch {}: genloss {} latloss{}".format(epoch, np.mean(gen_loss), np.mean(lat_loss)))
                    if idx == range(int(self.n_samples / batchsize))[-1] and epoch % 50 == 0:
                        end_time = time.time() 
                        print("Training epoch {}: --- {} seconds --- genloss {} latloss{}".format(epoch,end_time-start_time, np.mean(gen_loss), np.mean(lat_loss)))
                        
                        # plt.scatter(latent_var[class0_idx,0],latent_var[class0_idx,1],c='red', s=1)
                        # plt.scatter(latent_var[class1_idx,0],latent_var[class1_idx,1],c='blue', s=1)
                        # plt.savefig('latent_plots/unbalance01_epoch{}_{}.png'.format(epoch, idx))
                        # plt.close()

                        # ratio: 10:1
                        num_random_samples = 900
                        im_w, im_h = 30, 30
                        r = np.random.RandomState(1234)
                        randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)], dtype = np.float32)
                        random_samples = np.zeros((1,28,28))
                        for i in range(num_random_samples//100):                            
                            generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100], self.batchsize: batchsize})
                            generated_test = generated_test.reshape(batchsize,28,28)
                            random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        random_samples = random_samples[1:]
                        if epoch == 500:
                            ims("results/random01_epoch500_10to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # if epoch == 500:
                        #     st()
                        #     for i, image in enumerate(random_samples):
                        #         ims('results_random_samples/random01_epoch500_10to1/temp{}.jpg'.format(i), image)


                        # # ratio: 100:1
                        # num_random_samples = 10000
                        # im_w, im_h = 100, 100
                        # r = np.random.RandomState(1234)
                        # randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)], dtype = np.float32)
                        # random_samples = np.zeros((1,28,28))
                        # for i in range(num_random_samples//100):                            
                        #     generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100], self.batchsize: batchsize})
                        #     generated_test = generated_test.reshape(batchsize,28,28)
                        #     random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        # random_samples = random_samples[1:]
                        # if epoch == 500:
                        #     ims("results/random01_epoch500_100to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # # if epoch == 500:
                        # #     for i, image in enumerate(random_samples):
                        # #         ims('results_random_samples/random_epoch500_100to1/temp{}.jpg'.format(i), image)

                        # ratio: 1000:1
                        # num_random_samples = 96100
                        # im_w, im_h = 310, 310
                        # r = np.random.RandomState(1234)
                        # randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)], dtype = np.float32)
                        # random_samples = np.zeros((1,28,28))
                        # for i in range(num_random_samples//100):
                        #     st()                            
                        #     generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100], self.batchsize: batchsize})
                        #     generated_test = generated_test.reshape(batchsize,28,28)
                        #     random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        # random_samples = random_samples[1:]
                        # if epoch == 500:
                        #     ims("results/random01_epoch500_1000to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # # if epoch == 500:
                        # #     for i, image in enumerate(random_samples):
                        # #         ims('results_random_samples/random_epoch500_100to1/temp{}.jpg'.format(i), image)


                        start_time = time.time()

model = LatentAttention()
model.train()
