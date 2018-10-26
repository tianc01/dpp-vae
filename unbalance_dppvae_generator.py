import tensorflow as tf
import numpy as np
import input_data_ratio10to1
import matplotlib
matplotlib.use("TAgg")
import matplotlib.pyplot as plt
import os
import math
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

        # Set parameters
        self.n_hidden = 500
        self.n_z = 20
        self.sigma2_LX = 1e-35
        # self.sigma2_LX = 0.0
        self.alpha, self.rho, self.sigma = 1000.0, 1.0, 1.0

        self.log_norm_constant = tf.placeholder(tf.float64)
        self.batch_size = tf.placeholder(tf.int32, name = 'batch_size')

        self.images = tf.placeholder(tf.float64, [None, 784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])
        self.z_mean, self.z_logsigma2 = self.recognition(tf.cast(image_matrix, tf.float32))

        self.z_mean = tf.cast(self.z_mean, tf.float64)
        self.z_logsigma2 = tf.cast(self.z_logsigma2, tf.float64)

        print('Sampling latent variable...')
        samples = tf.random_normal([self.batch_size,self.n_z],0,1,dtype=tf.float64)
        self.guessed_z = tf.identity(self.z_mean + tf.exp(0.5 * self.z_logsigma2) * samples, name = 'guessed_z')

        print('Calculate reconstructed x...')
        self.generated_images = self.generation(tf.cast(self.guessed_z, tf.float32))
        self.generated_images = tf.cast(self.generated_images, tf.float64)
        generated_flat = tf.reshape(self.generated_images, [self.batch_size, 28*28])

        print('Calculate generation loss...')
        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)


        print('Calculate KL divergence loss...')
        self.mat_LX = self.get_mat_LX(self.guessed_z)

        self.logdet_mat_LZ = tf.linalg.logdet(self.mat_LX)
        self.latent_loss = tf.reduce_sum(0.5 * tf.reduce_sum(- self.z_logsigma2 - 1, reduction_indices = 1)) - self.logdet_mat_LZ + self.log_norm_constant
        self.latent_loss = self.latent_loss/tf.cast(self.batch_size, dtype=tf.float64)
        
        # self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.cost = tf.reduce_mean(self.generation_loss) + self.latent_loss
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def get_mat_LX(self, X_latent):
        '''
        INPUT
        ---------
        X_latent: np.matrix(dim_N, dim_P)
        '''
        dim_P = int(X_latent.shape[1])
        col = tf.reduce_sum(X_latent*X_latent, 1)
        col = tf.reshape(col, [-1, 1])
        prod = tf.matmul(X_latent, tf.transpose(X_latent))

        log_mat_LX = - (0.5/self.rho+0.5/self.sigma)*col + 1/self.sigma*prod - (0.5/self.rho+0.5/self.sigma)*tf.transpose(col) + 2*math.log(self.alpha**0.5)-2*dim_P*math.log((math.pi*self.rho)**0.5)
        mat_LX = tf.exp(log_mat_LX) + self.sigma2_LX*tf.diag(tf.exp(tf.reduce_sum(X_latent-X_latent, 1)))

        return mat_LX

    # encoder
    def recognition(self, input_images):
        with tf.variable_scope("recognition"):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            h2_flat = tf.reshape(h2,[self.batch_size, 7*7*32])

            # w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            # w_logsigma2 = dense(h2_flat, 7*7*32, self.n_z, "w_logsigma2")

            w_mean = lrelu(dense(h2_flat, 7*7*32, self.n_z, "w_mean"))
            w_logsigma2 = lrelu(dense(h2_flat, 7*7*32, self.n_z, "w_logsigma2"))

        return w_mean, w_logsigma2

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batch_size, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    def elem_sym_poly(self, dim_kDPP, lam_list):
        '''
        Computing the elementary symmetric polynomials

        lam_list: list of eigenvalues
        dim_kDPP: dimension of kDPP
        '''

        # Initialization
        N = len(lam_list)
        mat_elem = np.empty([N+1,dim_kDPP+1])
        for n in range(N+1):
            mat_elem[n, 0] = 1
        for l in range(1, dim_kDPP+1):
            mat_elem[0,l] = 0

        for l in range(1, dim_kDPP+1):
            for n in range(1, N+1):
                mat_elem[n,l] = mat_elem[n-1,l] + lam_list[n-1]*mat_elem[n-1,l-1]
        return mat_elem[N,dim_kDPP]

    def elem_sym_poly_upper(self, lam_list, dim_kDPP):
        trace_L = self.alpha

        result = 0
        for j in range(dim_kDPP+1):
            result += (trace_L - sum(lam_list))**j/math.factorial(j)*self.elem_sym_poly(dim_kDPP-j, lam_list)
        return result

    def get_lamlist(self, approx_dim):
        # Get eigenvalues lambda1, ..., lambdaM
        gamma = self.sigma/self.rho
        beta = (1+2/gamma)**0.25

        lam_list = []
        for i in range(approx_dim):
          n = i+1
          factor = ((0.5*(beta**2+1)+0.5/gamma)**(-0.5)) * ((gamma*(beta**2+1)+1)**(1-n))
          lam_list.append(self.alpha*(factor**(self.n_z)))
        print('Lambda: {}'.format(lam_list))
        # lam_list = tf.reshape(tf.stack(lam_list), [approx_dim])

        return lam_list

    def train(self):
        batch_size = 100       
        # train
        # saver = tf.train.Saver(max_to_keep=2)

        # get eigenvalues lambda and normalization term
        lam_list = self.get_lamlist(approx_dim = 10)
        norm_constant = self.elem_sym_poly_upper(lam_list, dim_kDPP=batch_size)
        log_norm_constant = math.log(norm_constant)
        # log_norm_constant = 1.0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Initial performance
            start_time = time.time()
            for epoch in range(501):
                for idx in range(int(self.n_samples / batch_size)):
                    # batch, batch_y = self.mnist.unbalance01.next_batch(batch_size)
                    batch  = self.mnist.unbalance01.next_batch(batch_size)[0]
                    _,  loss, gen_loss, lat_loss = sess.run((self.optimizer, self.cost, self.generation_loss, self.latent_loss), feed_dict={
                        self.images: batch, 
                        self.batch_size: batch_size,
                        self.log_norm_constant: log_norm_constant})

                    if idx == range(int(self.n_samples / batch_size))[-1] and epoch % 50 == 0:
                        end_time = time.time()
                        print("Training epoch {}: --- {} seconds --- loss {} genloss {} latloss {}".format(epoch,end_time-start_time, loss, np.mean(gen_loss), np.mean(lat_loss)))
                        
                        # ratio: 10:1
                        num_random_samples = 900
                        im_w, im_h = 30, 30
                        r = np.random.RandomState(1234)
                        randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)])
                        random_samples = np.zeros((1,28,28))
                        for i in range(num_random_samples//100):
                            generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100],  self.batch_size: batch_size})
                            generated_test = generated_test.reshape(batch_size,28,28)
                            random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        random_samples = random_samples[1:]
                        if epoch == 500:
                            ims("results/random01_dpp_epoch500_10to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # if epoch == 500:
                        #     for i, image in enumerate(random_samples):
                        #         ims('results_random_samples/random01_dpp_epoch500_10to1/temp{}.jpg'.format(i), image)

                        # # ratio: 100:1
                        # num_random_samples = 10000
                        # im_w, im_h = 100, 100
                        # r = np.random.RandomState(1234)
                        # randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)])
                        # random_samples = np.zeros((1,28,28))
                        # for i in range(num_random_samples//100):
                        #     generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100], self.batch_size: batch_size})
                        #     generated_test = generated_test.reshape(batch_size,28,28)
                        #     random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        # random_samples = random_samples[1:]
                        # if epoch == 500:
                        #     ims("results_random_samples/random01_dpp_epoch500_100to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # # if epoch == 500:
                        # #     for i, image in enumerate(random_samples):
                        # #         ims('results/random57_dpp_epoch500_100to1/temp{}.jpg'.format(i), image)

                        # # ratio: 1000:1
                        # num_random_samples = 96100
                        # im_w, im_h = 310, 310
                        # r = np.random.RandomState(1234)
                        # randoms = np.array([r.normal(0, 1, 20) for _ in range(num_random_samples)])
                        # random_samples = np.zeros((1,28,28))
                        # for i in range(num_random_samples//100):
                        #     generated_test = sess.run(self.generated_images, feed_dict={self.guessed_z: randoms[i*100:(i+1)*100], self.batch_size: batch_size})
                        #     generated_test = generated_test.reshape(batch_size,28,28)
                        #     random_samples = np.concatenate((random_samples, generated_test), axis=0)
                        # random_samples = random_samples[1:]
                        # if epoch == 500:
                        #     ims("results_random_samples/random01_dpp_epoch500_1000to1/all_images.jpg",merge(random_samples,[im_w, im_h]))
                        # # if epoch == 500:
                        # #     for i, image in enumerate(random_samples):
                        # #         ims('results/random57_dpp_epoch500_100to1/temp{}.jpg'.format(i), image)

                        start_time = time.time()
                        
model = LatentAttention()
model.train()
