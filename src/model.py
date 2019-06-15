'''
Author: Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, B. W. Hong and S. Soatto. Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.
https://arxiv.org/abs/1903.07309

@article{wong2019bilateral,
  title={Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction},
  author={Wong, Alex and Hong, Byung-Woo and Soatto, Stefano},
  journal={arXiv preprint arXiv:1903.07309},
  year={2019}
}
'''

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *

import network as net
import losses

class MonoDispNet(object):

  def __init__(self, left, right, n_pyramid=4, max_disparity=0.30,
               w_ph=0.15, w_st=0.85, w_sm=0.10, w_bc=1.05, w_ar=5.0,
               reuse_variables=None, model_index=0):
    self.left = left
    self.right = right
    self.n_pyramid = n_pyramid
    self.max_disparity = max_disparity
    self.model_collection = ['model_' + str(model_index)]
    self.w_ph = w_ph
    self.w_st = w_st
    self.w_sm = w_sm
    self.w_bc = w_bc
    self.w_ar = w_ar

    self.reuse_variables = reuse_variables

    self.build_model()
    self.build_outputs()

    self.build_losses()
    self.build_summaries()

  def scale_pyramid(self, img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
      ratio = 2 ** (i + 1)
      nh = h // ratio
      nw = w // ratio
      scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs

  def build_model(self):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
      with tf.variable_scope('model', reuse=self.reuse_variables):

        self.data0  = self.scale_pyramid(self.left,  self.n_pyramid)
        self.data1 = self.scale_pyramid(self.right, self.n_pyramid)

        self.model_input = self.left
        self.disp_est, self.init_est = net.build_vgg(self.model_input)
        self.init_est = self.init_est[0:self.n_pyramid]
        self.disp_est = self.disp_est[0:self.n_pyramid]

  def build_outputs(self):
    # STORE DISPARITIES
    with tf.variable_scope('disparities'):
      self.model0i = [tf.expand_dims(d[:,:,:,0], 3) for d in self.init_est]
      self.model1i = [tf.expand_dims(d[:,:,:,1], 3) for d in self.init_est]
      self.model0 = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
      self.model1 = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

    # GENERATE IMAGES
    with tf.variable_scope('images'):
      self.data0iw = [bilinear_sampler_1d_h(self.data1[i], -self.model0i[i]) for i in range(len(self.model0i))]
      self.data1iw = [bilinear_sampler_1d_h(self.data0[i], self.model1i[i]) for i in range(len(self.model0i))]
      self.data0w = [bilinear_sampler_1d_h(self.data1[i], -self.model0[i]) for i in range(self.n_pyramid)]
      self.data1w = [bilinear_sampler_1d_h(self.data0[i], self.model1[i]) for i in range(self.n_pyramid)]

    # BILATERAL CYCLIC CONSISTENCY
    with tf.variable_scope('left-right'):
      self.model0w = []
      self.model1w = []
      for i in range(self.n_pyramid):
        # First build model1w then build model0w
        self.model0w.append(bilinear_sampler_1d_h(bilinear_sampler_1d_h(self.model0[i], self.model1[i]),
                                                  -self.model0[i]))
        # First build model0w then build model1w
        self.model1w.append(bilinear_sampler_1d_h(bilinear_sampler_1d_h(self.model1[i], -self.model0[i]),
                                                  self.model1[i]))

  def build_losses(self):
    with tf.variable_scope('losses', reuse=self.reuse_variables):
      self.l_ph0 = []
      self.l_ph1 = []
      self.l_st0 = []
      self.l_st1 = []
      self.l_sm0 = []
      self.l_sm1 = []
      self.l_bc0 = []
      self.l_bc1 = []
      self.w_res0 = []
      self.w_res1 = []
      for i in range(self.n_pyramid):
        # PHOTOMETRIC IMAGE RECONSTRUCTION
        self.l_ph0.append(losses.photometric_loss_func(self.data0iw[i], self.data0[i]))
        self.l_ph1.append(losses.photometric_loss_func(self.data1iw[i], self.data1[i]))
        self.l_ph0.append(losses.photometric_loss_func(self.data0w[i], self.data0[i]))
        self.l_ph1.append(losses.photometric_loss_func(self.data1w[i], self.data1[i]))
        # STRUCTURAL SSIM
        self.l_st0.append(losses.structural_loss_func(self.data0iw[i], self.data0[i]))
        self.l_st1.append(losses.structural_loss_func(self.data1iw[i], self.data1[i]))
        self.l_st0.append(losses.structural_loss_func(self.data0w[i], self.data0[i]))
        self.l_st1.append(losses.structural_loss_func(self.data1w[i], self.data1[i]))
        # Weights for regularization
        self.w_res0.append(losses.residual_regularity_weight_func(self.data0[i], self.data0w[i], self.w_ar))
        self.w_res1.append(losses.residual_regularity_weight_func(self.data1[i], self.data1w[i], self.w_ar))
        # DISPARITY SMOOTHNESS
        self.l_sm0.append(losses.smoothness_loss_func(self.model0[i], self.data0[i], w=self.w_res0[i]))
        self.l_sm1.append(losses.smoothness_loss_func(self.model1[i], self.data1[i], w=self.w_res1[i]))
        self.l_sm0[i] = self.l_sm0[i]/2**i
        self.l_sm1[i] = self.l_sm1[i]/2**i
        # LR CONSISTENCY
        self.l_bc0.append(losses.bc_consistency_loss_func(self.model0w[i], self.model0[i], self.w_res0[i]))
        self.l_bc1.append(losses.bc_consistency_loss_func(self.model1w[i], self.model1[i], self.w_res1[i]))
      l_ph = tf.add_n(self.l_ph0+self.l_ph1)
      l_st = tf.add_n(self.l_st0+self.l_st1)
      l_sm = tf.add_n(self.l_sm0+self.l_sm1)
      l_bc = tf.add_n(self.l_bc0+self.l_bc1)
      # TOTAL LOSS
      self.total_loss = self.w_ph*l_ph+self.w_st*l_st+self.w_sm*l_sm+self.w_bc*l_bc

  def build_summaries(self):
    with tf.device('/cpu:0'):
      for i in range(self.n_pyramid):
        tf.summary.scalar('photometric_loss_res'+str(i), self.l_ph0[i]+self.l_ph1[i], collections=self.model_collection)
        tf.summary.scalar('structural_loss_res'+str(i), self.l_st0[i]+self.l_st1[i], collections=self.model_collection)
        tf.summary.scalar('smoothness_loss_res'+str(i), self.l_sm0[i]+self.l_sm1[i], collections=self.model_collection)
        tf.summary.scalar('bc_consistency_loss_res'+str(i), self.l_bc0[i]+self.l_bc1[i], collections=self.model_collection)
        tf.summary.scalar('res_weight0_res'+str(i), tf.reduce_mean(self.w_res0[i]), collections=self.model_collection)
        tf.summary.scalar('res_weight1_res'+str(i), tf.reduce_mean(self.w_res1[i]), collections=self.model_collection)
        tf.summary.histogram('res_weight0_hist_res'+str(i), self.w_res0[i], collections=self.model_collection)
        tf.summary.histogram('res_weight1_hist_res'+str(i), self.w_res1[i], collections=self.model_collection)
        # Record images of output
        tf.summary.image('model0_res'+str(i), self.model0[i], max_outputs=4, collections=self.model_collection)
        tf.summary.image('model1_res'+str(i), self.model1[i], max_outputs=4, collections=self.model_collection)
        tf.summary.image('model0i_res'+str(i), self.model0i[i], max_outputs=4, collections=self.model_collection)
        tf.summary.image('model1i_res'+str(i), self.model1i[i], max_outputs=4, collections=self.model_collection)


