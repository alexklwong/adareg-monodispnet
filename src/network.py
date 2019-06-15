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

def conv2d(x, ksize, stride, act_fn=tf.nn.elu):
  p = np.floor((ksize[0]-1)/2).astype(np.int32)
  p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
  return slim.conv2d(p_x, ksize[2], ksize[0:2], stride,
                     padding='VALID', activation_fn=act_fn)

def conv2d_block_vgg(x, ksize):
  conv1 = conv2d(x,     ksize=ksize, stride=1)
  conv2 = conv2d(conv1, ksize=ksize, stride=2)
  return conv2

def max_pool(x, ksize):
  p = np.floor((ksize[0]-1)/2).astype(np.int32)
  p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
  return slim.max_pool2d(p_x, ksize)

def upsample2d(x, ratio):
  shape = tf.shape(x)
  h = shape[1]
  w = shape[2]
  return tf.image.resize_nearest_neighbor(x, [h*ratio, w*ratio])

def upconv2d(x, ksize, scale):
  upsample = upsample2d(x, scale)
  conv = conv2d(upsample, ksize, 1)
  return conv

def get_disp(x, ksize=[3, 3, 2], max_disparity=0.30):
  return max_disparity*conv2d(x, ksize=ksize, stride=1, act_fn=tf.nn.sigmoid)

def build_vgg(data):
  with tf.variable_scope('encoder'):
    conv0 = conv2d(data, ksize=[7, 7, 32], stride=1)
    conv1 = conv2d_block_vgg(conv0, ksize=[7, 7, 32]) # H/2
    conv2 = conv2d_block_vgg(conv1, ksize=[5, 5, 64]) # H/4
    conv3 = conv2d_block_vgg(conv2, ksize=[3, 3, 128]) # H/8
    conv4 = conv2d_block_vgg(conv3, ksize=[3, 3, 256]) # H/16
    conv5 = conv2d_block_vgg(conv4, ksize=[3, 3, 512]) # H/32
    conv6 = conv2d_block_vgg(conv5, ksize=[3, 3, 512]) # H/64

  with tf.variable_scope('skips'):
    skip0 = conv0
    skip1 = conv1
    skip2 = conv2
    skip3 = conv3
    skip4 = conv4
    skip5 = conv5
    skip6 = conv6

  with tf.variable_scope('decoder'):
    upconv6 = upconv2d(conv6, ksize=[3, 3, 512], scale=2) #H/32
    concat6 = tf.concat([upconv6, skip5], 3)
    iconv6  = conv2d(concat6,  ksize=[3, 3, 512], stride=1)

    upconv5 = upconv2d(iconv6, ksize=[3, 3, 256], scale=2) #H/16
    concat5 = tf.concat([upconv5, skip4], 3)
    iconv5  = conv2d(concat5,  ksize=[3, 3, 256], stride=1)

    # Up convolution for initial features, and residual features
    iupconv4  = upconv2d(iconv5, ksize=[3, 3, 128], scale=2) #H/8
    # Concatenate initial prediction features
    iconcat4  = tf.concat([iupconv4, skip3], axis=3)
    iconv4    = conv2d(iconcat4, ksize=[3, 3, 128], stride=1)
    init4     = get_disp(iconv4)
    # Learn the residual features needed from skip connection
    sconv4    = conv2d(skip3, ksize=[3, 3, 128], stride=1)
    sconv4    = conv2d(sconv4, ksize=[3, 3, 128], stride=1, act_fn=tf.identity)
    sconv4    = tf.nn.elu(skip3+sconv4)
    rconcat4  = tf.concat([iconv4, init4, sconv4], axis=3)
    rconv4    = conv2d(rconcat4, ksize=[3, 3, 128], stride=1)
    disp4     = get_disp(rconv4)
    # Upsample predictions
    uinit4    = upsample2d(init4, ratio=2)
    udisp4    = upsample2d(disp4, ratio=2)

    # Up convolution for initial features, and residual features
    iupconv3  = upconv2d(iconv4, ksize=[3, 3, 64], scale=2) #H/4
    rupconv3  = upconv2d(rconv4, ksize=[3, 3, 64], scale=2)
    # Concatenate initial prediction features
    iconcat3  = tf.concat([iupconv3, skip2, uinit4], axis=3)
    iconv3    = conv2d(iconcat3, ksize=[3, 3, 64], stride=1)
    init3     = get_disp(iconv3)
    # Concatenate residual estimation features
    sconv3    = conv2d(skip2, ksize=[3, 3, 64], stride=1)
    sconv3    = conv2d(sconv3, ksize=[3, 3, 64], stride=1, act_fn=tf.identity)
    sconv3    = tf.nn.elu(skip2+sconv3)
    rconcat3  = tf.concat([iconv3, init3, sconv3, rupconv3, udisp4], axis=3)
    rconv3    = conv2d(rconcat3, ksize=[3, 3, 64], stride=1)
    disp3     = get_disp(rconv3)
    # Upsample predictions
    uinit3    = upsample2d(init3, ratio=2)
    udisp3    = upsample2d(disp3, ratio=2)

    # Up convolution for initial features, and residual features
    iupconv2  = upconv2d(iconv3, ksize=[3, 3, 32], scale=2) #H/2
    rupconv2  = upconv2d(rconv3, ksize=[3, 3, 32], scale=2)
    # Concatenate initial prediction features
    iconcat2  = tf.concat([iupconv2, skip1, uinit3], axis=3)
    iconv2    = conv2d(iconcat2, ksize=[3, 3, 32], stride=1)
    init2     = get_disp(iconv2)
    # Concatenate residual estimation features
    sconv2    = conv2d(skip1, ksize=[3, 3, 32], stride=1)
    sconv2    = conv2d(sconv2, ksize=[3, 3, 32], stride=1, act_fn=tf.identity)
    rconcat2  = tf.concat([iconv2, init2, sconv2, rupconv2, udisp3], axis=3)
    rconv2    = conv2d(rconcat2, ksize=[3, 3, 32], stride=1)
    disp2     = get_disp(rconv2)
    # Upsample predictions
    uinit2    = upsample2d(init2, ratio=2)
    udisp2    = upsample2d(disp2, ratio=2)

    # Up convolution for initial features, and residual features
    iupconv1  = upconv2d(iconv2, ksize=[3, 3, 16], scale=2) #H
    rupconv1  = upconv2d(rconv2, ksize=[3, 3, 16], scale=2)
    # Concatenate initial prediction features
    iconcat1  = tf.concat([iupconv1, skip0, uinit2], axis=3)
    iconv1    = conv2d(iconcat1, ksize=[3, 3, 16], stride=1)
    init1     = get_disp(iconv1)
    # Concatenate residual prediction features
    sconv1    = conv2d(skip0, ksize=[3, 3, 32], stride=1)
    sconv1    = conv2d(sconv1, ksize=[3, 3, 32], stride=1, act_fn=tf.identity)
    sconv1    = tf.nn.elu(skip0+sconv1)
    rconcat1  = tf.concat([iconv1, init1, sconv1, rupconv1, udisp2], axis=3)
    rconv1    = conv2d(rconcat1, ksize=[3, 3, 16], stride=1)
    disp1     = get_disp(rconv1, ksize=[5, 5, 2])

  return [disp1, disp2, disp3, disp4], [init1, init2, init3, init4]
