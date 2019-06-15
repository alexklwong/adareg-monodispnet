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

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def photometric_loss_func(src, tgt):
  return tf.reduce_mean(tf.abs(src-tgt))

def structural_loss_func(src, tgt):
  return tf.reduce_mean(ssim(src, tgt))

def bc_consistency_loss_func(src, tgt, w):
  return tf.reduce_mean(w*tf.abs(src-tgt))

def ssim(A, B):
  C1 = 0.01**2
  C2 = 0.03**2
  mu_A = slim.avg_pool2d(A, 3, 1, 'VALID')
  mu_B = slim.avg_pool2d(B, 3, 1, 'VALID')
  sigma_A  = slim.avg_pool2d(A**2, 3, 1, 'VALID')-mu_A**2
  sigma_B  = slim.avg_pool2d(B**2, 3, 1, 'VALID')-mu_B**2
  sigma_AB = slim.avg_pool2d(A*B , 3, 1, 'VALID')-mu_A*mu_B
  numer = (2*mu_A*mu_B+C1)*(2*sigma_AB+C2)
  denom = (mu_A**2+mu_B**2+C1)*(sigma_A+sigma_B+C2)
  score = numer/denom
  return tf.clip_by_value((1-score)/2, 0, 1)

def smoothness_loss_func(model, data, w):
  disp_gradients_y, disp_gradients_x = gradient_yx(model)
  laplacian = im_laplacian(tf.image.rgb_to_grayscale(data), ksize=5, channels=1)
  l = tf.exp(tf.multiply(tf.abs(laplacian), -1))
  weights_x = l[:, :, 1:, :]
  weights_y = l[:, 1:, :, :]
  smoothness_x = tf.reduce_mean(tf.abs(disp_gradients_x)*weights_x)
  smoothness_y = tf.reduce_mean(tf.abs(disp_gradients_y)*weights_y)
  return smoothness_x+smoothness_y

def residual_regularity_weight_func(src, tgt, w):
  local_res = tf.reduce_sum(tf.abs(src-tgt), [3], keepdims=True)
  shape = tf.shape(local_res)
  global_res = tf.reduce_mean(tf.reduce_sum(tf.abs(src-tgt), [3]), [1, 2])
  global_res = tf.reshape(global_res, [tf.shape(global_res)[0], 1, 1, 1])
  global_res = tf.tile(global_res, [1, shape[1], shape[2], 1])
  weights = tf.exp(-1.0*w*tf.multiply(local_res, global_res))
  return weights

def im_laplacian(T, ksize=3, channels=3, use_gaussian=True):
  T = tf.cast(T, tf.float32)
  laplacian_kernel = get_laplacian2d(ksize, channels, channels)
  gaussian_kernel = get_gaussian2d(ksize, channels, channels)
  n_pad = ksize//2
  T_pad = tf.pad(T, [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]], 'SYMMETRIC')
  if use_gaussian:
    T_pad = tf.nn.conv2d(T_pad, gaussian_kernel, strides=[1, 1, 1, 1], padding='VALID')
    T_pad = tf.pad(T_pad, [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]], 'SYMMETRIC')
  T_laplacian = tf.nn.conv2d(T_pad, laplacian_kernel, strides=[1, 1, 1, 1], padding='VALID')
  return T_laplacian

def gradient_yx(T):
  gx = T[:,:,:-1,:]-T[:,:,1:,:]
  gy = T[:,:-1,:,:]-T[:,1:,:,:]
  return gy, gx

def get_laplacian2d(ksize=3, in_channels=1, out_channels=1, name='laplacian2d'):
  center = int(np.ceil(ksize/2.0-1))
  laplacian = -1*np.ones([ksize, ksize])
  laplacian[center, center] = ksize**2-1
  laplacian_kernel = np.reshape(laplacian, [ksize, ksize, 1, 1])
  laplacian_kernel = np.tile(laplacian_kernel, [1, 1, in_channels, out_channels])
  return tf.constant(value=laplacian_kernel, dtype=tf.float32)

def get_gaussian2d(ksize=3, in_channels=1, out_channels=1, name='gaussian2d'):
  if ksize == 5:
    gaussian = [ [1.0, 4.0,  7.0,  4.0,  1.0],
                 [4.0, 16.0, 26.0, 16.0, 4.0],
                 [7.0, 26.0, 41.0, 26.0, 7.0],
                 [4.0, 16.0, 26.0, 16.0, 4.0],
                 [1.0, 4.0,  7.0,  4.0,  1.0] ]
    gaussian = np.asarray(gaussian, dtype=np.float32)/273.0
  else:
    ksize = 3
    gaussian = [ [1.0, 2.0, 1.0],
                 [2.0, 4.0, 2.0],
                 [1.0, 2.0, 1.0] ]
    gaussian = np.asarray(gaussian, dtype=np.float32)/16.0
  gaussian_kernel = np.reshape(gaussian, [ksize, ksize, 1, 1])
  gaussian_kernel = np.tile(gaussian_kernel, [1, 1, in_channels, out_channels])
  return tf.constant(value=gaussian_kernel, dtype=tf.float32)
