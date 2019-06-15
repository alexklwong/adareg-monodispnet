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

import os, sys, re, time
import numpy as np
import tensorflow as tf
import cv2

import global_constants as settings
from average_gradients import *
from dataloader import DataLoader
from model import MonoDispNet
import data_utils
from data_utils import log
import eval_utils

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

def train(trn_im0_path, trn_im1_path,
          n_epoch=settings.N_EPOCH, n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT, n_width=settings.N_WIDTH, n_channel=settings.N_CHANNEL,
          learning_rates=settings.LEARNING_RATES, learning_bounds=settings.LEARNING_BOUNDS,
          n_pyramid=settings.N_PYRAMID, max_disparity=settings.MAX_DISPARITY,
          w_ph=settings.W_PH, w_st=settings.W_ST, w_sm=settings.W_SM, w_bc=settings.W_BC, w_ar=settings.W_AR,
          n_checkpoint=settings.N_CHECKPOINT, n_summary=settings.N_SUMMARY,
          checkpoint_path=settings.CHECKPOINT_PATH, restore_path=settings.RESTORE_PATH,
          n_gpu=settings.N_GPU, n_thread=settings.N_THREAD):

  event_path = os.path.join(checkpoint_path, 'event')
  model_path = os.path.join(checkpoint_path, 'model.ckpt')
  log_path = os.path.join(checkpoint_path, 'results.txt')

  # Load image paths from paths file for training and validation
  trn_im0_paths = data_utils.read_paths(trn_im0_path)
  trn_im1_paths = data_utils.read_paths(trn_im1_path)
  n_trn_sample = len(trn_im0_paths)
  n_trn_step =  n_epoch*np.ceil(n_trn_sample/n_batch).astype(np.int32)

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    # Initialize optimizer
    boundaries = [np.int32(b*n_trn_step) for b in learning_bounds]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Initialize dataloader
    dataloader = DataLoader(shape=[n_batch, n_height, n_width, n_channel],
                            name='dataloader', n_thread=n_thread, prefetch_size=8, normalize=True,
                            random_flip=True, random_gamma=True, gamma_range=[0.8, 1.2],
                            random_brightness=True, brightness_range=[0.5, 2.0],
                            random_color=True, color_range=[0.8, 1.2])
    # Split data into towers for each GPU
    im0_split = tf.split(dataloader.next_element[0], n_gpu, 0)
    im1_split = tf.split(dataloader.next_element[1], n_gpu, 0)
    # Build computation graph
    tower_gradients  = []
    tower_losses = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(n_gpu):
        with tf.device('/gpu:%d' % i):
          params = []
          model = MonoDispNet(im0_split[i], im1_split[i], n_pyramid=n_pyramid,
                              w_ph=w_ph, w_st=w_st, w_sm=w_sm, w_bc=w_bc, w_ar=w_ar,
                              max_disparity=max_disparity, reuse_variables=tf.AUTO_REUSE,
                              model_index=i)
          loss = model.total_loss
          tower_losses.append(loss)
          tower_gradients.append(optimizer.compute_gradients(loss))
    # Set up gradient computations
    avg_gradients = average_gradients(tower_gradients)
    gradients = optimizer.apply_gradients(avg_gradients, global_step=global_step)

    total_loss = tf.reduce_mean(tower_losses)

    tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
    tf.summary.scalar('total_loss', total_loss, ['model_0'])
    trn_summary = tf.summary.merge_all('model_0')

    # Count trainable parameters
    n_parameter = 0
    for variable in tf.trainable_variables():
      n_parameter += np.array(variable.get_shape().as_list()).prod()
    # Log network parameters
    log('Network Parameters:', log_path)
    log('n_batch=%d  n_height=%d  n_width=%d  n_channel=%d  ' %
        (n_batch, n_height, n_width, n_channel), log_path)
    log('n_pyramid=%d  max_disparity=%.3f' %
        (n_pyramid, max_disparity), log_path)
    log('n_sample=%d  n_epoch=%d  n_step=%d  n_param=%d' %
        (n_trn_sample, n_epoch, n_trn_step, n_parameter), log_path)
    log('learning_rates=[%s]' %
        ', '.join('{:.6f}'.format(r) for r in learning_rates), log_path)
    log('boundaries=[%s]' %
        ', '.join('{:.2f}:{}'.format(l, b) for l, b in zip(learning_bounds, boundaries)), log_path)
    log('w_ph=%.3f  w_st=%.3f  w_sm=%.3f  w_bc=%.3f  w_ar=%.3f' %
        (w_ph, w_st, w_sm, w_bc, w_ar), log_path)
    log('Restoring from: %s' % ('None' if restore_path == '' else restore_path),
        log_path)

    # Initialize Tensorflow session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    # Initialize saver for storing and restoring checkpoints
    summary_writer = tf.summary.FileWriter(model_path, session.graph)
    train_saver = tf.train.Saver()
    # Initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    # If given, load the weights from the restore path
    if restore_path != '':
      train_saver.restore(session, restore_path)

    # Begin training
    log('Begin training...')
    start_step = global_step.eval(session=session)
    time_start = time.time()
    trn_step = start_step
    while trn_step < n_trn_step:
      trn_im0_paths_epoch, trn_im1_paths_epoch = data_utils.shuffle_and_drop([trn_im0_paths, trn_im1_paths], n_batch)
      dataloader.initialize(session,
                            im0_paths=trn_im0_paths_epoch,
                            im1_paths=trn_im1_paths_epoch,
                            augment=True)
      while trn_step < n_trn_step:
        try:
          _, loss_value = session.run([gradients, total_loss])
          if trn_step % n_summary == 0:
            summary = session.run(trn_summary)
            summary_writer.add_summary(summary, global_step=trn_step)
          if trn_step and trn_step % n_checkpoint == 0:
            time_elapse = (time.time()-time_start)/3600*trn_step/(trn_step-start_step+1)
            time_remain = (n_trn_step/trn_step-1.0)*time_elapse

            checkpoint_log = 'batch {:>6}  loss: {:.5f}  time elapsed: {:.2f}h  time left: {:.2f}h'
            log(checkpoint_log.format(trn_step, loss_value, time_elapse, time_remain), log_path)
            train_saver.save(session, model_path, global_step=trn_step)
          trn_step += 1
        except tf.errors.OutOfRangeError:
          break

    train_saver.save(session, model_path, global_step=n_trn_step)

def evaluate(im0_path, restore_path, output_path,
             n_batch=settings.N_BATCH, n_height=settings.N_HEIGHT, n_width=settings.N_WIDTH, n_channel=settings.N_CHANNEL,
             n_pyramid=settings.N_PYRAMID, max_disparity=settings.MAX_DISPARITY,
             n_gpu=settings.N_GPU, n_thread=settings.N_THREAD):
  """Test function."""
  # Create dataloader for computation graph
  dataloader = DataLoader(shape=[n_batch, n_height, n_width, n_channel],
                          name='dataloader', n_thread=n_thread, prefetch_size=n_thread, normalize=True,
                          random_flip=False, random_gamma=False, gamma_range=[0.8, 1.2],
                          random_brightness=False, brightness_range=[0.5, 2.0],
                          random_color=False, color_range=[0.8, 1.2])
  # Build model
  model = MonoDispNet(dataloader.next_element[0], dataloader.next_element[1],
                      n_pyramid=n_pyramid, max_disparity=max_disparity)
  # Start a Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  # Initialize saver that will be used for restore
  train_saver = tf.train.Saver()
  # Initialize all variables
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())
  # Restore weights from checkpoint
  log('Restoring from: %s' % restore_path)
  train_saver.restore(session, restore_path)
  # Load the files for evaluation
  im0_paths = data_utils.read_paths(im0_path)
  n_sample = len(im0_paths)
  im0_paths = data_utils.pad_batch(im0_paths, n_batch)
  n_step = len(im0_paths)//n_batch
  log('Evaluating %d files...' % n_sample)
  dataloader.initialize(session,im0_paths=im0_paths, im1_paths=im0_paths,
                        augment=False)

  d_arr = np.zeros((n_step*n_batch, n_height, n_width), dtype=np.float32)
  start_time = time.time()
  for step in range(n_step):
    batch_start = step*n_batch
    batch_end = step*n_batch+n_batch
    d = session.run(model.model0[0])
    d_arr[batch_start:batch_end, :, :] = d[:, :, :, 0]
  end_time = time.time()
  log('Total time: %.1f ms  Average time per image: %.1f ms' %
      (1000*(end_time-start_time), (1000*(end_time-start_time)/n_sample)))
  d_arr = d_arr[0:n_sample, :, :]
  output_path = os.path.join(output_path, 'disparities.npy')
  log('Storing predictions to %s' % output_path)
  if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
  np.save(output_path, d_arr)

