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

import tensorflow as tf

class DataLoader(object):
  """
  Data loader class based on tf.data.Dataset
  """

  def __init__(self, shape, name='', normalize=True, shuffle=False, n_thread=8, prefetch_size=8,
               random_flip=False, random_gamma=False, gamma_range=None,
               random_brightness=False, brightness_range=None, random_color=False, color_range=None):
    self.n_batch = shape[0]
    self.n_height = shape[1]
    self.n_width = shape[2]
    self.n_channel = shape[3]
    self.normalize = normalize
    self.n_thread = n_thread
    self.prefetch_size = prefetch_size
    self.random_flip = random_flip
    self.random_gamma = random_gamma
    self.gamma_range = gamma_range
    self.random_brightness = random_brightness
    self.brightness_range = brightness_range
    self.random_color = random_color
    self.color_range = color_range
    self.shuffle = shuffle

    # The dataset is in charge of data shuffling by default.
    # But we can also shuffle the data beforehand and pass them to
    # dataset loader without asking the dataset to do so.
    self.scope_name = 'dataloader_'+name
    with tf.variable_scope(self.scope_name):
      # Boolean placeholder for switching on/off data augmentation
      self.augment_placeholder = tf.placeholder(tf.bool, shape=())
      self.entry0_placeholder = tf.placeholder(tf.string, shape=[None])
      self.entry1_placeholder = tf.placeholder(tf.string, shape=[None])

      self.dataset = tf.data.Dataset.from_tensor_slices(
          (self.entry0_placeholder, self.entry1_placeholder))

      # Random shuffling and data augmentation
      if self.shuffle:
        self.dataset = self.dataset.shuffle(buffer_size=100000)
      self.dataset = self.dataset \
          .map(self._load_func, num_parallel_calls=self.n_thread) \
          .map(self._augment_func, num_parallel_calls=self.n_thread) \
          .batch(self.n_batch) \
          .prefetch(buffer_size=self.prefetch_size)

      self.iterator = self.dataset.make_initializable_iterator()
      self.next_element = self.iterator.get_next()

      # Manually specify the shape of tensors
      # Since in tensorflow, the tensors returned by next_element only
      # have partial shapes -- [?, H, W, C] where batch size is determined
      # by the actual size of the batch returned (last batch might have
      # less than batch_size images).
      # However, since we pad the sequences, we always have fixed-size batches.
      self.next_element[0].set_shape(
          [self.n_batch, self.n_height, self.n_width, self.n_channel])
      self.next_element[1].set_shape(
          [self.n_batch, self.n_height, self.n_width, self.n_channel])

  def _load_func(self, path0, path1):
    with tf.variable_scope(self.scope_name):
      im0 = tf.image.resize_images(
          tf.expand_dims(tf.image.decode_png(tf.read_file(path0)), dim=0),
          [self.n_height, self.n_width],
          method=tf.image.ResizeMethod.AREA)
      im0 = tf.squeeze(im0, axis=0)
      im1 = tf.image.resize_images(
          tf.expand_dims(tf.image.decode_png(tf.read_file(path1)), dim=0),
          [self.n_height, self.n_width],
          method=tf.image.ResizeMethod.AREA)
      im1 = tf.squeeze(im1, axis=0)
      if self.normalize:
        # Normaize to [0, 1]
        im0 = tf.divide(im0, 255.0)
        im1 = tf.divide(im1, 255.0)

    return im0, im1

  def _augment_func(self, im0, im1):

    def augment(in0, in1):
      if self.random_flip:
        do_flip = tf.random_uniform([], 0.0, 1.0)
        im0 = tf.cond(do_flip > 0.5,
                      lambda: tf.image.flip_left_right(in1),
                      lambda: in0)
        im1 = tf.cond(do_flip > 0.5,
                      lambda: tf.image.flip_left_right(in0),
                      lambda: in1)
      else:
        im0, im1 = in0, in1
      # Color augmentations
      do_color = tf.random_uniform([], 0.0, 1.0)
      if self.random_gamma and self.gamma_range is not None:
        im0, im1 = tf.cond(do_color > 0.5,
                           lambda: self._augment_gamma(im0, im1, self.gamma_range),
                           lambda: (im0, im1))
      if self.random_brightness and self.brightness_range is not None:
        im0, im1 = tf.cond(do_color > 0.5,
                           lambda: self._augment_brightness(im0, im1, self.brightness_range),
                           lambda: (im0, im1))
      if self.random_color and self.color_range is not None:
        im0, im1 = tf.cond(do_color > 0.5,
                           lambda: self._augment_color(im0, im1, self.color_range),
                           lambda: (im0, im1))
      im0 = tf.clip_by_value(im0, 0.0, 1.0)
      im1 = tf.clip_by_value(im1, 0.0, 1.0)
      return im0, im1

    with tf.variable_scope(self.scope_name):
      im0, im1 = tf.cond(self.augment_placeholder,
                         lambda: augment(im0, im1),
                         lambda: (im0, im1))

    return im0, im1

  def _augment_gamma(self, im0, im1, gamma_range):
    gamma =  tf.random_uniform([1], gamma_range[0], gamma_range[1])
    im0 = tf.pow(im0, gamma)
    im1 = tf.pow(im1, gamma)
    return im0, im1

  def _augment_brightness(self, im0, im1, brightness_range):
    bright = tf.random_uniform([1], brightness_range[0], brightness_range[1])
    im0 = tf.multiply(im0, bright)
    im1 = tf.multiply(im1, bright)
    return im0, im1

  def _augment_color(self, im0, im1, color_range):
    dims = im0.get_shape().as_list()
    colors = tf.random_uniform([3], color_range[0], color_range[1])
    colors = tf.tile(tf.reshape(colors, [1, 1, 3]), [dims[0], dims[1], 1])
    im0 = tf.multiply(im0, colors)
    im1 = tf.multiply(im1, colors)
    return im0, im1

  def initialize(self, session=None, augment=False,
                 im0_paths=None, im1_paths=None):
    assert session is not None

    feed_dict = {
        self.entry0_placeholder: im0_paths,
        self.entry1_placeholder: im1_paths,
        self.augment_placeholder: augment
    }
    session.run(self.iterator.initializer, feed_dict)

