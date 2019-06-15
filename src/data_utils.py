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
import os, random, pickle
import numpy as np
import cv2
import tensorflow as tf
from collections import Counter
from matplotlib import pyplot as plt

def log(s, filepath=None, to_console=True):
  if to_console:
    print(s)
  if filepath != None:
    if not os.path.isdir(os.path.dirname(filepath)):
      os.makedirs(os.path.dirname(filepath))
      with open(filepath, "w+") as o:
        o.write(s+'\n')
    else:
      with open(filepath, "a+") as o:
        o.write(s+'\n')

"""
  Util functions for reading, storing and setting up data
"""
def read_paths(filepath):
  path_list = []
  with open(filepath) as f:
     while True:
      path = f.readline().rstrip('\n')
      # If there was nothing to read
      if path == '':
        break
      path_list.append(path)

  return path_list

def load_disparities(filepaths):
  disparities = []
  for i in range(len(filepaths)):
    d = cv2.imread(filepaths[i], 0)
    disparities.append(d)
  return disparities

def load_calibration(path):
  float_chars = set("0123456789.e+- ")
  data = {}
  with open(path, 'r') as f:
    for line in f.readlines():
      key, value = line.split(':', 1)
      value = value.strip()
      data[key] = value
      if float_chars.issuperset(value):
        try:
          data[key] = np.array(map(float, value.split(' ')))
        except ValueError:
          pass
  return data

def load_velodyne(path):
  points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
  points[:, 3] = 1.0  # Homogeneous
  return points

def load_focal_length_baseline(calibration_dir, camera_id):
  cam2cam = load_calibration(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
  P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
  P3_rect = cam2cam['P_rect_03'].reshape(3, 4)
  # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
  b2 = P2_rect[0, 3]/-P2_rect[0, 0]
  b3 = P3_rect[0, 3]/-P3_rect[0, 0]
  baseline = b3-b2
  if camera_id == 2:
      focal_length = P2_rect[0, 0]
  elif camera_id == 3:
      focal_length = P3_rect[0, 0]

  return focal_length, baseline

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def velodyne2depth(calibration_dir, velodyne_path, shape, camera_id=2):
  # Load calibration files
  cam2cam = load_calibration(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
  velo2cam = load_calibration(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))
  velo2cam = np.hstack((velo2cam['R'].reshape(3,3), velo2cam['T'][..., np.newaxis]))
  velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
  # Compute projection matrix from velodyne to image plane
  R_cam2rect = np.eye(4)
  R_cam2rect[:3,:3] = cam2cam['R_rect_00'].reshape(3, 3)
  P_rect = cam2cam['P_rect_0'+str(camera_id)].reshape(3, 4)
  P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
  # Load velodyne points and remove all that are behind image plane (approximation)
  # Each row of the velodyne data refers to forward, left, up, reflectance
  velo = load_velodyne(velodyne_path)
  velo = velo[velo[:, 0] >= 0, :]
  # Project the points to the camera
  velo_pts_im = np.dot(P_velo2im, velo.T).T
  velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]
  velo_pts_im[:, 2] = velo[:, 0]
  # Check if in bounds (use minus 1 to get the exact same value as KITTI matlab code)
  velo_pts_im[:, 0] = np.round(velo_pts_im[:,0])-1
  velo_pts_im[:, 1] = np.round(velo_pts_im[:,1])-1
  val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
  val_inds = val_inds & (velo_pts_im[:,0] < shape[1]) & (velo_pts_im[:,1] < shape[0])
  velo_pts_im = velo_pts_im[val_inds, :]
  # Project to image
  depth = np.zeros(shape)
  depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]
  # Find the duplicate points and choose the closest depth
  inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
  dupe_inds = [item for item, count in Counter(inds).iteritems() if count > 1]
  for dd in dupe_inds:
    pts = np.where(inds==dd)[0]
    x_loc = int(velo_pts_im[pts[0], 0])
    y_loc = int(velo_pts_im[pts[0], 1])
    depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
  # Clip all depth values less than 0 to 0
  depth[depth < 0] = 0
  return depth.astype(np.float32)

def disparity2depth(d, b, f):
  z = f*b/d
  z[np.isinf(z)] = 0
  return z

"""
  Util functions for setting up image batches and epochs
"""
def get_shuffle(n_samples):
  order = range(n_samples)
  random.shuffle(order)
  return order

def make_epoch(data, order):
  assert(len(data) == len(order))
  data_epoch = [data[i] for i in order]
  return data_epoch

def shuffle_and_drop(list_of_paths, batch_size):
    assert len(list_of_paths)
    n = len(list_of_paths[0]) // batch_size
    n *= batch_size
    idx = np.arange(len(list_of_paths[0]))
    np.random.shuffle(idx)
    return [[paths[i] for i in idx[:n]] for paths in list_of_paths]

def pad_batch(filepaths, n_batch):
  n_samples = len(filepaths)
  if n_samples % n_batch > 0:
    n_pad = n_batch-(n_samples % n_batch)
    filepaths.extend([filepaths[-1]]*n_pad)
  return filepaths


