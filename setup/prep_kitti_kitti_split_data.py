'''
Author: Alex Wong <alexw@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, B. W. Hong and S. Soatto. Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.
https://arxiv.org/abs/1903.07309

@article{wong2018bilateral,
  title={Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction},
  author={Wong, Alex and Hong, Byung-Woo and Soatto, Stefano},
  journal={arXiv preprint arXiv:1903.07309},
  year={2019}
}
'''

import sys, os
sys.path.insert(0, 'src')
import data_utils
import numpy as np
import cv2

DATA_SPLIT_DIR = 'data_split'

KITTI_TRN_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'kitti_train.txt')
KITTI_VAL_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'kitti_validation.txt')
KITTI_UNU_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'kitti_test.txt')
KITTI_TST_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'kitti_split_test.txt')
DATA_DIR_PREFIX = 'data/kitti_raw_data'
STEREO_FLOW_DIR_PREFIX = 'data/data_scene_flow'
STEREO_FLOW_GT_DIR = os.path.join(os.path.join(STEREO_FLOW_DIR_PREFIX, 'training'),
                                  'disp_noc_0')

TRN_REFS_DIRPATH = 'training'
VAL_REFS_DIRPATH = 'validation'
TST_REFS_DIRPATH = 'testing'

TRN_GTD_DIRPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti/ground_truth')
VAL_GTD_DIRPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti/ground_truth')
TST_GTD_DIRPATH = os.path.join(TST_REFS_DIRPATH, 'kitti/ground_truth')

TRN_IM0_FILEPATH = os.path.join(TRN_REFS_DIRPATH, 'kitti_trn_im0.txt')
TRN_IM1_FILEPATH = os.path.join(TRN_REFS_DIRPATH, 'kitti_trn_im1.txt')
VAL_IM0_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_val_im0.txt')
VAL_IM1_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_val_im1.txt')
VAL_GTD_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_val_gtd.txt')
UNU_IM0_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_unu_im0.txt')
UNU_IM1_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_unu_im1.txt')
UNU_GTD_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_unu_gtd.txt')
TST_IM0_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_tst_im0.txt')
TST_IM1_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_tst_im1.txt')
TST_GTD_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'kitti_tst_gtd.txt')

kitti_trn_paths = data_utils.read_paths(KITTI_TRN_PATHS_FILE)
kitti_val_paths = data_utils.read_paths(KITTI_VAL_PATHS_FILE)
kitti_unu_paths = data_utils.read_paths(KITTI_UNU_PATHS_FILE)
kitti_tst_paths = data_utils.read_paths(KITTI_TST_PATHS_FILE)

if not os.path.exists(TRN_REFS_DIRPATH):
  os.makedirs(TRN_REFS_DIRPATH)
if not os.path.exists(VAL_REFS_DIRPATH):
  os.makedirs(VAL_REFS_DIRPATH)
if not os.path.exists(TST_REFS_DIRPATH):
  os.makedirs(TST_REFS_DIRPATH)

kitti_trn_im0_paths = []
kitti_trn_im1_paths = []
for trn_idx in range(len(kitti_trn_paths)):
  sys.stdout.write(
    'Reading {}/{} training file paths...\r'.format(trn_idx+1, len(kitti_trn_paths)))
  sys.stdout.flush()
  kitti_trn_im0_path, kitti_trn_im1_path = kitti_trn_paths[trn_idx].split()
  kitti_trn_im0_path = os.path.join(DATA_DIR_PREFIX, kitti_trn_im0_path)
  kitti_trn_im1_path = os.path.join(DATA_DIR_PREFIX, kitti_trn_im1_path)
  kitti_trn_im0_paths.append(kitti_trn_im0_path)
  kitti_trn_im1_paths.append(kitti_trn_im1_path)
print('Completed reading {}/{} training file paths'.format(trn_idx+1, len(kitti_trn_paths)))
assert(len(kitti_trn_im0_paths) == len(kitti_trn_im1_paths))

print('Storing training image 0 file paths into: %s' % TRN_IM0_FILEPATH)
with open(TRN_IM0_FILEPATH, "w") as o:
  for idx in range(len(kitti_trn_im0_paths)):
    o.write(kitti_trn_im0_paths[idx]+'\n')
print('Storing training image 1 file paths into: %s' % TRN_IM1_FILEPATH)
with open(TRN_IM1_FILEPATH, "w") as o:
  for idx in range(len(kitti_trn_im1_paths)):
    o.write(kitti_trn_im1_paths[idx]+'\n')

kitti_val_im0_paths = []
kitti_val_im1_paths = []
for val_idx in range(len(kitti_val_paths)):
  sys.stdout.write(
    'Reading {}/{} validation file paths...\r'.format(val_idx+1, len(kitti_val_paths)))
  sys.stdout.flush()
  kitti_val_im0_path, kitti_val_im1_path = kitti_val_paths[val_idx].split()
  kitti_val_im0_path = os.path.join(DATA_DIR_PREFIX, kitti_val_im0_path)
  kitti_val_im1_path = os.path.join(DATA_DIR_PREFIX, kitti_val_im1_path)
  kitti_val_im0_paths.append(kitti_val_im0_path)
  kitti_val_im1_paths.append(kitti_val_im1_path)
print('Completed reading {}/{} validation file paths'.format(val_idx+1, len(kitti_val_paths)))
assert(len(kitti_val_im0_paths) == len(kitti_val_im1_paths))

print('Storing validation image 0 file paths into: %s' % VAL_IM0_FILEPATH)
with open(VAL_IM0_FILEPATH, "w") as o:
  for idx in range(len(kitti_val_im0_paths)):
    o.write(kitti_val_im0_paths[idx]+'\n')
print('Storing validation image 1 file paths into: %s' % VAL_IM1_FILEPATH)
with open(VAL_IM1_FILEPATH, "w") as o:
  for idx in range(len(kitti_val_im1_paths)):
    o.write(kitti_val_im1_paths[idx]+'\n')

kitti_unu_im0_paths = []
kitti_unu_im1_paths = []
for unu_idx in range(len(kitti_unu_paths)):
  sys.stdout.write(
    'Reading {}/{} unused test file paths...\r'.format(unu_idx+1, len(kitti_unu_paths)))
  sys.stdout.flush()
  kitti_unu_im0_path, kitti_unu_im1_path = kitti_unu_paths[unu_idx].split()
  kitti_unu_im0_path = os.path.join(DATA_DIR_PREFIX, kitti_unu_im0_path)
  kitti_unu_im1_path = os.path.join(DATA_DIR_PREFIX, kitti_unu_im1_path)
  kitti_unu_im0_paths.append(kitti_unu_im0_path)
  kitti_unu_im1_paths.append(kitti_unu_im1_path)
print('Completed reading {}/{} unused test file paths'.format(unu_idx+1, len(kitti_unu_paths)))
assert(len(kitti_unu_im0_paths) == len(kitti_unu_im1_paths))

print('Storing unused test image 0 file paths into: %s' % UNU_IM0_FILEPATH)
with open(UNU_IM0_FILEPATH, "w") as o:
  for idx in range(len(kitti_unu_im0_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx+1, len(kitti_unu_im0_paths)))
    sys.stdout.flush()
    o.write(kitti_unu_im0_paths[idx]+'\n')
print('Storing unused test image 1 file paths into: %s' % UNU_IM1_FILEPATH)
with open(UNU_IM1_FILEPATH, "w") as o:
  for idx in range(len(kitti_unu_im1_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx, len(kitti_unu_im1_paths)))
    sys.stdout.flush()
    o.write(kitti_unu_im1_paths[idx]+'\n')

kitti_tst_im0_paths = []
kitti_tst_im1_paths = []
for tst_idx in range(len(kitti_tst_paths)):
  sys.stdout.write(
    'Reading {}/{} test file paths...\r'.format(tst_idx+1, len(kitti_tst_paths)))
  sys.stdout.flush()
  kitti_tst_im0_path, kitti_tst_im1_path = kitti_tst_paths[tst_idx].split()
  kitti_tst_im0_path = os.path.join(STEREO_FLOW_DIR_PREFIX, kitti_tst_im0_path)
  kitti_tst_im1_path = os.path.join(STEREO_FLOW_DIR_PREFIX, kitti_tst_im1_path)
  kitti_tst_im0_paths.append(kitti_tst_im0_path)
  kitti_tst_im1_paths.append(kitti_tst_im1_path)
print('Completed reading {}/{} test file paths'.format(tst_idx+1, len(kitti_tst_paths)))
assert(len(kitti_tst_im0_paths) == len(kitti_tst_im1_paths))

print('Storing test image 0 file paths into: %s' % TST_IM0_FILEPATH)
with open(TST_IM0_FILEPATH, "w") as o:
  for idx in range(len(kitti_tst_im0_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx+1, len(kitti_tst_im0_paths)))
    sys.stdout.flush()
    o.write(kitti_tst_im0_paths[idx]+'\n')
print('Storing test image 1 file paths into: %s' % TST_IM1_FILEPATH)
with open(TST_IM1_FILEPATH, "w") as o:
  for idx in range(len(kitti_tst_im1_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx, len(kitti_tst_im1_paths)))
    sys.stdout.flush()
    o.write(kitti_tst_im1_paths[idx]+'\n')

kitti_tst_gtd_paths = os.listdir(STEREO_FLOW_GT_DIR)
kitti_tst_gtd_paths.sort()
for gtd_idx in range(len(kitti_tst_gtd_paths)):
  sys.stdout.write(
    'Reading {}/{} test file paths...\r'.format(gtd_idx+1, len(kitti_tst_gtd_paths)))
  sys.stdout.flush()
  kitti_tst_gtd_paths[gtd_idx] = os.path.join(STEREO_FLOW_GT_DIR, kitti_tst_gtd_paths[gtd_idx])
print('Completed reading {}/{} ground-truth test file paths'.format(gtd_idx+1, len(kitti_tst_gtd_paths)))
assert(len(kitti_tst_gtd_paths) == len(kitti_tst_im0_paths))

print('Storing ground-truth test  file paths into: %s' % TST_GTD_FILEPATH)
with open(TST_GTD_FILEPATH, "w") as o:
  for idx in range(len(kitti_tst_gtd_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx+1, len(kitti_tst_im0_paths)))
    sys.stdout.flush()
    o.write(kitti_tst_gtd_paths[idx]+'\n')
