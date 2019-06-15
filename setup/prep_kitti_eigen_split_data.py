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

EIGEN_TRN_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'eigen_train.txt')
EIGEN_VAL_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'eigen_validation.txt')
EIGEN_TST_PATHS_FILE = os.path.join(DATA_SPLIT_DIR, 'eigen_test.txt')
DATA_DIR_PREFIX = 'data/kitti_raw_data'

TRN_REFS_DIRPATH = 'training'
VAL_REFS_DIRPATH = 'validation'
TST_REFS_DIRPATH = 'testing'

TRN_GTD_DIRPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen/ground_truth')
VAL_GTD_DIRPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen/ground_truth')
TST_GTD_DIRPATH = os.path.join(TST_REFS_DIRPATH, 'eigen/ground_truth')

TRN_IM0_FILEPATH = os.path.join(TRN_REFS_DIRPATH, 'eigen_trn_im0.txt')
TRN_IM1_FILEPATH = os.path.join(TRN_REFS_DIRPATH, 'eigen_trn_im1.txt')
VAL_IM0_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_im0.txt')
VAL_IM1_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_im1.txt')
VAL_GTD_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtd.txt')
VAL_GTF_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtf.npy')
VAL_GTB_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtb.npy')
TST_IM0_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_im0.txt')
TST_IM1_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_im1.txt')
TST_GTD_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtd.txt')
TST_GTF_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtf.npy')
TST_GTB_FILEPATH = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtb.npy')

eigen_trn_paths = data_utils.read_paths(EIGEN_TRN_PATHS_FILE)
eigen_val_paths = data_utils.read_paths(EIGEN_VAL_PATHS_FILE)
eigen_tst_paths = data_utils.read_paths(EIGEN_TST_PATHS_FILE)

if not os.path.exists(TRN_REFS_DIRPATH):
  os.makedirs(TRN_REFS_DIRPATH)
if not os.path.exists(VAL_REFS_DIRPATH):
  os.makedirs(VAL_REFS_DIRPATH)
if not os.path.exists(TST_REFS_DIRPATH):
  os.makedirs(TST_REFS_DIRPATH)

eigen_trn_im0_paths = []
eigen_trn_im1_paths = []
for trn_idx in range(len(eigen_trn_paths)):
  sys.stdout.write(
    'Reading {}/{} training file paths...\r'.format(trn_idx+1, len(eigen_trn_paths)))
  sys.stdout.flush()
  eigen_trn_im0_path, eigen_trn_im1_path = eigen_trn_paths[trn_idx].split()
  eigen_trn_im0_path = os.path.join(DATA_DIR_PREFIX, eigen_trn_im0_path)
  eigen_trn_im1_path = os.path.join(DATA_DIR_PREFIX, eigen_trn_im1_path)
  eigen_trn_im0_path = eigen_trn_im0_path[:-3]+'png'
  eigen_trn_im1_path = eigen_trn_im1_path[:-3]+'png'
  eigen_trn_im0_paths.append(eigen_trn_im0_path)
  eigen_trn_im1_paths.append(eigen_trn_im1_path)
print('Completed reading {}/{} training file paths'.format(trn_idx+1, len(eigen_trn_paths)))
assert(len(eigen_trn_im0_paths) == len(eigen_trn_im1_paths))

print('Storing training image 0 file paths into: %s' % TRN_IM0_FILEPATH)
with open(TRN_IM0_FILEPATH, "w") as o:
  for idx in range(len(eigen_trn_im0_paths)):
    o.write(eigen_trn_im0_paths[idx]+'\n')
print('Storing training image 1 file paths into: %s' % TRN_IM1_FILEPATH)
with open(TRN_IM1_FILEPATH, "w") as o:
  for idx in range(len(eigen_trn_im1_paths)):
    o.write(eigen_trn_im1_paths[idx]+'\n')

eigen_val_im0_paths = []
eigen_val_im1_paths = []
for val_idx in range(len(eigen_val_paths)):
  sys.stdout.write(
    'Reading {}/{} validation file paths...\r'.format(val_idx+1, len(eigen_val_paths)))
  sys.stdout.flush()
  eigen_val_im0_path, eigen_val_im1_path = eigen_val_paths[val_idx].split()
  eigen_val_im0_path = os.path.join(DATA_DIR_PREFIX, eigen_val_im0_path)
  eigen_val_im1_path = os.path.join(DATA_DIR_PREFIX, eigen_val_im1_path)
  eigen_val_im0_path = eigen_val_im0_path[:-3]+'png'
  eigen_val_im1_path = eigen_val_im1_path[:-3]+'png'
  eigen_val_im0_paths.append(eigen_val_im0_path)
  eigen_val_im1_paths.append(eigen_val_im1_path)
print('Completed reading {}/{} validation file paths'.format(val_idx+1, len(eigen_val_paths)))
assert(len(eigen_val_im0_paths) == len(eigen_val_im1_paths))

print('Storing validation image 0 file paths into: %s' % VAL_IM0_FILEPATH)
with open(VAL_IM0_FILEPATH, "w") as o:
  for idx in range(len(eigen_val_im0_paths)):
    o.write(eigen_val_im0_paths[idx]+'\n')
print('Storing validation image 1 file paths into: %s' % VAL_IM1_FILEPATH)
with open(VAL_IM1_FILEPATH, "w") as o:
  for idx in range(len(eigen_val_im1_paths)):
    o.write(eigen_val_im1_paths[idx]+'\n')

eigen_tst_im0_paths = []
eigen_tst_im1_paths = []
for tst_idx in range(len(eigen_tst_paths)):
  sys.stdout.write(
    'Reading {}/{} testing file paths...\r'.format(tst_idx+1, len(eigen_tst_paths)))
  sys.stdout.flush()
  eigen_tst_im0_path, eigen_tst_im1_path = eigen_tst_paths[tst_idx].split()
  eigen_tst_im0_path = os.path.join(DATA_DIR_PREFIX, eigen_tst_im0_path)
  eigen_tst_im1_path = os.path.join(DATA_DIR_PREFIX, eigen_tst_im1_path)
  eigen_tst_im0_path = eigen_tst_im0_path[:-3]+'png'
  eigen_tst_im1_path = eigen_tst_im1_path[:-3]+'png'
  eigen_tst_im0_paths.append(eigen_tst_im0_path)
  eigen_tst_im1_paths.append(eigen_tst_im1_path)
print('Completed reading {}/{} testing file paths'.format(tst_idx+1, len(eigen_tst_paths)))
assert(len(eigen_tst_im0_paths) == len(eigen_tst_im1_paths))

print('Storing test image 0 file paths into: %s' % TST_IM0_FILEPATH)
with open(TST_IM0_FILEPATH, "w") as o:
  for idx in range(len(eigen_tst_im0_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx+1, len(eigen_tst_im0_paths)))
    sys.stdout.flush()
    o.write(eigen_tst_im0_paths[idx]+'\n')
  print('Completed storing {}/{} test image 0...'.format(idx+1, len(eigen_tst_im0_paths)))
print('Storing test image 1 file paths into: %s' % TST_IM1_FILEPATH)
with open(TST_IM1_FILEPATH, "w") as o:
  for idx in range(len(eigen_tst_im1_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx, len(eigen_tst_im1_paths)))
    sys.stdout.flush()
    o.write(eigen_tst_im1_paths[idx]+'\n')
  print('Completed storing {}/{} test image 1...'.format(idx+1, len(eigen_tst_im1_paths)))

# Generate depth maps for validation data
eigen_val_gtd_paths = []
num_errors = 0
f = np.zeros(len(eigen_val_im0_paths))
B = np.zeros(len(eigen_val_im0_paths))
for idx, path in enumerate(eigen_val_im0_paths):
  sys.stdout.write(
    'Generating ground-truth for validation {}/{}...\r'.format(idx+1, len(eigen_val_im0_paths)))
  sys.stdout.flush()
  # Extract useful components from paths
  _, _, date, sequence, camera, _, filename = path.split('/')
  camera_id = np.int32(camera[-1])
  file_id, ext = os.path.splitext(filename)
  velodyne_path = '{}/{}/velodyne_points/data/{}.bin'.format(date, sequence, file_id)
  velodyne_path = os.path.join(DATA_DIR_PREFIX, velodyne_path)
  calibration_dir = os.path.join(DATA_DIR_PREFIX, date)
  # Make sure both the file and velodyne path exists
  if os.path.isfile(path) and os.path.isfile(velodyne_path):
    shape = cv2.imread(path).shape[:2]
    depth = data_utils.velodyne2depth(calibration_dir, velodyne_path, shape, camera_id=2)
    f[idx], B[idx] = data_utils.load_focal_length_baseline(calibration_dir, camera_id=2)
    eigen_val_gtd_path = os.path.join(VAL_GTD_DIRPATH, os.path.splitext(path)[0]+'.npy')
    if not os.path.exists(os.path.dirname(eigen_val_gtd_path)):
      os.makedirs(os.path.dirname(eigen_val_gtd_path))
    np.save(eigen_val_gtd_path, depth)
    eigen_val_gtd_paths.append(eigen_val_gtd_path)
  else:
    num_errors += 1
    print('ERROR: %s does not exist!' % path)
print('Completed generating ground-truth for validation {}/{}\r'.format(idx+1, len(eigen_val_im0_paths)))
print('Storing validation focal lengths into: %s' % VAL_GTF_FILEPATH)
np.save(VAL_GTF_FILEPATH, f)
print('Storing validation baselines into: %s' % VAL_GTB_FILEPATH)
np.save(VAL_GTB_FILEPATH, B)
print('Storing validation image depths with file paths into: %s' % VAL_GTD_FILEPATH)
with open(VAL_GTD_FILEPATH, "w") as o:
  for idx in range(len(eigen_val_gtd_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx, len(eigen_val_gtd_paths)))
    sys.stdout.flush()
    o.write(eigen_val_gtd_paths[idx]+'\n')
  print('Completed storing ground-truth for validation {}/{}\r'.format(idx+1, len(eigen_val_im0_paths)))
# Generate depth maps for test data
eigen_tst_gtd_paths = []
num_errors = 0
f = np.zeros(len(eigen_tst_im0_paths))
B = np.zeros(len(eigen_tst_im0_paths))
for idx, path in enumerate(eigen_tst_im0_paths):
  sys.stdout.write(
    'Generating ground-truth for testing {}/{}...\r'.format(idx+1, len(eigen_tst_im0_paths)))
  sys.stdout.flush()
  # Extract useful components from paths
  _, _, date, sequence, camera, _, filename = path.split('/')
  camera_id = np.int32(camera[-1])
  file_id, ext = os.path.splitext(filename)
  velodyne_path = '{}/{}/velodyne_points/data/{}.bin'.format(date, sequence, file_id)
  velodyne_path = os.path.join(DATA_DIR_PREFIX, velodyne_path)
  calibration_dir = os.path.join(DATA_DIR_PREFIX, date)
  # Make sure both the file and velodyne path exists
  if os.path.isfile(path) and os.path.isfile(velodyne_path):
    shape = cv2.imread(path).shape[:2]
    depth = data_utils.velodyne2depth(calibration_dir, velodyne_path, shape, camera_id=2)
    f[idx], B[idx] = data_utils.load_focal_length_baseline(calibration_dir, camera_id=2)
    eigen_tst_gtd_path = os.path.join(TST_GTD_DIRPATH, os.path.splitext(path)[0]+'.npy')
    if not os.path.exists(os.path.dirname(eigen_tst_gtd_path)):
      os.makedirs(os.path.dirname(eigen_tst_gtd_path))
    np.save(eigen_tst_gtd_path, depth)
    eigen_tst_gtd_paths.append(eigen_tst_gtd_path)
  else:
    num_errors += 1
    print('ERROR: %s does not exist!' % path)
print('Completed generating ground-truth for testing {}/{}\r'.format(idx+1, len(eigen_tst_im0_paths)))
print('Storing validation focal lengths into: %s' % TST_GTF_FILEPATH)
np.save(TST_GTF_FILEPATH, f)
print('Storing validation baselines into: %s' % TST_GTB_FILEPATH)
np.save(TST_GTB_FILEPATH, B)
print('Storing testing image depths with file paths into: %s' % TST_GTD_FILEPATH)
with open(TST_GTD_FILEPATH, "w") as o:
  for idx in range(len(eigen_tst_gtd_paths)):
    sys.stdout.write('Progress {}/{}...\r'.format(idx, len(eigen_tst_gtd_paths)))
    sys.stdout.flush()
    o.write(eigen_tst_gtd_paths[idx]+'\n')
  print('Completed storing ground-truth for testing {}/{}\r'.format(idx+1, len(eigen_tst_im0_paths)))
