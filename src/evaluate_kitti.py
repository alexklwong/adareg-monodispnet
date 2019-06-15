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
import os, cv2
import argparse
import data_utils
import eval_utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path',            type=str, required=True)
parser.add_argument('--ims_path',            type=str, required=True)
parser.add_argument('--gts_path',            type=str, required=True)
parser.add_argument('--split',               type=str, required=True)
parser.add_argument('--min_depth',           type=float, default=1e-3)
parser.add_argument('--max_depth',           type=float, default=80)
parser.add_argument('--save_ims',            action='store_true')
parser.add_argument('--save_gts',            action='store_true')
parser.add_argument('--save_ds',             action='store_true')
parser.add_argument('--out_path',            type=str, default='./')

args = parser.parse_args()

KITTI = 'kitti'
EIGEN = 'eigen'

if __name__ == '__main__':

  f = { 1226:707.0912, 1242:721.5377, 1241:718.856, 1238:718.3351, 1224:707.0493 }
  B = { 1226:0.5379045, 1242:0.53272545, 1241:0.5323319, 1238:0.53014046, 1224:0.5372559 }

  args.split = args.split.lower()
  # Load numpy array containing predictions
  np_arr = np.squeeze(np.load(args.npy_path))
  # Read image and ground truth paths
  im_paths = data_utils.read_paths(args.ims_path)
  gt_paths = data_utils.read_paths(args.gts_path)
  assert(len(im_paths) == len(gt_paths))
  assert(len(im_paths) == np_arr.shape[0])
  # Load data from paths
  im_arr = []
  gt_arr = []
  d_arr = []
  n_sample = len(im_paths)
  for i in range(n_sample):
    im_arr.append(cv2.imread(im_paths[i]))
    # Load ground truth
    if args.split == KITTI:
      gt_arr.append(cv2.imread(gt_paths[i], -1).astype(np.float32)/255.0)
    elif args.split == EIGEN:
      gt_arr.append(np.load(gt_paths[i]))
    # Resize predictions back to original size
    d = cv2.resize(np_arr[i], (im_arr[i].shape[1], im_arr[i].shape[0]),
                   interpolation=cv2.INTER_LINEAR)
    d_arr.append(d*d.shape[1])
  # List all metrics
  rms     = np.zeros(n_sample, np.float32)
  log_rms = np.zeros(n_sample, np.float32)
  abs_rel = np.zeros(n_sample, np.float32)
  sq_rel  = np.zeros(n_sample, np.float32)
  d1_all  = np.zeros(n_sample, np.float32)
  a1      = np.zeros(n_sample, np.float32)
  a2      = np.zeros(n_sample, np.float32)
  a3      = np.zeros(n_sample, np.float32)
  # Compute metrics for each sample
  for i in range(n_sample):
    d = d_arr[i]
    gt = gt_arr[i]
    # Apply crops/mask to ground truth depending on split
    if args.split == KITTI:
      d1_all[i] = 100.0*eval_utils.end_point_err(d_arr[i], gt_arr[i])
      mask = gt > 0
      gt = eval_utils.disparity2depth(gt[mask], f[gt.shape[1]], B[gt.shape[1]])
    elif args.split == EIGEN:
      mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
      height, width = gt.shape
      crop = np.array([0.40810811*height, 0.99189189*height,
                       0.03594771*width,  0.96405229*width]).astype(np.int32)
      crop_mask = np.zeros(mask.shape)
      crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
      mask = np.logical_and(mask, crop_mask)
      gt = gt[mask]
    z = eval_utils.disparity2depth(d[mask], f[d.shape[1]], B[d.shape[1]])
    z[np.isinf(z)] = 0
    z[z < args.min_depth] = args.min_depth
    z[z > args.max_depth] = args.max_depth
    # Compute metrics
    abs_rel[i]  = eval_utils.abs_rel_err(z, gt)
    sq_rel[i]   = eval_utils.sq_rel_err(z, gt)
    rms[i]      = eval_utils.rms_err(z, gt)
    log_rms[i]  = eval_utils.log_rms_err(z, gt)
    a1[i]       = eval_utils.ratio_out_thresh_err(z, gt, tau=1.25)
    a2[i]       = eval_utils.ratio_out_thresh_err(z, gt, tau=1.25**2)
    a3[i]       = eval_utils.ratio_out_thresh_err(z, gt, tau=1.25**3)
  # Take the mean over all samples
  abs_rel = np.mean(abs_rel)
  sq_rel  = np.mean(sq_rel)
  rms     = np.mean(rms)
  log_rms = np.mean(log_rms)
  d1_all  = np.mean(d1_all)
  a1      = np.mean(a1)
  a2      = np.mean(a2)
  a3      = np.mean(a3)

  print("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        'AbsRel', 'SqRel', 'RMSE', 'LogRMSE', 'EPE(D1)', 'd<1.25', 'd<1.25^2', 'd<1.25^3'))
  print("{:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}".format(
        abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3))

  if args.save_ims or args.save_gts or args.save_ds:
    print("Storing visualizations into %s" % args.out_path)
    for i in range(len(im_paths)):
      name, _ = os.path.splitext(im_paths[i].replace("/", "_"))
      if args.save_ims:
        out_path_im = os.path.join(args.out_path, 'im')
        if not os.path.isdir(out_path_im):
          os.makedirs(out_path_im)
        plt.imsave(os.path.join(out_path_im, "{}_im.png".format(name)), im_arr[i])
      if args.save_gts:
        out_path_gt = os.path.join(args.out_path, 'gt')
        if not os.path.isdir(out_path_gt):
          os.makedirs(out_path_gt)
        plt.imsave(os.path.join(out_path_gt, "{}_gt.png".format(name)), gt_arr[i], cmap='plasma')
      if args.save_ds:
        out_path_d = os.path.join(args.out_path, 'd')
        if not os.path.isdir(out_path_d):
          os.makedirs(out_path_d)
        plt.imsave(os.path.join(out_path_d, "{}_d.png".format(name)), d_arr[i], cmap='plasma')

