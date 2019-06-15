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

from __future__ import with_statement
import numpy as np

F = { 1226:707.0912, 1242:721.5377, 1241:718.856, 1238:718.3351, 1224:707.0493 }
B = { 1226:0.5379045, 1242:0.53272545, 1241:0.5323319, 1238:0.53014046, 1224:0.5372559 }

def disparity2depth(d, f, b):
  return f*b/d

def end_point_err(d, gt, tau_d=3, tau_p=0.05):
  with np.errstate(divide='ignore', invalid='ignore'):
    d = np.where(gt > 0, d, gt)
    n_eval = len(np.where(gt > 0)[0])
    # Compute error
    err = np.abs(d-gt)
    # ||d_est - d_gt|| > tau_d
    cond_d = err > tau_d
    # ||d_est - d_gt||/||d_gt|| > tau_p
    cond_p = np.where(gt > 0, np.divide(err, np.abs(gt)), gt) > tau_p
    if tau_p == 0.0:
      cond = cond_d
    elif tau_d == 0.0:
      cond = cond_p
    else:
      cond = np.logical_and(cond_d, cond_p)
    err_mat = np.where(cond, np.ones(gt.shape), np.zeros(gt.shape))
    # Return the percentage error
    return np.divide(np.sum(err_mat), n_eval)

def ratio_out_thresh_err(d, gt, tau=1.25):
  a_ratio = np.maximum((gt/d), (d/gt))
  a_mean = np.mean((a_ratio < tau))
  return a_mean

def rms_err(d, gt):
  rmse = (gt-d)**2
  rmse = np.sqrt(np.mean(rmse))
  return rmse

def abs_log_err(d, gt):
  return np.mean(np.abs(np.log10(gt)-np.log10(d)))

def log_rms_err(d, gt):
  rmse_log = (np.log(gt)-np.log(d))**2
  rmse_log = np.sqrt(np.mean(rmse_log))
  return rmse_log

def abs_rel_err(d, gt):
  abs_rel = np.mean(np.abs(gt-d)/gt)
  return abs_rel

def sq_rel_err(d, gt):
  sq_rel = np.mean(((gt-d)**2)/gt)
  return sq_rel
