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

import os, sys
import argparse

import global_constants as settings
from monodispnet import train

parser = argparse.ArgumentParser(
  description='Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction -- TensorFlow implementation')

# Training and validation input filepaths
parser.add_argument('--trn_im0_path',         type=str, required=True)
parser.add_argument('--trn_im1_path',         type=str, required=True)
# Batch parameters
parser.add_argument('--n_batch',              type=int, default=settings.N_BATCH)
parser.add_argument('--n_height',             type=int, default=settings.N_HEIGHT)
parser.add_argument('--n_width',              type=int, default=settings.N_WIDTH)
parser.add_argument('--n_channel',            type=int, default=settings.N_CHANNEL)
# Hyper parameters
parser.add_argument('--n_epoch',              type=int, default=settings.N_EPOCH)
parser.add_argument('--learning_rates',       type=str, default=settings.LEARNING_RATES_TXT)
parser.add_argument('--learning_bounds',      type=str, default=settings.LEARNING_BOUNDS_TXT)
parser.add_argument('--n_pyramid',            type=int, default=settings.N_PYRAMID)
parser.add_argument('--max_disparity',        type=float, default=settings.MAX_DISPARITY)
parser.add_argument('--w_ph',                 type=float, default=settings.W_PH)
parser.add_argument('--w_st',                 type=float, default=settings.W_ST)
parser.add_argument('--w_sm',                 type=float, default=settings.W_SM)
parser.add_argument('--w_bc',                 type=float, default=settings.W_BC)
parser.add_argument('--w_ar',                 type=float, default=settings.W_AR)
# Checkpoint and restore paths
parser.add_argument('--checkpoint_path',      type=str, default=settings.CHECKPOINT_PATH)
parser.add_argument('--restore_path',         type=str, default=settings.RESTORE_PATH)
parser.add_argument('--n_checkpoint',         type=int, default=settings.N_CHECKPOINT)
parser.add_argument('--n_summary',            type=int, default=settings.N_SUMMARY)
# Hardware settings
parser.add_argument('--n_gpu',                type=int, default=settings.N_GPU)
parser.add_argument('--n_thread',             type=int, default=settings.N_THREAD)

args = parser.parse_args()

if __name__ == '__main__':

  args.learning_rates = args.learning_rates.split(',')
  args.learning_rates = [float(r) for r in args.learning_rates]
  args.learning_bounds = args.learning_bounds.split(',')
  args.learning_bounds = [float(b) for b in args.learning_bounds]
  assert(len(args.learning_rates) == len(args.learning_bounds)+1)

  train(args.trn_im0_path, args.trn_im1_path,
        n_batch=args.n_batch, n_height=args.n_height, n_width=args.n_width, n_channel=args.n_channel,
        n_epoch=args.n_epoch, learning_rates=args.learning_rates, learning_bounds=args.learning_bounds,
        n_pyramid=args.n_pyramid, max_disparity=args.max_disparity,
        w_ph=args.w_ph, w_st=args.w_st, w_sm=args.w_sm, w_bc=args.w_bc, w_ar=args.w_ar,
        n_checkpoint=args.n_checkpoint, n_summary=args.n_summary,
        checkpoint_path=args.checkpoint_path, restore_path=args.restore_path,
        n_gpu=args.n_gpu, n_thread=args.n_thread)
