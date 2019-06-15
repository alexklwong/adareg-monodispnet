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
from monodispnet import evaluate

parser = argparse.ArgumentParser(
  description='Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction -- TensorFlow implementation')

# File paths to images, checkpoint and output locations
parser.add_argument('--im0_path',             type=str, required=True)
parser.add_argument('--restore_path',         type=str, required=True)
parser.add_argument('--output_path',          type=str, default=settings.OUTPUT_PATH)
# Input batch parameters
parser.add_argument('--n_batch',              type=int, default=settings.N_BATCH)
parser.add_argument('--n_height',             type=int, default=settings.N_HEIGHT)
parser.add_argument('--n_width',              type=int, default=settings.N_WIDTH)
parser.add_argument('--n_channel',            type=int, default=settings.N_CHANNEL)
# Network prediction parameters
parser.add_argument('--n_pyramid',            type=int, default=settings.N_PYRAMID)
parser.add_argument('--max_disparity',        type=float, default=settings.MAX_DISPARITY)
# Hardware settings
parser.add_argument('--n_gpu',                type=int, default=settings.N_GPU)
parser.add_argument('--n_thread',             type=int, default=settings.N_THREAD)

args = parser.parse_args()

if __name__ == '__main__':
  evaluate(args.im0_path, args.restore_path, args.output_path,
           n_batch=args.n_batch, n_height=args.n_height, n_width=args.n_width, n_channel=args.n_channel,
           n_pyramid=args.n_pyramid, max_disparity=args.max_disparity, n_gpu=args.n_gpu, n_thread=args.n_thread)
