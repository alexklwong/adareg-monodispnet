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

import os

TRN_REFS_DIRPATH  = 'training'
VAL_REFS_DIRPATH  = 'validation'
TST_REFS_DIRPATH  = 'testing'
# Ground truth directory paths
TRN_GTD_DIRPATH   = os.path.join(VAL_REFS_DIRPATH, 'eigen/ground_truth')
VAL_GTD_DIRPATH   = os.path.join(VAL_REFS_DIRPATH, 'eigen/ground_truth')
TST_GTD_DIRPATH   = os.path.join(TST_REFS_DIRPATH, 'eigen/ground_truth')
# Training, testing validation input filepaths
TRN_IM0_FILEPATH  = os.path.join(TRN_REFS_DIRPATH, 'eigen_trn_im0.txt')
TRN_IM1_FILEPATH  = os.path.join(TRN_REFS_DIRPATH, 'eigen_trn_im1.txt')
VAL_IM0_FILEPATH  = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_im0.txt')
VAL_IM1_FILEPATH  = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_im1.txt')
VAL_GTD_FILEPATH  = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtd.txt')
VAL_GTF_FILEPATH  = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtf.npy')
VAL_GTB_FILEPATH  = os.path.join(VAL_REFS_DIRPATH, 'eigen_val_gtb.npy')
TST_IM0_FILEPATH  = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_im0.txt')
TST_IM1_FILEPATH  = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_im1.txt')
TST_GTD_FILEPATH  = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtd.txt')
TST_GTF_FILEPATH  = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtf.npy')
TST_GTB_FILEPATH  = os.path.join(TST_REFS_DIRPATH, 'eigen_tst_gtb.npy')
# Checkpoint paths
CHECKPOINT_PATH     = 'log'
RESTORE_PATH        = ''
N_CHECKPOINT        = 500
N_SUMMARY           = 100
OUTPUT_PATH         = 'out'
# Input image dimensions
N_BATCH             = 8
N_HEIGHT            = 256
N_WIDTH             = 512
N_CHANNEL           = 3
# Network Hyperparameters
N_EPOCH             = 50
LEARNING_RATE       = 1e-4
LEARNING_RATES      = [1.5e-4, 7.5e-5, 3.75e-05]
LEARNING_RATES_TXT  = '1.5e-4, 7.5e-5, 3.75e-05'
LEARNING_BOUNDS     = [0.60, 0.80]
LEARNING_BOUNDS_TXT = '0.60, 0.80'
N_PYRAMID           = 4
MAX_DISPARITY       = 0.33
# Weights for loss function
W_PH                = 0.15
W_ST                = 0.85
W_SM                = 0.10
W_BC                = 1.05
W_AR                = 5.00
# Hardware settings
N_GPU               = 1
N_THREAD            = 8


