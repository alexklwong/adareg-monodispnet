# Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction

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

## Getting Started

The following guide assumes that you are located in the root directory of this repository  
and that you have Tensorflow 1.0+ installed

Create a symbolic link to your dataset directory

```
ln -s /path/to/data/directory/containing/kitti/root/folder data
```

where `/path/to/data/directory/containing/kitti/root/folder` contains your raw KITTI dataset and KITTI 2015 Stereo benchmark

```
/path/to/data/directory/containing/kitti/root/folder/kitti_raw_data
/path/to/data/directory/containing/kitti/root/folder/kitti_stereo_flow
```

Run the KITTI data setup script to generate text files containing KITTI training and validation filepaths:

```
python setup/prep_kitti_eigen_split_data.py
python setup/prep_kitti_kitti_split_data.py
```

## Training the Monocular Disparity Network

For training on KITTI Eigen Split:

```
python src/train_monodispnet.py \
--trn_im0_path training/eigen_trn_im0.txt \
--trn_im1_path training/eigen_trn_im1.txt \
--learning_rates 1.8e-4,2.0e-4,1.0e-4,5.0e-5 \
--learning_bounds 0.01,0.90,0.95 \
--max_disparity 0.33 \
--w_ph 0.15 \
--w_st 0.85 \
--w_sm 0.10 \
--w_bc 1.05 \
--n_checkpoint 5000 \
--checkpoint_path checkpoints/eigen_model
```

For training on KITTI KITTI 2015 Split:

```
python src/train_monodispnet.py \
--trn_im0_path training/kitti_trn_im0.txt \
--trn_im1_path training/kitti_trn_im1.txt \
--learning_rates 1.8e-4,2.0e-4,1.0e-4,5.0e-5 \
--learning_bounds 0.01,0.90,0.95 \
--max_disparity 0.33 \
--w_ph 0.15 \
--w_st 0.85 \
--w_sm 0.10 \
--w_bc 1.05 \
--n_checkpoint 5000 \
--checkpoint_path checkpoints/kitti_model
```

## Evaluation on KITTI Eigen Split and KITTI 2015 Split Benchmark

Run the following script to evaluate your model:

Generating output for KITTI Eigen Split

```
python src/run_monodispnet.py \
--im0_path testing/eigen_tst_im0.txt \
--restore_path checkpoints/eigen_model/model.ckpt-000000 \
--output_path checkpoints/eigen_model/outputs \
--max_disparity 0.33
```

Evaluating KITTI Eigen Split

```
python src/evaluate_kitti.py \
--npy_path checkpoints/eigen_model/outputs/disparities.npy \
--ims_path testing/eigen_tst_im0.txt \
--gts_path testing/eigen_tst_gtd.txt \
--split eigen \
--max_depth 80
```

Generating output for KITTI KITTI 2015 Split

```
python src/run_monodispnet.py \
--im0_path testing/kitti_tst_im0.txt \
--restore_path checkpoints/kitti_model/model.ckpt-000000 \
--output_path checkpoints/kitti_model/outputs \
--max_disparity 0.33
```

Evaluating KITTI 2015 Split

```
python src/evaluate_kitti.py \
--npy_path checkpoints/kitti_model/outputs/disparities.npy \
--ims_path testing/kitti_tst_im0.txt \
--gts_path testing/kitti_tst_gtd.txt \
--split kitti
```

## Downloading Pre-trained Models 

To get the pre-trained models on Eigen and KITTI split and output disparities please visit:

```
https://tinyurl.com/y2adhhb3
```
