#!/bin/bash

# Calculate final difference volumes
fslmaths data/MNI152_T1_2mm_warped -sub \
  data/MNI152_T1_2mm \
  data_test_OptimiserLevenbergMarquardt/diff_pre_reg

#fslmaths data/MNI152_T1_2mm -sub \
#  data/patient_x_robust_flirt.nii.gz \
#  data_test_OptimiserLevenbergMarquardt/diff_pre_reg_real

fslmaths data/MNI152_T1_2mm_warped -sub \
  data_test_OptimiserLevenbergMarquardt/final_vol_ssd_warp_field \
  data_test_OptimiserLevenbergMarquardt/diff_ssd_warp_field

fslmaths data/MNI152_T1_2mm_warped -sub \
  data_test_OptimiserLevenbergMarquardt/final_vol_regularised \
  data_test_OptimiserLevenbergMarquardt/diff_regularised

#fslmaths data/MNI152_T1_2mm -sub \
#  data_test_OptimiserLevenbergMarquardt/final_vol_real \
#  data_test_OptimiserLevenbergMarquardt/diff_real

# Calculate final difference warps
fslmaths data/actual_warp_field -sub \
  data_test_OptimiserLevenbergMarquardt/ssd_warp_field_warp \
  data_test_OptimiserLevenbergMarquardt/diff_ssd_warp_field_warp

fslmaths data/actual_warp_field -sub \
  data_test_OptimiserLevenbergMarquardt/regularised_warp \
  data_test_OptimiserLevenbergMarquardt/diff_regularised_warp

# Display in fsleyes
fsleyes \
  data_test_OptimiserLevenbergMarquardt/ssd_warp_field_jacobian_determinants \
  -dr 0 3 \
  data_test_OptimiserLevenbergMarquardt/diff_ssd_warp_field_warp \
  -dr -10 10 \
  data_test_OptimiserLevenbergMarquardt/ssd_warp_field_warp \
  -dr -10 10 \
  data_test_OptimiserLevenbergMarquardt/diff_ssd_warp_field \
  data_test_OptimiserLevenbergMarquardt/diff_pre_reg \
  data_test_OptimiserLevenbergMarquardt/final_vol_ssd_warp_field \
  -dr 1000 9000 \
  data/MNI152_T1_2mm_warped \
  -dr 1000 9000 \
  &

fsleyes \
  data_test_OptimiserLevenbergMarquardt/regularised_jacobian_determinants \
  -dr 0 3 \
  data_test_OptimiserLevenbergMarquardt/diff_regularised_warp \
  -dr -10 10 \
  data_test_OptimiserLevenbergMarquardt/regularised_warp \
  -dr -10 10 \
  data_test_OptimiserLevenbergMarquardt/diff_regularised \
  data_test_OptimiserLevenbergMarquardt/diff_pre_reg \
  data_test_OptimiserLevenbergMarquardt/final_vol_regularised \
  -dr 1000 9000 \
  data/MNI152_T1_2mm_warped \
  -dr 1000 9000 \
  &

fsleyes \
  data_test_OptimiserLevenbergMarquardt/real_jacobian_determinants \
  data_test_OptimiserLevenbergMarquardt/real_warp \
  data_test_OptimiserLevenbergMarquardt/final_vol_real \
  -dr 0 343 \
  data/patient_x_flirt \
  -dr 0 343 \
  data/MNI152_T1_2mm \
  &
