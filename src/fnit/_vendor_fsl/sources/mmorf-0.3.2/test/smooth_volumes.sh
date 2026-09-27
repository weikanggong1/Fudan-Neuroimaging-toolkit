#!/bin/bash

fslmaths data/MNI152_T1_2mm -s 4.246 data/MNI152_T1_2mm_smooth_10mmFWHM
fslmaths data/MNI152_T1_2mm_warped -s 4.246 data/MNI152_T1_2mm_smooth_10mmFWHM_warped
fslmaths data/patient_x_flirt -s 4.246 data/patient_x_flirt_smooth_10mmFWHM

#deriv_file=data/MNI152_T1_2mm_warped_deriv
#warped_file=data/MNI152_T1_2mm_smooth_10mmFWHM_warped_deriv

#for direction in 0 1 2 ; do
#  fslmaths ${deriv_file}_${direction} -s 4.246 ${warped_file}_${direction}
#done

#deriv_file=data/MNI152_T1_2mm_deriv
#warped_file=data/MNI152_T1_2mm_smooth_10mmFWHM_deriv

#for direction in 0 1 2 ; do
#  fslmaths ${deriv_file}_${direction} -s 4.246 ${warped_file}_${direction}
#done
