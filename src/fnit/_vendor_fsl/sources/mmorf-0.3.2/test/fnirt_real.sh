#!/bin/bash

# Perform a fnirt registration using roughly the same parameters as MMORF, i.e.
#   warp_res = 10,10,10
#   subsamp = 0,0,0

fnirt \
  --ref=./data/MNI152_T1_2mm \
  --in=./data/patient_x_flirt \
  --iout=./data/patient_x_fnirt \
  --fout=./data/fnirt_warp \
  --jout=./data/fnirt_jacob \
  --warpres=10,10,10 \
  --splineorder=3 \
  --ssqlambda=0 \
  --regmod=bending_energy \
  --subsamp=1,1,1 \
  --miter=5,5,5 \
  --reffwhm=2,1,0.5 \
  --infwhm=4,2,1 \
  --lambda=1e8,1e7,1e6 \
  --estint=0,0,0 \
  --applyrefmask=0,0,0 \
  --applyinmask=0,0,0 \
  --minmet=lm \
  --interp=spline \
  -v


