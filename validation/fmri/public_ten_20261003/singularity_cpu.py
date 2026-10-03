#!/usr/bin/env python3
import os,sys
real="/public/software/apps/singularity/4.2.2/bin/singularity"
args=sys.argv[1:]
if args and args[0]=="exec":args=["exec","--env","CUDA_VISIBLE_DEVICES=",*args[1:]]
os.execv(real,[real,*args])
