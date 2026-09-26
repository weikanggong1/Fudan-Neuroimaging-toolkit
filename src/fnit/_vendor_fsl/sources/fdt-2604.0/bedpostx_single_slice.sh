#!/bin/sh
#   Copyright (C) 2012 University of Oxford
#
#   SHCOPYRIGHT

subjdir=$1
slice=$2
shift
shift
opts=$*

slicezp=`${FSLDIR}/bin/zeropad $slice 4`

${FSLDIR}/bin/xfibres\
 --data=$subjdir/data_slice_$slicezp\
 --mask=$subjdir/nodif_brain_mask_slice_$slicezp\
 -b $subjdir/bvals -r $subjdir/bvecs\
 --forcedir --logdir=$subjdir.bedpostX/diff_slices/data_slice_$slicezp \
 $opts  > $subjdir.bedpostX/logs/log$slicezp  && echo Done && touch $subjdir.bedpostX/logs/monitor/$slice

