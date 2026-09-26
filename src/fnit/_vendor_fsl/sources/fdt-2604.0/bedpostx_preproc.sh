#!/bin/sh
#   Copyright (C) 2012 University of Oxford
#
#   SHCOPYRIGHT
subjdir=$1
gflag=$2

echo Copying files to bedpost directory
cp ${subjdir}/bvecs ${subjdir}/bvals ${subjdir}.bedpostX
${FSLDIR}/bin/imcp ${subjdir}/nodif_brain_mask ${subjdir}.bedpostX

if [ `${FSLDIR}/bin/imtest ${subjdir}/nodif` = 1 ] ; then
    ${FSLDIR}/bin/fslmaths ${subjdir}/nodif -mas ${subjdir}/nodif_brain_mask ${subjdir}.bedpostX/nodif_brain
fi

${FSLDIR}/bin/fslslice ${subjdir}/data
${FSLDIR}/bin/fslslice ${subjdir}/nodif_brain_mask
if [ ${gflag} -eq 1 ]; then
    ${FSLDIR}/bin/fslslice ${subjdir}/grad_dev
fi

echo Done
