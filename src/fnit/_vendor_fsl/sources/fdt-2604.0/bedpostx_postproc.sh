#!/bin/sh
#   Copyright (C) 2012 University of Oxford
#
#   SHCOPYRIGHT
subjdir=$1

numfib=`${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_0000/f*samples* | wc -w | awk '{print $1}'`

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/f0samples` -eq 1 ];then
    numfib=$(($numfib - 1))
fi

fib=1
while [ $fib -le $numfib ]
do
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/merged_th${fib}samples `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/th${fib}samples*`
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/merged_ph${fib}samples `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/ph${fib}samples*`
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/merged_f${fib}samples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/f${fib}samples*`
    ${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/merged_th${fib}samples -Tmean ${subjdir}.bedpostX/mean_th${fib}samples
    ${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/merged_ph${fib}samples -Tmean ${subjdir}.bedpostX/mean_ph${fib}samples
    ${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/merged_f${fib}samples -Tmean ${subjdir}.bedpostX/mean_f${fib}samples

    ${FSLDIR}/bin/make_dyadic_vectors ${subjdir}.bedpostX/merged_th${fib}samples ${subjdir}.bedpostX/merged_ph${fib}samples ${subjdir}.bedpostX/nodif_brain_mask ${subjdir}.bedpostX/dyads${fib}
    if [ $fib -ge 2 ];then
	${FSLDIR}/bin/maskdyads ${subjdir}.bedpostX/dyads${fib} ${subjdir}.bedpostX/mean_f${fib}samples
	${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/mean_f${fib}samples -div ${subjdir}.bedpostX/mean_f1samples ${subjdir}.bedpostX/mean_f${fib}_f1samples
	${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/dyads${fib}_thr0.05 -mul ${subjdir}.bedpostX/mean_f${fib}_f1samples ${subjdir}.bedpostX/dyads${fib}_thr0.05_modf${fib}
	${FSLDIR}/bin/imrm ${subjdir}.bedpostX/mean_f${fib}_f1samples
    fi

    fib=$(($fib + 1))

done

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/mean_f1samples` -eq 1 ];then
    ${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/mean_f1samples -mul 0 ${subjdir}.bedpostX/mean_fsumsamples
    fib=1
    while [ $fib -le $numfib ]
    do
	${FSLDIR}/bin/fslmaths ${subjdir}.bedpostX/mean_fsumsamples -add ${subjdir}.bedpostX/mean_f${fib}samples ${subjdir}.bedpostX/mean_fsumsamples
	fib=$(($fib + 1))
    done
fi


if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_dsamples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_dsamples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_dsamples*`
fi

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_d_stdsamples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_d_stdsamples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_d_stdsamples*`
fi

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_Rsamples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_Rsamples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_Rsamples*`
fi

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_f0samples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_f0samples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_f0samples*`
fi

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_S0samples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_S0samples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_S0samples*`
fi

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/diff_slices/data_slice_0000/mean_tausamples` -eq 1 ];then
    ${FSLDIR}/bin/fslmerge -z ${subjdir}.bedpostX/mean_tausamples  `${FSLDIR}/bin/imglob ${subjdir}.bedpostX/diff_slices/data_slice_*/mean_tausamples*`
fi


echo Removing intermediate files

if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/merged_th1samples` -eq 1 ];then
  if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/merged_ph1samples` -eq 1 ];then
    if [ `${FSLDIR}/bin/imtest ${subjdir}.bedpostX/merged_f1samples` -eq 1 ];then
      rm -rf "${subjdir}.bedpostX/diff_slices"
      rm -f "${subjdir}/data_slice_"*
      rm -f "${subjdir}/nodif_brain_mask_slice_"*
      if [ `${FSLDIR}/bin/imtest ${subjdir}/grad_dev_slice_0000` -eq 1 ];then
	  rm -f "${subjdir}/grad_dev_slice_"*
      fi
    fi
  fi
fi

echo Creating identity xfm

xfmdir=${subjdir}.bedpostX/xfms
echo 1 0 0 0 > ${xfmdir}/eye.mat
echo 0 1 0 0 >> ${xfmdir}/eye.mat
echo 0 0 1 0 >> ${xfmdir}/eye.mat
echo 0 0 0 1 >> ${xfmdir}/eye.mat

echo Done
