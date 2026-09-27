include ${FSLCONFDIR}/default.mk

PROJNAME = avwutils

LIBS = -lfsl-newimage -lfsl-miscmaths -lfsl-cprob -lfsl-NewNifti -lfsl-utils -lfsl-znz

XFILES = fslorient fslstats fslmerge \
         fslpspec fslroi fslnvols fsl2ascii fslascii2img \
         fslsplit fslcc fslinterleave \
         fslhd fslcpgeom fslcreatehd fslmaths \
         fslcomplex fslfft fslmeants fslslice fslswapdim_exe \
         calc_grad_perc_dev fslsmoothfill \
         fslselectvols distancemap

SCRIPTS = fslval fslsize fslinfo fsledithd \
          avw2fsl fslswapdim fslreorient2std \
          fslmodhd fsladd

all: ${XFILES}

%: %.cc
	${CXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}

fslswapdim_exe: fslswapdim.cc
	${CXX} ${CXXFLAGS} -o $@ $< ${LDFLAGS}
