# This is the Makefile for the fdt project
#
# The FDT project provides some CPU-only libraries and executables,
# and some GPU/CUDA-enabled libraries and executables.
#
# This Makefile can be used in one of three modes:
#  - make:             Compile/install only CPU code
#  - make gpu=1:       Compile/install both CPU and GPU code
#  - make cpu=0 gpu=1: Compile/install only GPU code
#  - make cpu=0 gpu=0: Compile/install nothing
#
# A CUDA compiler and toolkit must be installed in order to compile
# the GPU components.
#
# Note that using gpu=1 only compiles/installs the xfibres_gpuX.Y
# and libfsl-bedpostx_cudaX.Y.so binaries - all other GPU-related
# scripts and binaries are compiled/ installed as part of the
# cpu=1 build.
#

include $(FSLCONFDIR)/default.mk

PROJNAME = fdt
LIBS     = -lfsl-warpfns -lfsl-basisfield -lfsl-meshclass \
           -lfsl-newimage -lfsl-miscmaths -lfsl-NewNifti \
           -lfsl-utils -lfsl-znz -lfsl-cprob
CUDALIBS = -lcurand
SCRIPTS  =
FSCRIPTS =
XFILES   =
FXFILES  =
RUNTCLS  =
SOFILES  =

cpu ?= 1
gpu ?= 0

# The gpu=1 flag only compiles/installs CUDA binaries
# (specifically xfibres_gpu<X.Y>). All other scripts/
# binaries are compiled/installed as part of the cpu=1
# build.
ifeq (${cpu}, 1)
    SCRIPTS  += eddy_correct zeropad maskdyads probtrack fdt_rotate_bvecs \
                select_dwi_vols bedpost bedpostx bedpostx_postproc.sh \
                bedpostx_preproc.sh bedpostx_single_slice.sh \
                bedpostx_datacheck CUDA/bedpostx_gpu \
                CUDA/bedpostx_postproc_gpu.sh xfibres_gpu
    FSCRIPTS += correct_and_average ocmr_preproc
    XFILES   += dtifit ccops medianfilter make_dyadic_vectors vecreg \
                xfibres probtrackx pvmfit dtigen eddy_combine \
                merge_parts_gpu CUDA/split_parts_gpu
    FXFILES  += reord_OM sausages replacevols fdt_matrix_ops indexer \
                rearrange xfibres_pred
    RUNTCLS  += Fdt
endif

ifeq ($(gpu), 1)
	XFILES  += xfibres_gpu${CUDA_VER}
    SOFILES += libfsl-bedpostx_cuda${CUDA_VER}.so
endif


all: ${XFILES} ${FXFILES} ${SOFILES}

%: %.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

ccops: ccops.o ccopsOptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

probtrackx: probtrackx.o probtrackxOptions.o streamlines.o ptx_simple.o ptx_seedmask.o ptx_twomasks.o ptx_nmasks.o ptx_meshmask.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

dtifit: dtifit.o dtifitOptions.o diffmodels.o Bingham_Watson_approx.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

xfibres: xfibres.o xfibresoptions.o diffmodels.o Bingham_Watson_approx.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

xfibres_2: xfibres_2.o xfibresoptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

kurtosis: kurtosis.o dtifitOptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

pvmfit: pvmfit.o pvmfitOptions.o diffmodels.o Bingham_Watson_approx.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

basgen: basgen.o diffmodels.o Bingham_Watson_approx.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

rubix: rubix.o diffmodels.o rubixvox.o rubixoptions.o Bingham_Watson_approx.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

libfsl-bedpostx_cuda${CUDA_VER}.so:
	${NVCC} --shared -o $@ \
	  ${NVCCFLAGS} -ICUDA -ICUDA/options \
      CUDA/init_gpu.cu CUDA/samples.cu CUDA/diffmodels.cu \
      CUDA/runmcmc.cu CUDA/xfibres_gpu.cu ${NVCCLDFLAGS}

merge_parts_gpu: merge_parts_gpu.o xfibresoptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

CUDA/split_parts_gpu: CUDA/split_parts_gpu.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

XFIBRES_OBJS = xfibres_gpu.o xfibresoptions.o diffmodels.o \
               Bingham_Watson_approx.o

xfibres_gpu${CUDA_VER}: libfsl-bedpostx_cuda${CUDA_VER}.so ${XFIBRES_OBJS}
	${NVCC} ${NVCCFLAGS} -o $@ ${XFIBRES_OBJS} \
        -lfsl-bedpostx_cuda${CUDA_VER} ${NVCCLDFLAGS}
