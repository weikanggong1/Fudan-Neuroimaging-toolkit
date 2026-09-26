# This is the Makefile for the ptx2 project
#
# The ptx2 project provides some CPU-only libraries and executables,
# some GPU/CUDA-enabled libraries and executables, and a
# Python application (fsl_streamlines)
#
# This Makefile can be used in one of three modes:
#  - make:             Compile/install only CPU code
#  - make gpu=1:       Compile/install both CPU and GPU code
#  - make cpu=0 gpu=1: Compile/install only GPU code
#  - make cpu=0 gpu=0: Compile/install nothing
#  - make streamlines: Install the fsl_streamlines tool
#
# The common build files from the fsl/base project provide a number of
# other options for compiling CUDA projects - refer to that project
# for more details.

# Note that using gpu=1 only compiles/installs the probtrackx2_gpuX.Y
# binary - all other GPU-related (but which don't require a GPU to run)
# scripts and binaries are compiled/installed as part of the cpu=1 build.
#
include $(FSLCONFDIR)/default.mk

PROJNAME = ptx2
LIBS     = -lfsl-newmeshclass -lfsl-warpfns -lfsl-basisfield \
           -lfsl-surface -lfsl-vtkio -lfsl-meshclass -lfsl-newimage \
           -lfsl-utils -lfsl-miscmaths -lfsl-NewNifti -lfsl-giftiio \
           -lfsl-first_lib -lfsl-znz -lfsl-cprob -lfsl-utils -lexpat
XFILES   =
SCRIPTS  =

cpu ?= 1
gpu ?= 0

ifeq (${cpu}, 1)
    XFILES  += probtrackx2 surfmaths surf2surf surf2volume \
               surf_proj label2surf find_the_biggest proj_thresh \
               fdt_matrix_merge
    SCRIPTS += probtrackx2_gpu
endif

# The probtrackx2_gpu executable is labelled
# with the CUDA version it is compiled against.
ifeq (${gpu}, 1)
    XFILES  += probtrackx2_gpu
endif


# fsl_streamlines. This target needs
# to be specifically requested
streamlines:
	${FSLDIR}/bin/python -m pip install ./fsl_streamlines \
	    --prefix ${DESTDIR} --no-deps --ignore-installed \
	    --no-cache-dir -vvv


all: ${XFILES}

PTX2OBJS = probtrackx.o probtrackxOptions.o streamlines.o ptx_simple.o \
           ptx_seedmask.o ptx_nmasks.o csv.o csv_mesh.o

probtrackx2: $(PTX2OBJS)
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

find_the_biggest: find_the_biggest.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

proj_thresh: proj_thresh.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fdt_matrix_ops: fdt_matrix_ops.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fdt_matrix_split: fdt_matrix_split.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

testfile: testfile.o streamlines.o csv.o csv_mesh.o probtrackxOptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

surf_proj: surf_proj.o csv.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fdt_matrix_merge: fdt_matrix_merge.o streamlines.o csv.o csv_mesh.o probtrackxOptions.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fdt_matrix_4_to_2: fdt_matrix_4_to_2.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

surf2surf: surf2surf.o csv.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

surf2volume: surf2volume.o csv.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

label2surf: label2surf.o csv_mesh.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

surfmaths: surfmaths.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

PTX2GPUOBJS = probtrackx_gpu.o tractography_gpu.o saveResults_ptxGPU.o \
              CUDA/tractographyInput.o CUDA/tractographyData.o \
              probtrackxOptions.o csv.o csv_mesh.o

tractography_gpu.o: CUDA/tractography_gpu.cu
	$(NVCC) ${NVCCFLAGS} -c -o $@ $<

probtrackx_gpu.o: probtrackx_gpu.cc
	$(NVCC) ${NVCCFLAGS} -c -o $@ $<

probtrackx2_gpu: ${PTX2GPUOBJS}
	${NVCC} ${NVCCFLAGS} -o $@ $^ ${NVCCLDFLAGS}
