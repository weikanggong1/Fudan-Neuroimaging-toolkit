# As we are using FSL classes, this line is needed
include ${FSLCONFDIR}/default.mk

# Where to find user code.
USER_DIR = ..
INC_DIR = $(USER_DIR)/inc
SRC_DIR = $(USER_DIR)/src

# CUDA coda
CUDA_DIR = /usr/local/cuda
INC_CUDA = $(CUDA_DIR)/include
LIB_CUDA = $(CUDA_DIR)/lib64

# CUSP code director
CUSP_DIR = $(HOME)/Sandbox/gitlab/cusplibrary
INC_CUSP = $(CUSP_DIR)

# Danny Ruijters' Cubic Interpolation (CI) code directory
CI_DIR = $(HOME)/Sandbox/gitlab/CubicInterpolationCUDA/code
INC_CI = $(CI_DIR)

# Basisfield needed
BASISFIELD_DIR = $(FSLDIR)/src/basisfield
INC_BASISFIELD = $(BASISFIELD_DIR)

# Where to find include files
USRINCFLAGS = -I$(INC_DIR) -I$(INC_NEWMAT) -I$(INC_ZLIB) -I$(INC_CI) -I$(INC_CUSP) -I$(INC_BASISFIELD) -I$(INC_CUDA) -DEXPOSE_TREACHEROUS
USRLDFLAGS = -L$(LIB_NEWMAT)
LIBS = -lbasisfield -lnewimage -lmiscmaths -lfslio -lniftiio -lutils -lnewmat -lznz -lm -lz -lpthread

# Building CUDA related code
NVCC = nvcc
NVCCFLAGS = -c -DCOMPILE_GPU -O3 -m 64 -std=c++11 -DARMA_ALLOW_FAKE_GCC

# Debug/profiling NVCCFLAGS
NVCCFLAGS += -lineinfo -Xptxas="-dlcm=ca" #-Xcompiler "-g -pg"#-maxrregcount 32

CUDALDFLAGS = -Xlinker -rpath $(LIB_CUDA) -L$(LIB_CUDA) -lcudart
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30 #-gencode arch=compute_35,code=sm_35
GENCODE_SM50 := -gencode arch=compute_50,code=sm_50
GENCODE_SM70 := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS := $(GENCODE_SM70)
CUDALIBS = -Lcuda  -lcudart -lcuda -lcurand -L${CUDA}/lib64 -L${CUDA}/lib -L/usr/local/cuda/lib

# Flags passed to the preprocessor.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
# NB NB NB! The -no-pie option is there because of changes to the default settings in gcc 7.
# This might be a Debian-only issue
CXXFLAGS += -std=c++11 -no-pie #-pg# -Wall -Wextra -pthread

# Googletest directory
GTEST_DIR = $(HOME)/Sandbox/googletest/googletest

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS = test_Parameterswarp test_VectorArmadillo test_VolumeBSpline \
        test_EddySplineField test_IndexND test_PositionND test_WarpFieldBSpline \
        test_CostFxnSSDWarpField

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# House-keeping build targets.

all : $(TESTS)

clean :
	rm -f $(TESTS) gtest.a gtest_main.a *.o

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

sample1.o : $(USER_DIR)/sample1.cc $(USER_DIR)/sample1.h $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample1.cc

sample1_unittest.o : $(USER_DIR)/sample1_unittest.cc \
                     $(USER_DIR)/sample1.h $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample1_unittest.cc

sample1_unittest : sample1.o sample1_unittest.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

###############################################################################

EddyKernels.o: $(EDDY_DIR)/cuda/EddyKernels.h $(EDDY_DIR)/cuda/EddyKernels.cu 
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o EddyKernels.o $(EDDY_DIR)/cuda/EddyKernels.cu ${AccumulatedIncFlags}

CudaVolume.o: $(EDDY_DIR)/cuda/CudaVolume.h $(EDDY_DIR)/cuda/CudaVolume.cu $(EDDY_DIR)/cuda/EddyKernels.h \
	$(EDDY_DIR)/cuda/EddyFunctors.h $(EDDY_DIR)/EddyHelperClasses.h 
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o CudaVolume.o $(EDDY_DIR)/cuda/CudaVolume.cu ${AccumulatedIncFlags}

###############################################################################

EddyHessianDefinitions.o: $(SRC_DIR)/EddyHessianDefinitions.cu
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -o EddyHessianDefinitions.o $(SRC_DIR)/EddyHessianDefinitions.cu ${AccumulatedIncFlags}

EddySparseDiagonalMatrix.o : $(SRC_DIR)/EddySparseDiagonalMatrix.cu \
                         EddyHessianDefinitions.o \
                         $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -o EddySparseDiagonalMatrix.o $(SRC_DIR)/EddySparseDiagonalMatrix.cu ${AccumulatedIncFlags}

###############################################################################

ParametersWarp.o : $(SRC_DIR)/ParametersWarp.cpp \
                   $(INC_DIR)/MmorfHelpers.h \
                   $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/ParametersWarp.cpp

test_ParametersWarp.o : $(SRC_DIR)/test_ParametersWarp.cpp \
                        $(INC_DIR)/ParametersWarp.h \
                        $(INC_DIR)/Parameters.h \
                        $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/test_ParametersWarp.cpp

test_ParametersWarp : ParametersWarp.o test_ParametersWarp.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

###############################################################################

VectorArmadillo.o : $(SRC_DIR)/VectorArmadillo.cpp \
                    $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/VectorArmadillo.cpp

test_VectorArmadillo.o : $(SRC_DIR)/test_VectorArmadillo.cpp \
                         $(INC_DIR)/VectorArmadillo.h \
                         $(INC_DIR)/Vector.h \
                         $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/test_VectorArmadillo.cpp

test_VectorArmadillo : VectorArmadillo.o test_VectorArmadillo.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -larmadillo -lpthread $^ -o $@

###############################################################################

VolumeBSpline.o : $(SRC_DIR)/VolumeBSpline.cu \
                 $(INC_DIR)/MmorfMemory.h \
                 $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o VolumeBSpline.o $(SRC_DIR)/VolumeBSpline.cu ${AccumulatedIncFlags}

test_VolumeBSpline.o : $(SRC_DIR)/test_VolumeBSpline.cpp \
                      $(INC_DIR)/VolumeBSpline.cuh \
                      $(INC_DIR)/Volume.h \
                      $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_VolumeBSpline.cpp

test_VolumeBSpline : VolumeBSpline.o test_VolumeBSpline.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_VolumeBSpline gtest_main.a test_VolumeBSpline.o VolumeBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

VolumeTensor.o : $(SRC_DIR)/VolumeTensor.cu \
                 $(INC_DIR)/MmorfMemory.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} ${AccumulatedIncFlags} -o VolumeTensor.o $(SRC_DIR)/VolumeTensor.cu

test_VolumeTensor.o : $(SRC_DIR)/test_VolumeTensor.cpp \
                      $(INC_DIR)/VolumeTensor.cuh \
                      $(INC_DIR)/Volume.h \
                      $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_VolumeTensor.cpp

test_VolumeTensor : VolumeTensor.o test_VolumeTensor.o VolumeBSpline.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_VolumeTensor gtest_main.a test_VolumeTensor.o VolumeTensor.o VolumeBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

EddySplineField.o : $(SRC_DIR)/EddySplineField.cu \
                    $(INC_DIR)/EddySplineField.cuh \
                    $(INC_DIR)/MmorfMemory.h \
                    EddyHessianDefinitions.o \
                    EddySparseDiagonalMatrix.o \
                    $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -o EddySplineField.o $(SRC_DIR)/EddySplineField.cu ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/EddySplineField.cpp

test_EddySplineField.o : $(SRC_DIR)/test_EddySplineField.cpp \
                      $(INC_DIR)/EddySplineField.cuh \
                      $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/test_EddySplineField.cpp

test_EddySplineField : EddySplineField.o test_EddySplineField.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_EddySplineField gtest_main.a test_EddySplineField.o EddySplineField.o EddySparseDiagonalMatrix.o EddyHessianDefinitions.o $(LIBS) $(CUDALDFLAGS)
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LIBS) -lpthread $^ -o $@

###############################################################################

IndexND.o : $(SRC_DIR)/IndexND.cpp \
            $(INC_DIR)/MmorfHelpers.h \
            $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/IndexND.cpp

test_IndexND.o : $(SRC_DIR)/test_IndexND.cpp \
                 $(INC_DIR)/IndexND.h \
                 $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/test_IndexND.cpp

test_IndexND : IndexND.o test_IndexND.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

###############################################################################

PositionND.o : $(SRC_DIR)/PositionND.cpp \
            $(INC_DIR)/MmorfHelpers.h \
            $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/PositionND.cpp

test_PositionND.o : $(SRC_DIR)/test_PositionND.cpp \
                 $(INC_DIR)/PositionND.h \
                 $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -c $(SRC_DIR)/test_PositionND.cpp

test_PositionND : PositionND.o test_PositionND.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

###############################################################################

WarpFieldBSpline.o : $(SRC_DIR)/WarpFieldBSpline.cu \
                     $(INC_DIR)/MmorfMemory.h \
                     $(INC_DIR)/VolumeBSpline.cuh \
                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o WarpFieldBSpline.o $(SRC_DIR)/WarpFieldBSpline.cu ${AccumulatedIncFlags}

test_WarpFieldBSpline.o : $(SRC_DIR)/test_WarpFieldBSpline.cpp \
                          $(INC_DIR)/WarpFieldBSpline.cuh \
                          $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_WarpFieldBSpline.cpp

test_WarpFieldBSpline : WarpFieldBSpline.o test_WarpFieldBSpline.o  VolumeBSpline.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_WarpFieldBSpline gtest_main.a test_WarpFieldBSpline.o WarpFieldBSpline.o VolumeBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

SparseDiagonalMatrix.o : $(SRC_DIR)/SparseDiagonalMatrix.cu \
                         $(INC_DIR)/MmorfMemory.h \
                         $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o SparseDiagonalMatrix.o $(SRC_DIR)/SparseDiagonalMatrix.cu ${AccumulatedIncFlags}

#test_SparseDiagonalMatrix.o : $(SRC_DIR)/test_SparseDiagonalMatrix.cpp \
#                      $(INC_DIR)/SparseDiagonalMatrix.cuh \
#                      $(INC_DIR)/Volume.h \
#                      $(GTEST_HEADERS)
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_SparseDiagonalMatrix.cpp

#test_SparseDiagonalMatrix : SparseDiagonalMatrix.o test_SparseDiagonalMatrix.o gtest_main.a
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_SparseDiagonalMatrix gtest_main.a test_SparseDiagonalMatrix.o SparseDiagonalMatrix.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

SparseDiagonalMatrixTiled.o : $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                              $(SRC_DIR)/SparseDiagonalMatrixTiled.cu \
                              $(INC_DIR)/MmorfMemory.h \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o SparseDiagonalMatrixTiled.o $(SRC_DIR)/SparseDiagonalMatrixTiled.cu ${AccumulatedIncFlags}

#test_SparseDiagonalMatrixTiled.o : $(SRC_DIR)/test_SparseDiagonalMatrixTiled.cpp \
#                      $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
#                      $(INC_DIR)/Volume.h \
#                      $(GTEST_HEADERS)
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_SparseDiagonalMatrixTiled.cpp

#test_SparseDiagonalMatrixTiled : SparseDiagonalMatrixTiled.o test_SparseDiagonalMatrixTiled.o gtest_main.a
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_SparseDiagonalMatrixTiled gtest_main.a test_SparseDiagonalMatrixTiled.o SparseDiagonalMatrixTiled.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

TextureHandleLinear.o : $(SRC_DIR)/TextureHandleLinear.cu \
                         $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o TextureHandleLinear.o $(SRC_DIR)/TextureHandleLinear.cu ${AccumulatedIncFlags}

###############################################################################

CostFxnDefinitions.o: $(SRC_DIR)/CostFxnDefinitions.cu
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -o CostFxnDefinitions.o $(SRC_DIR)/CostFxnDefinitions.cu ${AccumulatedIncFlags}

###############################################################################

CostFxnSSDWarpField.o : $(SRC_DIR)/CostFxnSSDWarpField.cu \
                        $(INC_DIR)/CostFxnKernels.cuh \
                        $(INC_DIR)/MmorfMemory.h \
                        $(INC_DIR)/VolumeBSpline.cuh \
                        $(INC_DIR)/WarpFieldBSpline.cuh \
                        $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                        $(INC_DIR)/TextureHandleLinear.cuh \
                        $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnSSDWarpField.o $(SRC_DIR)/CostFxnSSDWarpField.cu ${AccumulatedIncFlags}

test_CostFxnSSDWarpField.o : $(SRC_DIR)/test_CostFxnSSDWarpField.cpp \
                             $(INC_DIR)/CostFxnSSDWarpField.cuh \
                             $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnSSDWarpField.o $(SRC_DIR)/test_CostFxnSSDWarpField.cpp ${AccumulatedIncFlags}

test_CostFxnSSDWarpField : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnSSDWarpField.o test_CostFxnSSDWarpField.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnSSDWarpField gtest_main.a test_CostFxnSSDWarpField.o CostFxnSSDWarpField.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnSSDWarpFieldMasked.o : $(SRC_DIR)/CostFxnSSDWarpFieldMasked.cu \
                              $(INC_DIR)/CostFxnKernels.cuh \
                              $(INC_DIR)/MmorfMemory.h \
                              $(INC_DIR)/VolumeBSpline.cuh \
                              $(INC_DIR)/WarpFieldBSpline.cuh \
                              $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                              $(INC_DIR)/TextureHandleLinear.cuh \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnSSDWarpFieldMasked.o $(SRC_DIR)/CostFxnSSDWarpFieldMasked.cu ${AccumulatedIncFlags}

test_CostFxnSSDWarpFieldMasked.o : $(SRC_DIR)/test_CostFxnSSDWarpFieldMasked.cpp \
                                   $(INC_DIR)/CostFxnSSDWarpFieldMasked.cuh \
                                   $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnSSDWarpFieldMasked.o $(SRC_DIR)/test_CostFxnSSDWarpFieldMasked.cpp ${AccumulatedIncFlags}

test_CostFxnSSDWarpFieldMasked : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnSSDWarpFieldMasked.o test_CostFxnSSDWarpFieldMasked.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnSSDWarpFieldMasked gtest_main.a test_CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMasked.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnTensorL2WarpFieldMasked.o : $(SRC_DIR)/CostFxnTensorL2WarpFieldMasked.cu \
                                   $(INC_DIR)/CostFxnKernels.cuh \
                                   $(INC_DIR)/MmorfMemory.h \
                                   $(INC_DIR)/VolumeTensor.cuh \
                                   $(INC_DIR)/WarpFieldBSpline.cuh \
                                   $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                                   $(INC_DIR)/TextureHandleLinear.cuh \
                                   $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnTensorL2WarpFieldMasked.o $(SRC_DIR)/CostFxnTensorL2WarpFieldMasked.cu ${AccumulatedIncFlags}

test_CostFxnTensorL2WarpFieldMasked.o : $(SRC_DIR)/test_CostFxnTensorL2WarpFieldMasked.cpp \
                                        $(INC_DIR)/CostFxnTensorL2WarpFieldMasked.cuh \
                                        $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnTensorL2WarpFieldMasked.o $(SRC_DIR)/test_CostFxnTensorL2WarpFieldMasked.cpp ${AccumulatedIncFlags}

test_CostFxnTensorL2WarpFieldMasked : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnTensorL2WarpFieldMasked.o test_CostFxnTensorL2WarpFieldMasked.o WarpFieldBSpline.o VolumeBSpline.o VolumeTensor.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnTensorL2WarpFieldMasked gtest_main.a test_CostFxnTensorL2WarpFieldMasked.o CostFxnTensorL2WarpFieldMasked.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeTensor.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnSSDWarpFieldMaskedDiagHess.o : $(SRC_DIR)/CostFxnSSDWarpFieldMaskedDiagHess.cu \
                                      $(INC_DIR)/CostFxnKernels.cuh \
                                      $(INC_DIR)/MmorfMemory.h \
                                      $(INC_DIR)/VolumeBSpline.cuh \
                                      $(INC_DIR)/WarpFieldBSpline.cuh \
                                      $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                                      $(INC_DIR)/TextureHandleLinear.cuh
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnSSDWarpFieldMaskedDiagHess.o $(SRC_DIR)/CostFxnSSDWarpFieldMaskedDiagHess.cu ${AccumulatedIncFlags}

test_CostFxnSSDWarpFieldMaskedDiagHess.o : $(SRC_DIR)/test_CostFxnSSDWarpFieldMaskedDiagHess.cpp \
                                           $(INC_DIR)/CostFxnSSDWarpFieldMaskedDiagHess.cuh \
                                           $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnSSDWarpFieldMaskedDiagHess.o $(SRC_DIR)/test_CostFxnSSDWarpFieldMaskedDiagHess.cpp ${AccumulatedIncFlags}

test_CostFxnSSDWarpFieldMaskedDiagHess : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o test_CostFxnSSDWarpFieldMaskedDiagHess.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnSSDWarpFieldMaskedDiagHess gtest_main.a test_CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnSSDWarpFieldMasked.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnBendingEnergy.o : $(SRC_DIR)/CostFxnBendingEnergy.cu \
                         $(INC_DIR)/CostFxnKernels.cuh \
                         $(INC_DIR)/MmorfMemory.h \
                         $(INC_DIR)/WarpFieldBSpline.cuh \
                         $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                         $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnBendingEnergy.o $(SRC_DIR)/CostFxnBendingEnergy.cu ${AccumulatedIncFlags}

test_CostFxnBendingEnergy.o : $(SRC_DIR)/test_CostFxnBendingEnergy.cpp \
                              $(INC_DIR)/CostFxnBendingEnergy.cuh \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnBendingEnergy.o $(SRC_DIR)/test_CostFxnBendingEnergy.cpp ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_CostFxnBendingEnergy.cpp

test_CostFxnBendingEnergy : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnBendingEnergy.o test_CostFxnBendingEnergy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnBendingEnergy gtest_main.a test_CostFxnBendingEnergy.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnLogJacobian.o : $(SRC_DIR)/CostFxnLogJacobian.cu \
                       $(INC_DIR)/CostFxnKernels.cuh \
                       $(INC_DIR)/MmorfMemory.h \
                       $(INC_DIR)/VolumeBSpline.cuh \
                       $(INC_DIR)/WarpFieldBSpline.cuh \
                       $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                       $(INC_DIR)/TextureHandleLinear.cuh \
                       $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnLogJacobian.o $(SRC_DIR)/CostFxnLogJacobian.cu ${AccumulatedIncFlags}

test_CostFxnLogJacobian.o : $(SRC_DIR)/test_CostFxnLogJacobian.cpp \
                            $(INC_DIR)/CostFxnLogJacobian.cuh \
                            $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobian.o $(SRC_DIR)/test_CostFxnLogJacobian.cpp ${AccumulatedIncFlags}

test_CostFxnLogJacobian : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobian.o test_CostFxnLogJacobian.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobian gtest_main.a test_CostFxnLogJacobian.o CostFxnLogJacobian.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnLogJacobianSingularValues.o : $(SRC_DIR)/CostFxnLogJacobianSingularValues.cu \
                                     $(INC_DIR)/CostFxnKernels.cuh \
                                     $(INC_DIR)/MmorfMemory.h \
                                     $(INC_DIR)/VolumeBSpline.cuh \
                                     $(INC_DIR)/WarpFieldBSpline.cuh \
                                     $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                                     $(INC_DIR)/TextureHandleLinear.cuh \
                                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnLogJacobianSingularValues.o $(SRC_DIR)/CostFxnLogJacobianSingularValues.cu ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValues.o : $(SRC_DIR)/test_CostFxnLogJacobianSingularValues.cpp \
                                          $(INC_DIR)/CostFxnLogJacobianSingularValues.cuh \
                                          $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobianSingularValues.o $(SRC_DIR)/test_CostFxnLogJacobianSingularValues.cpp ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValues : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobianSingularValues.o test_CostFxnLogJacobianSingularValues.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobianSingularValues gtest_main.a test_CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValues.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnLogJacobianSingularValuesDiagHess.o : $(SRC_DIR)/CostFxnLogJacobianSingularValuesDiagHess.cu \
                                             $(INC_DIR)/CostFxnKernels.cuh \
                                             $(INC_DIR)/MmorfMemory.h \
                                             $(INC_DIR)/VolumeBSpline.cuh \
                                             $(INC_DIR)/WarpFieldBSpline.cuh \
                                             $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                                             $(INC_DIR)/TextureHandleLinear.cuh
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnLogJacobianSingularValuesDiagHess.o $(SRC_DIR)/CostFxnLogJacobianSingularValuesDiagHess.cu ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValuesDiagHess.o : $(SRC_DIR)/test_CostFxnLogJacobianSingularValuesDiagHess.cpp \
                                                  $(INC_DIR)/CostFxnLogJacobianSingularValuesDiagHess.cuh \
                                                  $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobianSingularValuesDiagHess.o $(SRC_DIR)/test_CostFxnLogJacobianSingularValuesDiagHess.cpp ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValuesDiagHess : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o test_CostFxnLogJacobianSingularValuesDiagHess.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobianSingularValuesDiagHess gtest_main.a test_CostFxnLogJacobianSingularValuesDiagHess.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnLogJacobianSingularValues.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnLogJacobianSingularValuesExact.o : $(SRC_DIR)/CostFxnLogJacobianSingularValuesExact.cu \
                                     $(INC_DIR)/CostFxnKernels.cuh \
                                     $(INC_DIR)/MmorfMemory.h \
                                     $(INC_DIR)/VolumeBSpline.cuh \
                                     $(INC_DIR)/WarpFieldBSpline.cuh \
                                     $(INC_DIR)/SparseDiagonalMatrixTiled.cuh \
                                     $(INC_DIR)/TextureHandleLinear.cuh \
                                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnLogJacobianSingularValuesExact.o $(SRC_DIR)/CostFxnLogJacobianSingularValuesExact.cu ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValuesExact.o : $(SRC_DIR)/test_CostFxnLogJacobianSingularValuesExact.cpp \
                                          $(INC_DIR)/CostFxnLogJacobianSingularValuesExact.cuh \
                                          $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobianSingularValuesExact.o $(SRC_DIR)/test_CostFxnLogJacobianSingularValuesExact.cpp ${AccumulatedIncFlags}

test_CostFxnLogJacobianSingularValuesExact : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobianSingularValuesExact.o test_CostFxnLogJacobianSingularValuesExact.o WarpFieldBSpline.o VolumeBSpline.o IntensityMapperPolynomial.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobianSingularValuesExact gtest_main.a test_CostFxnLogJacobianSingularValuesExact.o CostFxnLogJacobianSingularValuesExact.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o IntensityMapperPolynomial.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_CostFxnBendingEnergy_consistancy.o : $(SRC_DIR)/test_CostFxnBendingEnergy_consistancy.cpp \
                              $(INC_DIR)/CostFxnBendingEnergy.cuh \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnBendingEnergy_consistancy.o $(SRC_DIR)/test_CostFxnBendingEnergy_consistancy.cpp ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_CostFxnBendingEnergy_consistancy.cpp

test_CostFxnBendingEnergy_consistancy : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnBendingEnergy.o test_CostFxnBendingEnergy_consistancy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnBendingEnergy_consistancy gtest_main.a test_CostFxnBendingEnergy_consistancy.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_CostFxnLogJacobian_consistancy.o : $(SRC_DIR)/test_CostFxnLogJacobian_consistancy.cpp \
                                        $(INC_DIR)/CostFxnLogJacobian.cuh \
                                        $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobian_consistancy.o $(SRC_DIR)/test_CostFxnLogJacobian_consistancy.cpp ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_CostFxnLogJacobian_consistancy.cpp

test_CostFxnLogJacobian_consistancy : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobian.o test_CostFxnLogJacobian_consistancy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobian_consistancy gtest_main.a test_CostFxnLogJacobian_consistancy.o CostFxnLogJacobian.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_CostFxnLogJacobianSingularValues_consistancy.o : $(SRC_DIR)/test_CostFxnLogJacobianSingularValues_consistancy.cpp \
                              $(INC_DIR)/CostFxnLogJacobianSingularValues.cuh \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnLogJacobianSingularValues_consistancy.o $(SRC_DIR)/test_CostFxnLogJacobianSingularValues_consistancy.cpp ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_CostFxnLogJacobianSingularValues_consistancy.cpp

test_CostFxnLogJacobianSingularValues_consistancy : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnLogJacobianSingularValues.o test_CostFxnLogJacobianSingularValues_consistancy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnLogJacobianSingularValues_consistancy gtest_main.a test_CostFxnLogJacobianSingularValues_consistancy.o CostFxnLogJacobianSingularValues.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnRegularised.o : $(SRC_DIR)/CostFxnRegularised.cu \
                       $(INC_DIR)/MmorfMemory.h \
                       $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnRegularised.o $(SRC_DIR)/CostFxnRegularised.cu ${AccumulatedIncFlags}

test_CostFxnRegularised.o : $(SRC_DIR)/test_CostFxnRegularised.cpp \
                            $(INC_DIR)/CostFxnRegularised.cuh \
                            $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnRegularised.o $(SRC_DIR)/test_CostFxnRegularised.cpp ${AccumulatedIncFlags}

test_CostFxnRegularised : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnRegularised.o test_CostFxnRegularised.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnRegularised gtest_main.a test_CostFxnRegularised.o CostFxnRegularised.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnCompoundWarpField.o : $(SRC_DIR)/CostFxnCompoundWarpField.cu \
                             $(INC_DIR)/MmorfMemory.h \
                             $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnCompoundWarpField.o $(SRC_DIR)/CostFxnCompoundWarpField.cu ${AccumulatedIncFlags}

test_CostFxnCompoundWarpField.o : $(SRC_DIR)/test_CostFxnCompoundWarpField.cpp \
                                  $(INC_DIR)/CostFxnCompoundWarpField.cuh \
                                  $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnCompoundWarpField.o $(SRC_DIR)/test_CostFxnCompoundWarpField.cpp ${AccumulatedIncFlags}

test_CostFxnCompoundWarpField : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnCompoundWarpField.o test_CostFxnCompoundWarpField.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnCompoundWarpField gtest_main.a test_CostFxnCompoundWarpField.o CostFxnCompoundWarpField.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

CostFxnCompoundWarpFieldVarianceScaled.o : $(SRC_DIR)/CostFxnCompoundWarpFieldVarianceScaled.cu \
                             $(INC_DIR)/MmorfMemory.h \
                             $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o CostFxnCompoundWarpFieldVarianceScaled.o $(SRC_DIR)/CostFxnCompoundWarpFieldVarianceScaled.cu ${AccumulatedIncFlags}

test_CostFxnCompoundWarpFieldVarianceScaled.o : $(SRC_DIR)/test_CostFxnCompoundWarpFieldVarianceScaled.cpp \
                                  $(INC_DIR)/CostFxnCompoundWarpFieldVarianceScaled.cuh \
                                  $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_CostFxnCompoundWarpFieldVarianceScaled.o $(SRC_DIR)/test_CostFxnCompoundWarpFieldVarianceScaled.cpp ${AccumulatedIncFlags}

test_CostFxnCompoundWarpFieldVarianceScaled : SparseDiagonalMatrixTiled.o TextureHandleLinear.o CostFxnCompoundWarpFieldVarianceScaled.o test_CostFxnCompoundWarpFieldVarianceScaled.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_CostFxnCompoundWarpFieldVarianceScaled gtest_main.a test_CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_BasicOptimisation.o : $(SRC_DIR)/test_BasicOptimisation.cpp \
                           $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_BasicOptimisation.o $(SRC_DIR)/test_BasicOptimisation.cpp ${AccumulatedIncFlags}
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(INC_DIR) -I$(CI_DIR) -c $(SRC_DIR)/test_BasicOptimisation.cpp

test_BasicOptimisation : CostFxnSSDWarpField.o CostFxnBendingEnergy.o SparseDiagonalMatrix.o test_BasicOptimisation.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_BasicOptimisation gtest_main.a test_BasicOptimisation.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrix.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

create_test_data.o: $(SRC_DIR)/create_test_data.cpp
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o create_test_data.o $(SRC_DIR)/create_test_data.cpp ${AccumulatedIncFlags}

create_test_data : create_test_data.o WarpFieldBSpline.o VolumeBSpline.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o create_test_data gtest_main.a create_test_data.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

create_test_data_smoothed.o: $(SRC_DIR)/create_test_data_smoothed.cpp
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o create_test_data_smoothed.o $(SRC_DIR)/create_test_data_smoothed.cpp ${AccumulatedIncFlags}

create_test_data_smoothed : create_test_data_smoothed.o WarpFieldBSpline.o VolumeBSpline.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o create_test_data_smoothed gtest_main.a create_test_data_smoothed.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

OptimiserLevenbergMarquardt.o : $(SRC_DIR)/OptimiserLevenbergMarquardt.cu \
                                $(INC_DIR)/MmorfMemory.h \
                                $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o OptimiserLevenbergMarquardt.o $(SRC_DIR)/OptimiserLevenbergMarquardt.cu ${AccumulatedIncFlags}

test_OptimiserLevenbergMarquardt.o : $(SRC_DIR)/test_OptimiserLevenbergMarquardt.cpp \
                                     $(INC_DIR)/OptimiserLevenbergMarquardt.cuh \
                                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_OptimiserLevenbergMarquardt.o $(SRC_DIR)/test_OptimiserLevenbergMarquardt.cpp ${AccumulatedIncFlags}

test_OptimiserLevenbergMarquardt : LinearSolverCusp.o SparseDiagonalMatrixTiled.o OptimiserLevenbergMarquardt.o test_OptimiserLevenbergMarquardt.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o CostFxnSSDWarpField.o TextureHandleLinear.o CostFxnBendingEnergy.o CostFxnRegularised.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_OptimiserLevenbergMarquardt gtest_main.a test_OptimiserLevenbergMarquardt.o OptimiserLevenbergMarquardt.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnRegularised.o CostFxnDefinitions.o LinearSolverCusp.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

OptimiserLevenberg.o : $(SRC_DIR)/OptimiserLevenberg.cu \
                                $(INC_DIR)/MmorfMemory.h \
                                $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o OptimiserLevenberg.o $(SRC_DIR)/OptimiserLevenberg.cu ${AccumulatedIncFlags}

test_OptimiserLevenberg.o : $(SRC_DIR)/test_OptimiserLevenberg.cpp \
                            $(INC_DIR)/OptimiserLevenberg.cuh \
                            $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_OptimiserLevenberg.o $(SRC_DIR)/test_OptimiserLevenberg.cpp ${AccumulatedIncFlags}

test_OptimiserLevenberg : LinearSolverCusp.o SparseDiagonalMatrixTiled.o OptimiserLevenberg.o test_OptimiserLevenberg.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o CostFxnSSDWarpField.o TextureHandleLinear.o CostFxnBendingEnergy.o CostFxnRegularised.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_OptimiserLevenberg gtest_main.a test_OptimiserLevenberg.o OptimiserLevenberg.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnRegularised.o CostFxnDefinitions.o LinearSolverCusp.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

OptimiserScaledConjugateGradient.o : $(SRC_DIR)/OptimiserScaledConjugateGradient.cu \
                                     $(INC_DIR)/MmorfMemory.h \
                                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o OptimiserScaledConjugateGradient.o $(SRC_DIR)/OptimiserScaledConjugateGradient.cu ${AccumulatedIncFlags}

test_OptimiserScaledConjugateGradient.o : $(SRC_DIR)/test_OptimiserScaledConjugateGradient.cpp \
                                          $(INC_DIR)/OptimiserScaledConjugateGradient.cuh \
                                          $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_OptimiserScaledConjugateGradient.o $(SRC_DIR)/test_OptimiserScaledConjugateGradient.cpp ${AccumulatedIncFlags}

test_OptimiserScaledConjugateGradient : LinearSolverCusp.o SparseDiagonalMatrixTiled.o OptimiserScaledConjugateGradient.o test_OptimiserScaledConjugateGradient.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o CostFxnSSDWarpField.o TextureHandleLinear.o CostFxnBendingEnergy.o CostFxnLogJacobianSingularValues.o CostFxnRegularised.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_OptimiserScaledConjugateGradient gtest_main.a test_OptimiserScaledConjugateGradient.o OptimiserScaledConjugateGradient.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnLogJacobianSingularValues.o CostFxnRegularised.o CostFxnDefinitions.o LinearSolverCusp.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

LinearSolverArmadillo.o : $(SRC_DIR)/LinearSolverArmadillo.cu \
                          $(INC_DIR)/LinearSolverArmadillo.cuh \
                          $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o LinearSolverArmadillo.o $(SRC_DIR)/LinearSolverArmadillo.cu ${AccumulatedIncFlags}

test_LinearSolverArmadillo.o : $(SRC_DIR)/test_LinearSolverArmadillo.cu \
                               $(INC_DIR)/LinearSolverArmadillo.cuh \
                               $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_LinearSolverArmadillo.o $(SRC_DIR)/test_LinearSolverArmadillo.cu ${AccumulatedIncFlags}

test_LinearSolverArmadillo : LinearSolverArmadillo.o test_LinearSolverArmadillo.o CostFxnSSDWarpField.o CostFxnRegularised.o CostFxnBendingEnergy.o SparseDiagonalMatrixTiled.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o TextureHandleLinear.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_LinearSolverArmadillo gtest_main.a test_LinearSolverArmadillo.o LinearSolverArmadillo.o CostFxnRegularised.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

LinearSolverCusp.o : $(SRC_DIR)/LinearSolverCusp.cu \
                     $(INC_DIR)/LinearSolverCusp.cuh \
                     $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o LinearSolverCusp.o $(SRC_DIR)/LinearSolverCusp.cu ${AccumulatedIncFlags}

test_LinearSolverCusp.o : $(SRC_DIR)/test_LinearSolverCusp.cu \
                          $(INC_DIR)/LinearSolverCusp.cuh \
                          $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_LinearSolverCusp.o $(SRC_DIR)/test_LinearSolverCusp.cu ${AccumulatedIncFlags}

test_LinearSolverCusp : LinearSolverCusp.o test_LinearSolverCusp.o CostFxnSSDWarpField.o CostFxnRegularised.o CostFxnBendingEnergy.o SparseDiagonalMatrixTiled.o WarpFieldBSpline.o VolumeBSpline.o CostFxnDefinitions.o TextureHandleLinear.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_LinearSolverCusp gtest_main.a test_LinearSolverCusp.o LinearSolverCusp.o CostFxnRegularised.o CostFxnSSDWarpField.o CostFxnBendingEnergy.o CostFxnDefinitions.o SparseDiagonalMatrixTiled.o TextureHandleLinear.o VolumeBSpline.o WarpFieldBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

RegistrationCoordinator.o : $(SRC_DIR)/RegistrationCoordinator.cu \
                            $(INC_DIR)/RegistrationCoordinator.cuh \
                            $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o RegistrationCoordinator.o $(SRC_DIR)/RegistrationCoordinator.cu ${AccumulatedIncFlags}

test_RegistrationCoordinator.o : $(SRC_DIR)/test_RegistrationCoordinator.cpp \
                                 $(INC_DIR)/RegistrationCoordinator.cuh \
                                 $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_RegistrationCoordinator.o $(SRC_DIR)/test_RegistrationCoordinator.cpp ${AccumulatedIncFlags}

test_RegistrationCoordinator : RegistrationCoordinator.o test_RegistrationCoordinator.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_RegistrationCoordinator RegistrationCoordinator.o test_RegistrationCoordinator.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

RegistrationCoordinatorTensor.o : $(SRC_DIR)/RegistrationCoordinatorTensor.cu \
                            $(INC_DIR)/RegistrationCoordinatorTensor.cuh \
                            $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o RegistrationCoordinatorTensor.o $(SRC_DIR)/RegistrationCoordinatorTensor.cu ${AccumulatedIncFlags}

test_RegistrationTensor.o : $(SRC_DIR)/test_RegistrationTensor.cpp \
                                 $(INC_DIR)/RegistrationCoordinatorTensor.cuh \
                                 $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_RegistrationTensor.o $(SRC_DIR)/test_RegistrationTensor.cpp ${AccumulatedIncFlags}

test_RegistrationTensor : RegistrationCoordinatorTensor.o test_RegistrationTensor.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o CostFxnTensorL2WarpFieldMasked.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeTensor.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_RegistrationTensor RegistrationCoordinatorTensor.o test_RegistrationTensor.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o CostFxnTensorL2WarpFieldMasked.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeTensor.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_RegistrationToMNI.o : $(SRC_DIR)/test_RegistrationToMNI.cpp \
                                 $(INC_DIR)/RegistrationCoordinator.cuh \
                                 $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_RegistrationToMNI.o $(SRC_DIR)/test_RegistrationToMNI.cpp ${AccumulatedIncFlags}

test_RegistrationToMNI : RegistrationCoordinator.o test_RegistrationToMNI.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_RegistrationToMNI RegistrationCoordinator.o test_RegistrationToMNI.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

test_RegistrationMICCAI.o : $(SRC_DIR)/test_RegistrationMICCAI.cpp \
                                 $(INC_DIR)/RegistrationCoordinator.cuh \
                                 $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_RegistrationMICCAI.o $(SRC_DIR)/test_RegistrationMICCAI.cpp ${AccumulatedIncFlags}

test_RegistrationMICCAI : RegistrationCoordinator.o test_RegistrationMICCAI.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_RegistrationMICCAI RegistrationCoordinator.o test_RegistrationMICCAI.o CostFxnCompoundWarpFieldVarianceScaled.o CostFxnCompoundWarpField.o OptimiserLevenberg.o OptimiserLevenbergMarquardt.o OptimiserScaledConjugateGradient.o LinearSolverCusp.o CostFxnLogJacobianSingularValues.o CostFxnLogJacobianSingularValuesDiagHess.o CostFxnBendingEnergy.o CostFxnSSDWarpFieldMasked.o CostFxnSSDWarpFieldMaskedDiagHess.o CostFxnDefinitions.o TextureHandleLinear.o WarpFieldBSpline.o SparseDiagonalMatrixTiled.o VolumeBSpline.o IntensityMapperPolynomial.o gtest_main.a -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

IntensityMapperPolynomial.o : $(SRC_DIR)/IntensityMapperPolynomial.cu \
                              $(INC_DIR)/IntensityMapperPolynomial.cuh \
                              $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -o IntensityMapperPolynomial.o $(SRC_DIR)/IntensityMapperPolynomial.cu ${AccumulatedIncFlags}

test_IntensityMapperPolynomial.o : $(SRC_DIR)/test_IntensityMapperPolynomial.cpp \
                                   $(INC_DIR)/IntensityMapperPolynomial.cuh \
                                   $(GTEST_HEADERS)
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -I$(INC_DIR) -I$(CI_DIR) -I$(GTEST_DIR)/include -o test_IntensityMapperPolynomial.o $(SRC_DIR)/test_IntensityMapperPolynomial.cpp ${AccumulatedIncFlags}

test_IntensityMapperPolynomial : IntensityMapperPolynomial.o test_IntensityMapperPolynomial.o VolumeBSpline.o TextureHandleLinear.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -L$(LIB_NEWMAT) -o test_IntensityMapperPolynomial gtest_main.a test_IntensityMapperPolynomial.o IntensityMapperPolynomial.o TextureHandleLinear.o VolumeBSpline.o -lcudart -lcudadevrt -larmadillo $(LIBS) $(CUDALDFLAGS)

###############################################################################

# Print a varable name with "make print-VARIABLE"
print-%  : ; @echo $* = $($*)
