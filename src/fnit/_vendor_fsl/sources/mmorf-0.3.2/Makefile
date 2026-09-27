# Makefile for MMORF

# As we are using FSL classes, this line is needed
include ${FSLCONFDIR}/default.mk

PROJNAME = mmorf

# MMORF provides these executables. All binary
# files are directed into a sub-directory called
# "build".
XFILES       = build/mmorf build/tensor_average
BUILDDIR     = build/
CUDABUILDDIR = build/

# Where to find MMORF code.
INC_DIR = inc
SRC_DIR = src

# CUSP code directory
INC_CUSP = cusplibrary

# Danny Ruijters' Cubic Interpolation (CI) code directory
INC_CI = CubicInterpolationCUDA/code

USRINCFLAGS = -I$(INC_DIR) -I$(INC_CI) -I$(INC_CUSP)
USRCPPFLAGS = -DARMA_DONT_PRINT_ERRORS -DEXPOSE_TREACHEROUS
USRCXXFLAGS = -fopenmp
LIBS        = -lfsl-basisfield -lfsl-newimage -lfsl-miscmaths \
              -lfsl-NewNifti -lfsl-utils -lfsl-znz -lgomp
CUDALIBS    = -lcurand

OBJS = TextureHandleLinear.o \
       VolumeBSpline.o \
       VolumeTensor.o \
       WarpFieldBSpline.o \
       BiasFieldBSpline.o \
       SparseDiagonalMatrix.o \
       SparseDiagonalMatrixTiled.o \
       CostFxnDefinitions.o \
       IntensityMapperPolynomial.o \
       CostFxnSSDWarpFieldSymmetric.o \
       CostFxnSSDWarpFieldSymmetricMaskedExcluded.o \
       CostFxnSSDWarpFieldSymmetricMaskedExcludedDiagHess.o \
       CostFxnSSDBiasField.o \
       CostFxnSSDBiasFieldMaskedExcluded.o \
       CostFxnTensorL2WarpFieldSymmetric.o\
       CostFxnTensorL2WarpFieldSymmetricMaskedExcluded.o\
       CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess.o\
       CostFxnBendingEnergy.o \
       CostFxnBendingEnergyBiasField.o \
       CostFxnLogJacobian.o \
       CostFxnLogJacobianSingularValues.o \
       CostFxnLogJacobianSingularValuesDiagHess.o \
       CostFxnRegularised.o \
       CostFxnCompoundWarpField.o \
       CostFxnCompoundWarpFieldVarianceScaled.o \
       CostFxnCompoundBiasFieldVarianceScaled.o\
       OptimiserLevenbergMarquardt.o \
       OptimiserLevenberg.o \
       OptimiserScaledConjugateGradient.o \
       LinearSolverCusp.o \
       RegistrationCoordinatorMultimodal.o \
       mmorf.o
OBJS := $(OBJS:%.o=build/%.o)


all: $(XFILES)

clean:
	rm -rf build depend.mk

build/%.o: $(SRC_DIR)/%.cu
	@mkdir -p build
	${NVCC} ${NVCCFLAGS} -c -o $@ $<

build/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/tensor_average: build/tensor_average.o \
                      build/tensor_fxns.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ${LDFLAGS}

build/mmorf: $(OBJS)
	${NVCC} ${NVCCFLAGS} -o $@ $^ ${NVCCLDFLAGS}

# Print a varable name with "make print-VARIABLE"
print-%  : ; @echo $* = $($*)
