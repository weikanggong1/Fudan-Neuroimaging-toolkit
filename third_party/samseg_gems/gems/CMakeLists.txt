cmake_minimum_required(VERSION 3.18)
project(gems)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
find_package(ZLIB REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} interfaces)

if(NOT APPLE_ARM64 AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive -msse2 -mfpmath=sse")
endif()
if(NOT WIN32)
  add_compile_options(-Wno-inconsistent-missing-override -Wno-self-assign-field)
endif()
add_definitions(-DUSING_STATIC_ARRAY -DMAX_LOADINGS=100)

# Translation units required by the bundled GEMS bindings.
add_library(kvlGEMSCommon STATIC
  gzstream.C
  itkMGHImageIO.cxx
  itkMGHImageIOFactory.cxx
  kvlAtlasMeshAlphaDrawer.cxx
  kvlAtlasMeshCollection.cxx
  kvlAtlasMeshDeformationLBFGSOptimizer.cxx
  kvlAtlasMeshDeformationOptimizer.cxx
  kvlAtlasMeshJacobianDeterminantDrawer.cxx
  kvlAtlasMeshMultiAlphaDrawer.cxx
  kvlAtlasMeshPositionCostAndGradientCalculator.cxx
  kvlAtlasMeshProbabilityImageStatisticsCollector.cxx
  kvlAtlasMeshRasterizor.cxx
  kvlAtlasMeshSmoother.cxx
  kvlAtlasMeshStatisticsCollector.cxx
  kvlAtlasMeshToDSWbetaGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshToFrobeniusGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshToIntensityImageCostAndGradientCalculator.cxx
  kvlAtlasMeshToIntensityImageCostAndGradientCalculatorBase.cxx
  kvlAtlasMeshToIntensityImageLogDomainCostAndGradientCalculator.cxx
  kvlAtlasMeshToPointSetCostAndGradientCalculator.cxx
  kvlAtlasMeshToWishartGaussMixtureCostAndGradientCalculator.cxx
  kvlAtlasMeshValueDrawer.cxx
  kvlAverageAtlasMeshPositionCostAndGradientCalculator.cxx
  kvlCroppedImageReader.cxx
  kvlHistogrammer.cxx
  kvlMutualInformationCostAndGradientCalculator.cxx
)
target_link_libraries(kvlGEMSCommon ${ZLIB_LIBRARIES} ${ITK_LIBRARIES})
