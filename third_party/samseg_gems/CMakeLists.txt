cmake_minimum_required(VERSION 3.18)
project(fnit_gems_native LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(ITK REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(gems)
add_subdirectory(samseg/cxx)
set_target_properties(gemsbindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${FNIT_GEMS_OUTPUT_DIR}")

pybind11_add_module(gems_resample native_spline.cpp)
target_include_directories(gems_resample PRIVATE ${ITK_INCLUDE_DIRS})
target_link_libraries(gems_resample PRIVATE ${ITK_LIBRARIES})
set_target_properties(gems_resample PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${FNIT_GEMS_OUTPUT_DIR}")

pybind11_add_module(gems_warp native_warp.cpp)
target_compile_options(gems_warp PRIVATE -ffp-contract=off)
target_include_directories(gems_warp PRIVATE ${ITK_INCLUDE_DIRS})
target_link_libraries(gems_warp PRIVATE ${ITK_LIBRARIES})
set_target_properties(gems_warp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${FNIT_GEMS_OUTPUT_DIR}")
