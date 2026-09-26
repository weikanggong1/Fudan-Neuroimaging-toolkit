#include <string>

#include <cuda.h>

#include "sync_check.cuh"

namespace Xfibres {

void sync_check(const std::string& msg) {

  cudaError_t err = cudaDeviceSynchronize();

  if (cudaSuccess != err) {
    printf("cuda error: %s\n", cudaGetErrorString(err));
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("ERROR: %s: %s\n", msg, cudaGetErrorString(error));
    exit(-1);
  }
}



} // namespace Xfibres;
