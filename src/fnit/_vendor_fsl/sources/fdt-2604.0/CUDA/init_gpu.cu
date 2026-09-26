/*  init_gpu.cu

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include "init_gpu.cuh"
#include "sync_check.cuh"
#include <fstream>

namespace Xfibres {

void init_gpu(){

  int *q;
  cudaMalloc((void **)&q, sizeof(int));
  cudaFree(q);
  sync_check("init_gpu");

  int device;
  cudaGetDevice(&device);
  printf ("\n...................Allocated GPU %d...................\n", device);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  sync_check("init_gpu");
}

double timeval_diff(struct timeval *a, struct timeval *b){
  return (double)(a->tv_sec +(double)a->tv_usec/1000000) - (double)(b->tv_sec +(double)b->tv_usec/1000000);
}

} // namespace Xfibres
