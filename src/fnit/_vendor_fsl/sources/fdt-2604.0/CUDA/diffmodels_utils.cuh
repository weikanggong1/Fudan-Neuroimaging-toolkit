#ifndef __DIFFMODELS_UTILS_CUH__
#define __DIFFMODELS_UTILS_CUH__


namespace Xfibres {

extern __constant__ double two_pi_gpu;
extern __constant__ double FSMALL_gpu;

__device__ double beta2f_gpu(double beta);
__device__ double f2x_gpu(double x);
__device__ double x2f_gpu(double x);
__device__ int    sign_gpu(int x);
__device__ int    sign_gpu(float x);
__device__ int    sign_gpu(double x);

}  // namespace Xfibres


#endif
