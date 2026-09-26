#include "diffmodels_utils.cuh"


namespace Xfibres {

__constant__ double two_pi_gpu = 0.636619772;

__constant__ double FSMALL_gpu = 0.001;


__device__ double beta2f_gpu(double beta)
{
  double sinbeta= sin(beta);
  return sinbeta*sinbeta;
}

__device__ double f2x_gpu(double x)
{
  return tan(double((x)/two_pi_gpu));
}

__device__ double x2f_gpu(double x)
{
  return abs(two_pi_gpu*atan(x));
}

__device__ int sign_gpu(int x){
  if (x>0)      return 1;
  else if (x<0) return -1;
  else          return 0;
}

__device__ int sign_gpu(float x){
  if (x>0)      return 1;
  else if (x<0) return -1;
  else          return 0;
}

__device__ int sign_gpu(double x){
  if (x>0)      return 1;
  else if (x<0) return -1;
  else          return 0;
}

} // namespace Xfibres
