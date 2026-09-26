#ifndef __PVM_MULTI_CUH__
#define __PVM_MULTI_CUH__

namespace Xfibres {

__device__ void cf_PVM_multi(
  // INPUT
  const float* params,
  const float* mdata,
  const float* bvecs,
  const float* bvals,
  const float  R,
  const float  invR,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  const int    Gamma_for_ball_only,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _a,
  float*       _b,
  float*       sumf,
  //OUTPUT
  double*      cfv);


__device__ void grad_PVM_multi(
  //INPUT
  const float* params,
  const float* mdata,
  const float* bvecs,
  const float* bvals,
  const float  R,
  const float  invR,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  const int    Gamma_for_ball_only,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _a,
  float*       _b,
  float*       sumf,
  //OUTPUT
  float*       grad);


__device__ void hess_PVM_multi(
  //INPUT
  const float* params,
  const float* bvecs,
  const float* bvals,
  const float  R,
  const float  invR,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  const int    Gamma_for_ball_only,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _a,
  float*       _b,
  float*       sumf,
  //OUTPUT
  float*       hess);


__global__ void fit_PVM_multi_kernel(
  //INPUT
  const float* data,
  const float* params_PVM_single_c,
  const float* bvecs,
  const float* bvals,
  const float  R,
  const float  invR,
  const int    nvox,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const int    Gamma_for_ball_only,
  const bool   m_include_f0,
  const bool   gradnonlin,
  //OUTPUT
  float*       params);


__global__ void get_residuals_PVM_multi_kernel(
  //INPUT
  const float* data,
  const float* params,
  const float* bvecs,
  const float* bvals,
  const float  R,
  const float  invR,
  const int    vox,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const int    Gamma_for_ball_only,
  const bool   m_include_f0,
  const bool   gradnonlin,
  const bool*  includes_f0,
  //OUTPUT
  float*       residuals);

#endif

} // namespace Xfibres
