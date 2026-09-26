#ifndef __PVM_SINGLE_C_CUH__
#define __PVM_SINGLE_C_CUH__


namespace Xfibres {

__device__ void cf_PVM_single_c(
  //INPUT
  const float* params,
  const float* mdata,
  const float* bvecs,
  const float* bvals,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _d,
  float*       sumf,
  //OUTPUT
  double*      cfv);


__device__ void grad_PVM_single_c(
  //INPUT
  const float* params,
  const float* mdata,
  const float* bvecs,
  const float* bvals,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       f_deriv,
  float*       x,
  float*       _d,
  float*       sumf,
  //OUTPUT
  float*       grad);


__device__ void hess_PVM_single_c(
  //INPUT
  const float* params,
  const float* bvecs,
  const float* bvals,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       f_deriv,
  float*       x,
  float*       _d,
  float*       sumf,
  //OUTPUT
  float*       hess);

__global__ void fit_PVM_single_c_kernel(
  //INPUT
  const float* data,
  const float* bvecs,
  const float* bvals,
  const int    nvox,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_eval_BIC,
  const bool   m_include_f0,
  const bool   m_return_fanning,
  const bool   gradnonlin,
  //INPUT - OUTPUT
  float*       params);

__global__ void get_residuals_PVM_single_c_kernel(
  //INPUT
  const float* data,
  const float* params,
  const float* bvecs,
  const float* bvals,
  const int    nvox,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const bool   gradnonlin,
  const bool*  includes_f0,
  //OUTPUT
  float*       residuals);


} // namespace Xfibres

#endif
