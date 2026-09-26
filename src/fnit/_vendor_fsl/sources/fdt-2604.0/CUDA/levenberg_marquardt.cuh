#ifndef __LEVENBERG_MARQUARDT_CUH__
#define __LEVENBERG_MARQUARDT_CUH__


namespace Xfibres {


__device__ void levenberg_marquardt_PVM_single_gpu(
  //INPUT
  const float* mydata,
  const float* bvecs,
  const float* bvals,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  float*       step,
  float*       grad,
  float*       hess,
  float*       inverse,
  double*      pcf,
  double*      ncf,
  double*      lambda,
  double*      cftol,
  double*      ltol,
  double*      olambda,
  int*         success,
  int*         end,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _d,
  float*       sumf,
  float*       C,
  float*       el,
  int*         indx,
  //INPUT-OUTPUT
  float*       myparams
);

__device__ void levenberg_marquardt_PVM_single_c_gpu(
  //INPUT
  const float* mydata,
  const float* bvecs,
  const float* bvals,
  const int    ndirections,
  const int    nfib,
  const int    nparams,
  const bool   m_include_f0,
  const int    idSubVOX,
  float*       step,
  float*       grad,
  float*       hess,
  float*       inverse,
  double*      pcf,
  double*      ncf,
  double*      lambda,
  double*      cftol,
  double*      ltol,
  double*      olambda,
  int*         success,
  int*         end,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       f_deriv,
  float*       x,
  float*       _d,
  float*       sumf,
  float*       C,
  float*       el,
  int*         indx,
  //INPUT-OUTPUT
  float*       myparams
);



__device__ void levenberg_marquardt_PVM_multi_gpu(
  //INPUT
  const float* mydata,
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
  float*       step,
  float*       grad,
  float*       hess,
  float*       inverse,
  double*      pcf,
  double*      ncf,
  double*      lambda,
  double*      cftol,
  double*      ltol,
  double*      olambda,
  int*         success,
  int*         end,
  float*       J,
  float*       reduction,
  float*       fs,
  float*       x,
  float*       _a,
  float*       _b,
  float*       sumf,
  float*       C,
  float*       el,
  int*         indx,
  //INPUT-OUTPUT
  float*       myparams
);


} // namespace Xfibres

#endif
