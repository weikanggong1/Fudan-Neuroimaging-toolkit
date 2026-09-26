#ifndef __RUNMCMC_KERNELS_CUH__
#define __RUNMCMC_KERNELS_CUH__

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "fibre_gpu.cuh"


namespace Xfibres {


__global__ void setup_randoms_kernel(
  curandState* randstate,
  double       seed,
  int          nvox);

__global__ void init_Fibres_Multifibres_kernel(
   //INPUT
  const float*   datam,
  const float*   params,
  const float*   tau,
  const float*   bvals,
  const double*  alpha,
  const double*  beta,
  const float    R_priormean,
  const float    R_priorstd,
  const float    R_priorfudge,
  const int      ndirections,
  const int      nfib,
  const int      nparams_fit,
  const int      model,
  const float    fudgevalue,
  const bool     m_includef0,
  const bool     rician,
  const bool     m_ardf0,      // opts.ardf0.value()
  const bool     ard_value,    // opts.all_ard.value()
  const bool     no_ard_value, // opts.no_ard.value()
  const bool     gradnonlin,
  //TO USE
  double*        angtmp,
  //OUTPUT
  FibreGPU*      fibres,
  MultifibreGPU* multifibres,
  double*        signals,
  double*        isosignals);

__global__ void runmcmc_kernel(
  //INPUT
  const float*   datam,
  const float*   bvals,
  const double*  alpha,
  const double*  beta,
  curandState*   randstate,
  const float    R_priormean,
  const float    R_priorstd,
  const float    R_priorfudge,
  const int      ndirections,
  const int      nfib,
  const int      nparams,
  const int      model,
  const float    fudgevalue,
  const bool     m_include_f0,
  const bool     m_ardf0,
  const bool     can_use_ard,
  const bool     rician,
  const bool     gradnonlin,
  const int      updateproposalevery, //update every this number of iterations
  const int      iterations,
  const int      iters_burnin,        //iters in burin, we need it to continue the updates at the correct time.
  const int      record_every,        //record every this number
  const int      totalrecords,        //total number of records to do
  //TO USE
  double*        oldsignals,
  double*        oldisosignals,
  double*        angtmp,
  double*        oldangtmp,
  //INPUT-OUTPUT
  FibreGPU*      fibres,
  MultifibreGPU* multifibres,
  double*        signals,
  double*        isosignals,
  //OUTPUT
  float*         rf0,                 //record of parameters
  float*         rtau,
  float*         rs0,
  float*         rd,
  float*         rdstd,
  float*         rR,
  float*         rth,
  float*         rph,
  float*         rf);

} // namespace Xfibres


#endif
