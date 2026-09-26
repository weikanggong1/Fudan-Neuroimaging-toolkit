/*  diffmodels.cuh

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __DIFFMODELS_CUH__
#define __DIFFMODELS_CUH__

#include <string>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "armawrap/newmat.h"


namespace Xfibres {


void fit_PVM_single(
  //INPUT
  const std::vector<NEWMAT::ColumnVector> datam_vec,
  const std::vector<NEWMAT::Matrix>       bvecs_vec,
  const std::vector<NEWMAT::Matrix>       bvals_vec,
  thrust::device_vector<float>            datam_gpu,
  thrust::device_vector<float>            bvecs_gpu,
  thrust::device_vector<float>            bvals_gpu,
  int                                     ndirections,
  int                                     nfib,
  bool                                    m_include_f0,
  bool                                    gradnonlin,
  std::string                             output_file,
  //OUTPUT
  thrust::device_vector<float>&           params_gpu);

void fit_PVM_single_c(
  //INPUT
  const std::vector<NEWMAT::ColumnVector> datam_vec,
  const std::vector<NEWMAT::Matrix>       bvecs_vec,
  const std::vector<NEWMAT::Matrix>       bvals_vec,
  thrust::device_vector<float>            datam_gpu,
  thrust::device_vector<float>            bvecs_gpu,
  thrust::device_vector<float>            bvals_gpu,
  int                                     ndirections,
  int                                     nfib,
  bool                                    m_include_f0,
  bool                                    gradnonlin,
  std::string                             output_file,
  //OUTPUT
  thrust::device_vector<float>&           params_gpu);

void fit_PVM_multi(
  //INPUT
  thrust::device_vector<float>  datam_gpu,
  thrust::device_vector<float>  bvecs_gpu,
  thrust::device_vector<float>  bvals_gpu,
  int                           nvox,
  int                           ndirections,
  int                           nfib,
  bool                          m_include_f0,
  bool                          gradnonlin,
  float                         R_prior_mean,
  int                           Gamma_ball_only,
  std::string                   output_file,
  //OUTPUT
  thrust::device_vector<float>& params_gpu);

void calculate_tau(
  //INPUT
  thrust::device_vector<float> datam_gpu,
  thrust::device_vector<float> params_gpu,
  thrust::device_vector<float> bvecs_gpu,
  thrust::device_vector<float> bvals_gpu,
  thrust::host_vector<int>     vox_repeat,
  int                          nrepeat,
  int                          ndirections,
  int                          nfib,
  int                          model,
  bool                         m_include_f0,
  bool                         nonlin,
  bool                         gradnonlin,
  float                        R_prior_mean,
  int                          Gamma_ball_only,
  std::string                  output_file,
  //OUTPUT
  thrust::host_vector<float>&  tau);

} // namespace Xfibres

#endif
