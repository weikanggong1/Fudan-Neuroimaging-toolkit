/*  diffmodels.cuh

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "armawrap/newmat.h"


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


__device__ void cf_PVM_single(
                    //INPUT
                    const float* params,
                    const float* data,
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

__device__ void grad_PVM_single(
                    //INPUT
                    const float* params,
                    const float* data,
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
                    float*       x,
                    float*       _d,
                    float*       sumf,
                    //OUTPUT
                    float*       grad);

__device__ void hess_PVM_single(
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
                    float*       x,
                    float*       _d,
                    float*       sumf,
                    //OUTPUT
                    float*       hess);

__device__ void cf_PVM_single_c(
                    //INPUT
                    const float* params,
                    const float* data,
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
                    const float* data,
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
                    float*       ess);

__device__ void cf_PVM_multi(
                    //INPUT
                    const float* params,
                    const float* data,
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
                    const float* data,
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
