/*  xfibres_gpu.cuh

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>
#include <string>

#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "newimage/newimageall.h"

#include "fibre_gpu.h"


//implemented and used in xfibres_gpu.cu
void fit(
        //INPUT
        const std::vector<NEWMAT::ColumnVector> datam_vec,
        const std::vector<NEWMAT::Matrix>       bvecs_vec,
        const std::vector<NEWMAT::Matrix>       bvals_vec,
        thrust::host_vector<float>              datam_host,
        thrust::host_vector<float>              bvecs_host,
        thrust::host_vector<float>              bvals_host,
        thrust::device_vector<float>            datam_gpu,
        thrust::device_vector<float>            bvecs_gpu,
        thrust::device_vector<float>            bvals_gpu,
        int                                     ndirections,
        std::string                             output_file,
        //OUTPUT
        thrust::device_vector<float>&           params_gpu,
        thrust::host_vector<int>&               vox_repeat,
        int&                                    nrepeat);

//implemented and used in xfibres_gpu.cu
void prepare_data_gpu_FIT(
                //INPUT
                const NEWMAT::Matrix               datam,
                const NEWMAT::Matrix               bvecs,
                const NEWMAT::Matrix               bvals,
                const NEWMAT::Matrix               gradm,
                //OUTPUT
                std::vector<NEWMAT::ColumnVector>& datam_vec,
                std::vector<NEWMAT::Matrix>&       bvecs_vec,
                std::vector<NEWMAT::Matrix>&       bvals_vec,
                thrust::host_vector<float>&        datam_host,
                thrust::host_vector<float>&        bvecs_host,
                thrust::host_vector<float>&        bvals_host,
                thrust::host_vector<double>&       alpha_host,
                thrust::host_vector<double>&       beta_host,
                thrust::host_vector<float>&        params_host,
                thrust::host_vector<float>&        tau_host);


//implemented and used in xfibres_gpu.cu
void prepare_data_gpu_FIT_repeat(
                    //INPUT
                    thrust::host_vector<float>         datam_host,
                    thrust::host_vector<float>         bvecs_host,
                    thrust::host_vector<float>         bvals_host,
                    thrust::host_vector<int>           vox_repeat,
                    int                                nrepeat,
                    int                                ndirections,
                    //OUTPUT
                    std::vector<NEWMAT::ColumnVector>& datam_repeat_vec,
                    std::vector<NEWMAT::Matrix>&       bvecs_repeat_vec,
                    std::vector<NEWMAT::Matrix>&       bvals_repeat_vec,
                    thrust::host_vector<float>&        datam_repeat_host,
                    thrust::host_vector<float>&        bvecs_repeat_host,
                    thrust::host_vector<float>&        bvals_repeat_host,
                    thrust::host_vector<float>&        params_repeat_host);

//implemented and used in xfibres_gpu.cu
void mix_params(
            //INPUT
            thrust::host_vector<float>    params_repeat_gpu,
            thrust::host_vector<int>      vox_repeat,
            int                           nrepeat,
            int                           nvox,
            //INPUT-OUTPUT
            thrust::device_vector<float>& params_gpu);


//implemented and used in xfibres_gpu.cu
void prepare_data_gpu_MCMC(
                //INPUT
                int                                 nvox,
                int                                 ndirections,
                int                                 nfib,
                //OUTPUT
                thrust::host_vector<double>&        signals_host,
                thrust::host_vector<double>&        isosignals_host,
                thrust::host_vector<FibreGPU>&      fibres_host,
                thrust::host_vector<MultifibreGPU>& multifibres_host);

//implemented and used in xfibres_gpu.cu
void prepare_data_gpu_MCMC_record(
                    //INPUT
                    int                           nvox,
                    //OUTPUT
                    thrust::device_vector<float>& rf0_gpu,
                    thrust::device_vector<float>& rtau_gpu,
                    thrust::device_vector<float>& rs0_gpu,
                    thrust::device_vector<float>& rd_gpu,
                    thrust::device_vector<float>& rdstd_gpu,
                    thrust::device_vector<float>& rR_gpu,
                    thrust::device_vector<float>& rth_gpu,
                    thrust::device_vector<float>& rph_gpu,
                    thrust::device_vector<float>& rf_gpu);

void resize_structures(
                //INPUT
                int                                 nVOX_multiple,
                int                                 ndirections,
                //OUTPUT
                thrust::device_vector<float>&       datam_gpu,
                thrust::device_vector<float>&       params_gpu,
                thrust::device_vector<float>&       tau_gpu,
                thrust::device_vector<float>&       bvals_gpu,
                thrust::device_vector<double>&      alpha_gpu,
                thrust::device_vector<double>&      beta_gpu,
                thrust::device_vector<curandState>& randStates_gpu);

//implemented and used in xfibres_gpu.cu
void record_finish_voxels(
                //INPUT
                thrust::device_vector<float>& rf0_gpu,
                thrust::device_vector<float>& rtau_gpu,
                thrust::device_vector<float>& rs0_gpu,
                thrust::device_vector<float>& rd_gpu,
                thrust::device_vector<float>& rdstd_gpu,
                thrust::device_vector<float>& rR_gpu,
                thrust::device_vector<float>& rth_gpu,
                thrust::device_vector<float>& rph_gpu,
                thrust::device_vector<float>& rf_gpu,
                int                           nvox,
                int                           nVOX_multiple,
                int                           idpart);
