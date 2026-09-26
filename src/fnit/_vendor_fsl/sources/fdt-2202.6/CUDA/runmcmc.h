/*  runmcmc.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

//////////////////////////////////////////////////////
//   MCMC ON GPU
//////////////////////////////////////////////////////

#include <curand_kernel.h>

void init_Fibres_Multifibres(
                //INPUT
                thrust::device_vector<float>&         datam_gpu,
                thrust::device_vector<float>&         params_gpu,
                thrust::device_vector<float>&         tau_gpu,
                thrust::device_vector<float>&         bvals_gpu,
                thrust::device_vector<double>&        alpha_gpu,
                thrust::device_vector<double>&        beta_gpu,
                const int                             ndirections,
                std::string                           output_file,
                double                                seed,
                //OUTPUT
                thrust::device_vector<FibreGPU>&      fibres_gpu,
                thrust::device_vector<MultifibreGPU>& multifibres_gpu,
                thrust::device_vector<double>&        signals_gpu,
                thrust::device_vector<double>&        isosignals_gpu,
                thrust::device_vector<curandState>&   randStates_gpu);

void runmcmc_burnin(
            //INPUT
            thrust::device_vector<float>&         datam_gpu,
            thrust::device_vector<float>&         bvals_gpu,
            thrust::device_vector<double>&        alpha_gpu,
            thrust::device_vector<double>&        beta_gpu,
            const int                             ndirections,
            std::string                           output_file,
            //INPUT-OUTPUT
            thrust::device_vector<FibreGPU>&      fibres_gpu,
            thrust::device_vector<MultifibreGPU>& multifibres_gpu,
            thrust::device_vector<double>&        signals_gpu,
            thrust::device_vector<double>&        isosignals_gpu,
            thrust::device_vector<curandState>&   randStates_gpu);


void runmcmc_record(
            //INPUT
            thrust::device_vector<float>&         datam_gpu,
            thrust::device_vector<float>&         bvals_gpu,
            thrust::device_vector<double>&        alpha_gpu,
            thrust::device_vector<double>&        beta_gpu,
            thrust::device_vector<FibreGPU>&      fibres_gpu,
            thrust::device_vector<MultifibreGPU>& multifibres_gpu,
            thrust::device_vector<double>&        signals_gpu,
            thrust::device_vector<double>&        isosignals_gpu,
            const int                             ndirections,
            thrust::device_vector<curandState>&   randStates_gpu,
            std::string                           output_file,
            //OUTPUT
            thrust::device_vector<float>&         rf0_gpu,
            thrust::device_vector<float>&         rtau_gpu,
            thrust::device_vector<float>&         rs0_gpu,
            thrust::device_vector<float>&         rd_gpu,
            thrust::device_vector<float>&         rdstd_gpu,
            thrust::device_vector<float>&         rR_gpu,
            thrust::device_vector<float>&         rth_gpu,
            thrust::device_vector<float>&         rph_gpu,
            thrust::device_vector<float>&         rf_gpu);
