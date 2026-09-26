/*  runmcmc.cu

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */


#include <time.h>
#include <sys/time.h>
#include <string>
#include <fstream>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "xfibresoptions.h"
#include "runmcmc_kernels.cu"
#include "sync_check.h"

#include "init_gpu.h"

using namespace Xfibres;

//////////////////////////////////////////////////////
//   MCMC ON GPU
//////////////////////////////////////////////////////

void init_Fibres_Multifibres(	//INPUT
				thrust::device_vector<float>& 			datam_gpu,
				thrust::device_vector<float>& 			params_gpu,
				thrust::device_vector<float>& 			tau_gpu,
				thrust::device_vector<float>& 			bvals_gpu,
				thrust::device_vector<double>& 			alpha_gpu,
				thrust::device_vector<double>& 			beta_gpu,
				const int 					ndirections,
				std::string 						output_file,
				double 						seed,
				//OUTPUT
				thrust::device_vector<FibreGPU>& 		fibres_gpu,
				thrust::device_vector<MultifibreGPU>& 		multifibres_gpu,
				thrust::device_vector<double>&			signals_gpu,
				thrust::device_vector<double>&			isosignals_gpu,
				thrust::device_vector<curandState>&		randStates_gpu)
{
	std::ofstream myfile;
	myfile.open (output_file.data(), std::ios::out | std::ios::app );
   	myfile << "----- MCMC ALGORITHM PART INITIALITATION ON GPU ----- " << "\n";

   	struct timeval t1,t2;
   	double time;
   	gettimeofday(&t1,NULL);

	int nvox = multifibres_gpu.size();

	xfibresOptions& opts = xfibresOptions::getInstance();
	int nfib= opts.nfibres.value();
	int nparams_fit = 2+3*opts.nfibres.value();
	if(opts.modelnum.value()>=2) nparams_fit++;
	if(opts.f0.value()) nparams_fit++;

	thrust::device_vector<double> angtmp_gpu;
	angtmp_gpu.resize(nvox*ndirections*nfib);


	bool gradnonlin = opts.grad_file.set();

	int blocks = nvox/VOXELS_BLOCK_MCMC;
	if(nvox%VOXELS_BLOCK_MCMC) blocks++;
	int nthreads_block = THREADS_VOXEL_MCMC*VOXELS_BLOCK_MCMC;
  	dim3 Dim_Grid_MCMC(blocks, 1);
  	dim3 Dim_Block_MCMC(nthreads_block ,1);	///dimensions for MCMC

	float *datam_ptr = thrust::raw_pointer_cast(datam_gpu.data());
	float *params_ptr = thrust::raw_pointer_cast(params_gpu.data());
	float *tau_ptr = thrust::raw_pointer_cast(tau_gpu.data());
	float *bvals_ptr = thrust::raw_pointer_cast(bvals_gpu.data());
	double *alpha_ptr = thrust::raw_pointer_cast(alpha_gpu.data());
	double *beta_ptr = thrust::raw_pointer_cast(beta_gpu.data());
	FibreGPU *fibres_ptr =  thrust::raw_pointer_cast(fibres_gpu.data());
	MultifibreGPU *multifibres_ptr = thrust::raw_pointer_cast(multifibres_gpu.data());
	double *signals_ptr = thrust::raw_pointer_cast(signals_gpu.data());
	double *isosignals_ptr = thrust::raw_pointer_cast(isosignals_gpu.data());
	double *angtmp_ptr = thrust::raw_pointer_cast(angtmp_gpu.data());
	curandState *randStates_ptr = thrust::raw_pointer_cast(randStates_gpu.data());

	int amount_shared = VOXELS_BLOCK_MCMC*((THREADS_VOXEL_MCMC)*sizeof(double) + (3*nfib + 9)*sizeof(float) + sizeof(int));

	myfile << "Shared Memory Used in init_Fibres_Multifibres: " << amount_shared << "\n";

	init_Fibres_Multifibres_kernel<<< Dim_Grid_MCMC, Dim_Block_MCMC, amount_shared>>>(datam_ptr, params_ptr, tau_ptr, bvals_ptr, alpha_ptr, beta_ptr, opts.R_prior_mean.value(), opts.R_prior_std.value(),opts.R_prior_fudge.value(), ndirections, nfib, nparams_fit, opts.modelnum.value(), opts.fudge.value(), opts.f0.value(), opts.rician.value(), opts.ardf0.value(), opts.all_ard.value(), opts.no_ard.value(), gradnonlin, angtmp_ptr, fibres_ptr, multifibres_ptr, signals_ptr, isosignals_ptr);
	sync_check("init_Fibres_Multifibres_kernel");

	// Initialise Randoms
	int total_threads= nvox;
	int blocks_Rand = total_threads/THREADS_BLOCK_RAND;
	if(total_threads%THREADS_BLOCK_RAND) blocks_Rand++;
	dim3 Dim_Grid_Rand(blocks_Rand,1);
	dim3 Dim_Block_Rand(THREADS_BLOCK_RAND,1);
	setup_randoms_kernel <<<Dim_Grid_Rand,Dim_Block_Rand>>>(randStates_ptr,seed,nvox);
	sync_check("Setup_Randoms_kernel");

	gettimeofday(&t2,NULL);
    	time=timeval_diff(&t2,&t1);
   	myfile << "TIME: " << time << " seconds\n";
	myfile << "-----------------------------------------------------" << "\n\n" ;
	myfile.close();
}

void runmcmc_burnin(	//INPUT
			thrust::device_vector<float>& 			datam_gpu,
			thrust::device_vector<float>& 			bvals_gpu,
			thrust::device_vector<double>& 			alpha_gpu,
			thrust::device_vector<double>& 			beta_gpu,
			const int 					ndirections,
			std::string 						output_file,
			//INPUT-OUTPUT
			thrust::device_vector<FibreGPU>& 		fibres_gpu,
			thrust::device_vector<MultifibreGPU>& 		multifibres_gpu,
			thrust::device_vector<double>&			signals_gpu,
			thrust::device_vector<double>&			isosignals_gpu,
			thrust::device_vector<curandState>&		randStates_gpu)
{
	xfibresOptions& opts = xfibresOptions::getInstance();

	std::ofstream myfile;
	myfile.open (output_file.data(), std::ios::out | std::ios::app );
   	myfile << "--------- MCMC ALGORITHM PART BURNIN ON GPU --------- " << "\n";

   	struct timeval t_tot1,t_tot2;
   	double time;
   	time=0;

   	gettimeofday(&t_tot1,NULL);

	int nvox = multifibres_gpu.size();
   	int nfib= opts.nfibres.value();
	int nparams;

	bool gradnonlin=opts.grad_file.set();

	if(opts.f0.value()) nparams=3+nfib*3;
	else nparams=2+nfib*3;
	if(opts.modelnum.value()>=2) nparams++;
	if(opts.modelnum.value()==3) nparams++;
	if(opts.rician.value()) nparams++;

	thrust::device_vector<float> recors_null_gpu;
	recors_null_gpu.resize(1);

	thrust::device_vector<double> angtmp_gpu;
	thrust::device_vector<double> oldangtmp_gpu;
	thrust::device_vector<double> oldsignals_gpu;
	thrust::device_vector<double> oldisosignals_gpu;

	angtmp_gpu.resize(nvox*ndirections*nfib);
	oldangtmp_gpu.resize(nvox*ndirections);
	oldsignals_gpu.resize(nvox*ndirections*nfib);
	oldisosignals_gpu.resize(nvox*ndirections);

	myfile << "Processing " << nvox << " voxels \n";

  	int blocks = nvox/VOXELS_BLOCK_MCMC;
	if(nvox%VOXELS_BLOCK_MCMC) blocks++;
	int nthreads_block = THREADS_VOXEL_MCMC*VOXELS_BLOCK_MCMC;
  	dim3 Dim_Grid(blocks, 1);
  	dim3 Dim_Block(nthreads_block,1);	//dimensions for MCMC

   	myfile << "NUM BLOCKS: " << blocks << "\n";
   	myfile << "THREADS PER BLOCK : " << nthreads_block << "\n";


	//get pointers
	float *datam_ptr = thrust::raw_pointer_cast(datam_gpu.data());
	float *bvals_ptr = thrust::raw_pointer_cast(bvals_gpu.data());
	double *alpha_ptr = thrust::raw_pointer_cast(alpha_gpu.data());
	double *beta_ptr = thrust::raw_pointer_cast(beta_gpu.data());
	FibreGPU *fibres_ptr =  thrust::raw_pointer_cast(fibres_gpu.data());
	MultifibreGPU *multifibres_ptr = thrust::raw_pointer_cast(multifibres_gpu.data());
	double *signals_ptr = thrust::raw_pointer_cast(signals_gpu.data());
	double *isosignals_ptr = thrust::raw_pointer_cast(isosignals_gpu.data());
	curandState *randStates_ptr = thrust::raw_pointer_cast(randStates_gpu.data());

	double *angtmp_ptr = thrust::raw_pointer_cast(angtmp_gpu.data());
	double *oldangtmp_ptr = thrust::raw_pointer_cast(oldangtmp_gpu.data());
	double *oldsignals_ptr = thrust::raw_pointer_cast(oldsignals_gpu.data());
	double *oldisosignals_ptr = thrust::raw_pointer_cast(oldisosignals_gpu.data());

	float *records_null = thrust::raw_pointer_cast(recors_null_gpu.data());

	int amount_shared = VOXELS_BLOCK_MCMC*((THREADS_VOXEL_MCMC)*sizeof(double) + (10*nfib + 27)*sizeof(float) + (7*nfib + 20)*sizeof(int)+ sizeof(curandState));

	myfile << "Shared Memory Used in runmcmc_burnin: " << amount_shared << "\n";

   	if(nvox!=0){
		runmcmc_kernel<<< Dim_Grid, Dim_Block, amount_shared >>>(datam_ptr, bvals_ptr, alpha_ptr, beta_ptr, randStates_ptr, opts.R_prior_mean.value(), opts.R_prior_std.value(),opts.R_prior_fudge.value(), ndirections, nfib, nparams, opts.modelnum.value(), opts.fudge.value(), opts.f0.value(), opts.ardf0.value(), !opts.no_ard.value(), opts.rician.value(), gradnonlin, opts.updateproposalevery.value(), opts.nburn.value(), 0, 0, 0, oldsignals_ptr, oldisosignals_ptr, angtmp_ptr, oldangtmp_ptr, fibres_ptr, multifibres_ptr, signals_ptr, isosignals_ptr,records_null,records_null,records_null,records_null,records_null,records_null,records_null, records_null,records_null);
   		sync_check("runmcmc_burnin_kernel");
   	}

	gettimeofday(&t_tot2,NULL);
    	time=timeval_diff(&t_tot2,&t_tot1);
   	myfile << "TIME: " << time << " seconds\n";
	myfile << "-----------------------------------------------------" << "\n\n" ;
	myfile.close();
}


void runmcmc_record(	//INPUT
			thrust::device_vector<float>& 			datam_gpu,
			thrust::device_vector<float>& 			bvals_gpu,
			thrust::device_vector<double>& 			alpha_gpu,
			thrust::device_vector<double>& 			beta_gpu,
			thrust::device_vector<FibreGPU>& 		fibres_gpu,
			thrust::device_vector<MultifibreGPU>& 		multifibres_gpu,
			thrust::device_vector<double>&			signals_gpu,
			thrust::device_vector<double>&			isosignals_gpu,
			const int 					ndirections,
			thrust::device_vector<curandState>&		randStates_gpu,
			std::string 						output_file,
			//OUTPUT
			thrust::device_vector<float>&			rf0_gpu,
			thrust::device_vector<float>&			rtau_gpu,
			thrust::device_vector<float>&			rs0_gpu,
			thrust::device_vector<float>&			rd_gpu,
			thrust::device_vector<float>&			rdstd_gpu,
			thrust::device_vector<float>&			rR_gpu,
			thrust::device_vector<float>&			rth_gpu,
			thrust::device_vector<float>&			rph_gpu,
			thrust::device_vector<float>&			rf_gpu)
{
	xfibresOptions& opts = xfibresOptions::getInstance();

	std::ofstream myfile;
	myfile.open (output_file.data(), std::ios::out | std::ios::app );
   	myfile << "--------- MCMC ALGORITHM PART RECORD ON GPU --------- " << "\n";

   	struct timeval t_tot1,t_tot2;
   	double time;
   	time=0;

   	gettimeofday(&t_tot1,NULL);

	int totalrecords = (opts.njumps.value()/opts.sampleevery.value());

	int nvox = multifibres_gpu.size();
   	int nfib= opts.nfibres.value();
	int nparams;

	bool gradnonlin=opts.grad_file.set();

	if(opts.f0.value()) nparams=3+nfib*3;
	else nparams=2+nfib*3;
	if(opts.modelnum.value()>=2) nparams++;
	if(opts.modelnum.value()==3) nparams++;
	if(opts.rician.value()) nparams++;

	thrust::device_vector<double> angtmp_gpu;
	thrust::device_vector<double> oldangtmp_gpu;
	thrust::device_vector<double> oldsignals_gpu;
	thrust::device_vector<double> oldisosignals_gpu;

	angtmp_gpu.resize(nvox*ndirections*nfib);
	oldangtmp_gpu.resize(nvox*ndirections);
	oldsignals_gpu.resize(nvox*ndirections*nfib);
	oldisosignals_gpu.resize(nvox*ndirections);

	myfile << "Processing " << nvox << " voxels \n";

  	int blocks = nvox/VOXELS_BLOCK_MCMC;
	int nthreads_block = THREADS_VOXEL_MCMC*VOXELS_BLOCK_MCMC;
	if(nvox%VOXELS_BLOCK_MCMC) blocks++;
  	dim3 Dim_Grid(blocks, 1);
  	dim3 Dim_Block(nthreads_block,1);	//dimensions for MCMC

   	myfile << "NUM BLOCKS: " << blocks << "\n";
   	myfile << "THREADS PER BLOCK : " << nthreads_block << "\n";

	//get pointers
	float *datam_ptr = thrust::raw_pointer_cast(datam_gpu.data());
	float *bvals_ptr = thrust::raw_pointer_cast(bvals_gpu.data());
	double *alpha_ptr = thrust::raw_pointer_cast(alpha_gpu.data());
	double *beta_ptr = thrust::raw_pointer_cast(beta_gpu.data());
	FibreGPU *fibres_ptr =  thrust::raw_pointer_cast(fibres_gpu.data());
	MultifibreGPU *multifibres_ptr = thrust::raw_pointer_cast(multifibres_gpu.data());
	double *signals_ptr = thrust::raw_pointer_cast(signals_gpu.data());
	double *isosignals_ptr = thrust::raw_pointer_cast(isosignals_gpu.data());
	curandState *randStates_ptr = thrust::raw_pointer_cast(randStates_gpu.data());

	double *angtmp_ptr = thrust::raw_pointer_cast(angtmp_gpu.data());
	double *oldangtmp_ptr = thrust::raw_pointer_cast(oldangtmp_gpu.data());
	double *oldsignals_ptr = thrust::raw_pointer_cast(oldsignals_gpu.data());
	double *oldisosignals_ptr = thrust::raw_pointer_cast(oldisosignals_gpu.data());

	float *rf0_ptr = thrust::raw_pointer_cast(rf0_gpu.data());
	float *rtau_ptr = thrust::raw_pointer_cast(rtau_gpu.data());
	float *rs0_ptr = thrust::raw_pointer_cast(rs0_gpu.data());
	float *rd_ptr = thrust::raw_pointer_cast(rd_gpu.data());
	float *rdstd_ptr = thrust::raw_pointer_cast(rdstd_gpu.data());
	float *rR_ptr = thrust::raw_pointer_cast(rR_gpu.data());
	float *rth_ptr = thrust::raw_pointer_cast(rth_gpu.data());
	float *rph_ptr = thrust::raw_pointer_cast(rph_gpu.data());
	float *rf_ptr = thrust::raw_pointer_cast(rf_gpu.data());

	int amount_shared = VOXELS_BLOCK_MCMC*((THREADS_VOXEL_MCMC)*sizeof(double) + (10*nfib + 27)*sizeof(float) + (7*nfib + 20)*sizeof(int)+ sizeof(curandState));

	myfile << "Shared Memory Used in runmcmc_record: " << amount_shared << "\n";

   	if(nvox!=0){
		runmcmc_kernel<<< Dim_Grid, Dim_Block, amount_shared >>>(datam_ptr, bvals_ptr, alpha_ptr, beta_ptr, randStates_ptr, opts.R_prior_mean.value(), opts.R_prior_std.value(),opts.R_prior_fudge.value(), ndirections, nfib, nparams, opts.modelnum.value(), opts.fudge.value(), opts.f0.value(), opts.ardf0.value(), !opts.no_ard.value(), opts.rician.value(), gradnonlin, opts.updateproposalevery.value(), opts.njumps.value(), opts.nburn.value(), opts.sampleevery.value(), totalrecords, oldsignals_ptr, oldisosignals_ptr, angtmp_ptr, oldangtmp_ptr, fibres_ptr, multifibres_ptr, signals_ptr, isosignals_ptr, rf0_ptr, rtau_ptr, rs0_ptr, rd_ptr, rdstd_ptr, rR_ptr, rth_ptr, rph_ptr, rf_ptr);
   		sync_check("runmcmc_record_kernel");
   	}

   	gettimeofday(&t_tot2,NULL);
    	time=timeval_diff(&t_tot2,&t_tot1);
   	myfile << "TIME: " << time << " seconds\n";
	myfile << "-----------------------------------------------------" << "\n" ;
	myfile.close();
}
