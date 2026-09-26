/*  tractography_gpu.cu

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>
#include <string>
#include <iostream>

#include <CUDA/tractography_gpu.cuh>
#include <CUDA/tractographyKernels.cu>
#include <CUDA/memManager_gpu.cu>
#include <CUDA/tractography_CallKernels.cu>
#include <sys/time.h>

#include "newimage/newimage.h"

using namespace std;
using namespace NEWIMAGE;

void tractography_gpu(
  tractographyData&	data_host,
	volume<float>*&		mprob,
	volume<float>*&		mprob2,		// omeanpathlength
	int*&			keeptotal,
	float**			ConNet,
	float**			ConNetb,	// omeanpathlength
	float**			ConMat1,
	float**			ConMat1b,	// omeanpathlength
	float**			ConMat3,
	float**			ConMat3b,	// omeanpathlength
	float*&			m_s2targets,	// seed 2 targets
	float*&			m_s2targetsb,
	vector< vector<float> >& m_save_paths,
	volume4D<float>*&	mlocaldir)
{
  init_gpu();
  size_t free,total;
  cudaMemGetInfo(&free,&total);
  cout << "Device memory available (MB): "<< free/1048576 <<  " ---- Total device memory(MB): " << total/1048576 << "\n";

  probtrackxOptions& opts=probtrackxOptions::getInstance();

  tractographyData *data_gpu;
  copy_to_gpu(data_host,data_gpu);	// Copy all the masks, seeds and other info to the GPU

  copy_ToConstantMemory(data_host);	// Set Constant memory
  copy_ToTextureMemory(data_host);	// Set Texture memory

  cuMemGetInfo(&free,&total);
  cout << "Device memory available after copying data (MB): "<< free/1048576 << "\n";

  long MAX_SLs;
  long THREADS_STREAM; // MAX_Streamlines and NSTREAMS must be multiples

  ///// DATA in HOST ////
  int** lengths_host=new int*;			// Pinned Memory
  float** paths_host=new float*;		// Pinned Memory, only used if save_paths
  float** mprob_host=new float*;		// Pinned Memory
  float** mprob2_host=new float*;	    	// Pinned Memory
  float** mlocaldir_host=new float*;		// Pinned Memory

  float3** mat_crossed_host=new float3*; 	// .x id, .y triangle, .z value   Pinned Memory
  int** mat_numcrossed_host=new int*; 		// Pinned Memory
  long long size_mat_cross;
  int max_per_jump_mat;

  float3** lrmat_crossed_host=new float3*;	// Pinned Memory
  int** lrmat_numcrossed_host=new int*;		// Pinned Memory
  long long size_lrmat_cross;
  int max_per_jump_lrmat;

  //float** targVOLvalues_host=new float*;	// Pinned Memory
  //float** targvaluesb_host=new float*;        // Pinned Memory

  allocate_host_mem(data_host,MAX_SLs,THREADS_STREAM,
		    lengths_host,paths_host,mprob_host,mprob2_host,mlocaldir_host,
		    //targvalues_host,targvaluesb_host,
		    mat_crossed_host,mat_numcrossed_host,size_mat_cross,max_per_jump_mat,
		    lrmat_crossed_host,lrmat_numcrossed_host,size_lrmat_cross,max_per_jump_lrmat);
  ///////////////////////

  // Calculate number of Iterations
  int niters=0;
  unsigned long long totalSLs = (unsigned long long)data_host.nseeds*data_host.nparticles;
  printf("Total number of streamlines: %llu\n" ,totalSLs);
  niters=totalSLs/THREADS_STREAM;
  if(totalSLs%THREADS_STREAM) niters++;
  int last_iter = totalSLs-((niters-1)*THREADS_STREAM); // last iteration

  ///// DATA in GPU /////
  float** mprob_gpu=new float*;
  float** mprob2_gpu=new float*;
  int** beenhere_gpu=new int*;
  float** ConNet_gpu=new float*;
  float** ConNetb_gpu=new float*;
  bool net_flags_in_shared;
  float** net_flags_gpu=new float*;
  float** net_values_gpu=new float*;
  float** s2targets_gpu=new float*;
  float** s2targetsb_gpu=new float*;
  bool targ_flags_in_shared;
  float** targ_flags_gpu=new float*;
  float** mlocaldir_gpu=new float*;

  float** paths_gpu=new float*;
  int** lengths_gpu=new int*;
  int** loopcheckkeys_gpu=new int*;
  float3** loopcheckdirs_gpu=new float3*;

  float3** mat_crossed_gpu=new float3*;
  int** mat_numcrossed_gpu=new int*;
  float3** lrmat_crossed_gpu=new float3*;
  int** lrmat_numcrossed_gpu=new int*;

  allocate_gpu_mem(data_host,MAX_SLs,THREADS_STREAM,
		   mprob_gpu,mprob2_gpu,mlocaldir_gpu,beenhere_gpu,
		   ConNet_gpu,ConNetb_gpu,net_flags_in_shared,net_flags_gpu,net_values_gpu,
		   s2targets_gpu,s2targetsb_gpu,targ_flags_in_shared,targ_flags_gpu,
		   paths_gpu,lengths_gpu,loopcheckkeys_gpu,loopcheckdirs_gpu,
		   mat_crossed_gpu,mat_numcrossed_gpu,size_mat_cross,
		   lrmat_crossed_gpu,lrmat_numcrossed_gpu,size_lrmat_cross);
  ///////////////////////

  curandState* devStates;	// Random Seeds for GPU
  initialise_SeedsGPU(devStates,THREADS_STREAM);

  // create cuda streams:
  // 1 stream for processing on the GPU, 1 stream for copying GPU->CPU
  // This design is optimised for computing Matrix 1 and Matrix 3
  cudaStream_t streams[NSTREAMS];
  for(int i=0;i<NSTREAMS;i++){
    checkCuda(cudaStreamCreate(&streams[i]));
  }

  //The host memory involved in the data transfer must be pinned memory.
  //The default stream is different from other streams because it is a synchronizing stream with respect
  //to operations on the device: no operation in the default stream will begin until all previously issued
  //operations in any stream on the device have completed, and an operation in the default stream must complete
  //before any other operation (in any stream on the device) will begin.

  long long offset_SLs=0;	//offset Stream Lines

  int* PTR_lengths_gpuA=lengths_gpu[0];
  int* PTR_lengths_gpuB=lengths_gpu[0];
  float* PTR_paths_gpuA=paths_gpu[0];		// only change if save_paths
  float* PTR_paths_gpuB=paths_gpu[0];		// only used if save_paths
  float3* PTR_mat_crossed_gpuA=mat_crossed_gpu[0];
  int* PTR_mat_numcrossed_gpuA=mat_numcrossed_gpu[0];
  float3* PTR_mat_crossed_gpuB=mat_crossed_gpu[0];
  int* PTR_mat_numcrossed_gpuB=mat_numcrossed_gpu[0];
  float3* PTR_lrmat_crossed_gpuA=lrmat_crossed_gpu[0];
  int* PTR_lrmat_numcrossed_gpuA=lrmat_numcrossed_gpu[0];
  float3* PTR_lrmat_crossed_gpuB=lrmat_crossed_gpu[0];
  int* PTR_lrmat_numcrossed_gpuB=lrmat_numcrossed_gpu[0];

  //set num blocks in general (update kernel uses a different block size)
  int num_threads=THREADS_STREAM;

  int update_upper_limit = data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2];

  checkCuda(cudaDeviceSynchronize());
  cuMemGetInfo(&free,&total);

  // run iterations
  for(int iter=0;iter<niters;iter++){

    printf("Iteration %i out of %i\n",iter+1,niters);

    if(iter%2){
      PTR_lengths_gpuA=&lengths_gpu[0][THREADS_STREAM*2];  // here processing
      PTR_mat_crossed_gpuA=&mat_crossed_gpu[0][size_mat_cross];
      PTR_mat_numcrossed_gpuA=&mat_numcrossed_gpu[0][THREADS_STREAM];
      PTR_lrmat_crossed_gpuA=&lrmat_crossed_gpu[0][size_lrmat_cross];
      PTR_lrmat_numcrossed_gpuA=&lrmat_numcrossed_gpu[0][THREADS_STREAM];
      if(opts.save_paths.value()){
	      PTR_paths_gpuA=&paths_gpu[0][THREADS_STREAM*data_host.nsteps*3];
      }

      PTR_lengths_gpuB=lengths_gpu[0];			// here tranferring
      PTR_paths_gpuB=paths_gpu[0];
      PTR_mat_crossed_gpuB=mat_crossed_gpu[0];
      PTR_mat_numcrossed_gpuB=mat_numcrossed_gpu[0];
      PTR_lrmat_crossed_gpuB=lrmat_crossed_gpu[0];
      PTR_lrmat_numcrossed_gpuB=lrmat_numcrossed_gpu[0];
    }else{
      PTR_lengths_gpuA=lengths_gpu[0];			// here processing
      PTR_paths_gpuA=paths_gpu[0];
      PTR_mat_crossed_gpuA=mat_crossed_gpu[0];
      PTR_mat_numcrossed_gpuA=mat_numcrossed_gpu[0];
      PTR_lrmat_crossed_gpuA=lrmat_crossed_gpu[0];
      PTR_lrmat_numcrossed_gpuA=lrmat_numcrossed_gpu[0];

      PTR_lengths_gpuB=&lengths_gpu[0][THREADS_STREAM*2];	// here tranferring
      PTR_paths_gpuB=&paths_gpu[0][THREADS_STREAM*data_host.nsteps*3];
      PTR_mat_crossed_gpuB=&mat_crossed_gpu[0][size_mat_cross];
      PTR_mat_numcrossed_gpuB=&mat_numcrossed_gpu[0][THREADS_STREAM];
      PTR_lrmat_crossed_gpuB=&lrmat_crossed_gpu[0][size_lrmat_cross];
      PTR_lrmat_numcrossed_gpuB=&lrmat_numcrossed_gpu[0][THREADS_STREAM];
    }

    if(iter==(niters-1)){
      //nseeds_iter=last_iter;
      //set num blocks for the last iteration
      num_threads=last_iter;
    }
    checkCuda(cudaStreamSynchronize(streams[0])); // WAIT HERE FOR THE GPU-PROCESSING STREAM

    // CALCULATE PATH
    calculate_path(streams[0],data_gpu,num_threads,devStates,
		offset_SLs,loopcheckkeys_gpu,loopcheckdirs_gpu,PTR_paths_gpuA,PTR_lengths_gpuA);

    // STOP MASK
    stop_mask(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA);

    // WTSTOP MASKS
    wtstop_masks(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA);

    // AVOID MASK
    avoid_mask(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA);

    // WAYPOINTS MASK
    way_masks(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA);

    // NETWORK MASK
    net_masks(streams[0],data_host,data_gpu,num_threads,offset_SLs,PTR_paths_gpuA,PTR_lengths_gpuA,ConNet_gpu,ConNetb_gpu,net_flags_in_shared,net_flags_gpu,net_values_gpu);

    // TARGETS MASK
    if(opts.s2tout.value())
      targets_masks(streams[0],data_host,data_gpu,num_threads,offset_SLs,PTR_paths_gpuA,PTR_lengths_gpuA,s2targets_gpu,s2targetsb_gpu,targ_flags_in_shared,targ_flags_gpu);

    // UPDATE OUTPUT
    if(opts.simpleout.value()){
      update_path(streams[0],data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA,
		  beenhere_gpu,update_upper_limit,mprob_gpu,mprob2_gpu,mlocaldir_gpu);
    }

    // MATRIX 1
    if(opts.matrix1out.value()||opts.matrix2out.value()){
      matrix1(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA,
	    PTR_mat_crossed_gpuA,PTR_mat_numcrossed_gpuA,
	    PTR_lrmat_crossed_gpuA,PTR_lrmat_numcrossed_gpuA);
    }

    // MATRIX 3
    if(opts.matrix3out.value()){
      matrix3(streams[0],data_host,data_gpu,num_threads,PTR_paths_gpuA,PTR_lengths_gpuA,
	    PTR_mat_crossed_gpuA,PTR_mat_numcrossed_gpuA,
	    PTR_lrmat_crossed_gpuA,PTR_lrmat_numcrossed_gpuA);
    }

    if(iter>0){
      // COPY GPU -> HOST
      if(opts.matrix3out.value()){
        checkCuda(cudaMemcpyAsync(*mat_crossed_host,PTR_mat_crossed_gpuB,size_mat_cross*sizeof(float3),cudaMemcpyDeviceToHost,streams[1]));
        checkCuda(cudaMemcpyAsync(*mat_numcrossed_host,PTR_mat_numcrossed_gpuB,THREADS_STREAM*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
      }
      if(opts.matrix1out.value()||opts.matrix2out.value()||opts.lrmask3.value()!=""){
        checkCuda(cudaMemcpyAsync(*lrmat_crossed_host,PTR_lrmat_crossed_gpuB,size_lrmat_cross*sizeof(float3),cudaMemcpyDeviceToHost,streams[1]));
        checkCuda(cudaMemcpyAsync(*lrmat_numcrossed_host,PTR_lrmat_numcrossed_gpuB,THREADS_STREAM*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
      }

      checkCuda(cudaMemcpyAsync(*lengths_host,PTR_lengths_gpuB,THREADS_STREAM*2*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
      if(opts.save_paths.value()){
	      checkCuda(cudaMemcpyAsync(*paths_host,PTR_paths_gpuB,THREADS_STREAM*data_host.nsteps*3*sizeof(float),cudaMemcpyDeviceToHost,streams[1]));
      }

      checkCuda(cudaStreamSynchronize(streams[1])); // WAIT HERE UNTIL ALL COPIES HAVE FINISHED

      // HOST work
      // Update keeptotal
      size_t pos=0;
      if(!opts.network.value()){
	      for(int i=0;i<THREADS_STREAM;i++){
	        if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
	          keeptotal[0]++;	// Reduction ...maybe better in GPU and avoid memcpy
	        }
	        pos=pos+2;
	      }
      }else{
	      // Network mode
        size_t aux=0;
        for(int i=0;i<THREADS_STREAM;i++){
          size_t numseed=(offset_SLs-THREADS_STREAM+aux)/data_host.nparticles;
          if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
            keeptotal[data_host.seeds_ROI[numseed]]++; // maybe better in GPU
            if(opts.save_paths.value()){
	          }
	        }
          aux++;
          pos=pos+2;
	      }
      }
      if(opts.save_paths.value()){
        // save coordinates
	pos=0;
	for(size_t i=0;i<THREADS_STREAM;i++){
	  if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
	    vector<float> tmp(3*(lengths_host[0][pos] + lengths_host[0][pos+1]));
            if(lengths_host[0][pos]>0){
              size_t posSEED=i*data_host.nsteps*3;
              size_t posCURRENT=0;
              for(;posCURRENT<lengths_host[0][pos];posCURRENT++){
                tmp.push_back(paths_host[0][posSEED+posCURRENT*3]);
                tmp.push_back(paths_host[0][posSEED+posCURRENT*3+1]);
                tmp.push_back(paths_host[0][posSEED+posCURRENT*3+2]);
              }
            }
            if(lengths_host[0][pos+1]>0){
              size_t pos2=i*data_host.nsteps*3+((data_host.nsteps/2)*3);
              size_t co=0;
              for(;co<lengths_host[0][pos+1];co++){
                tmp.push_back(paths_host[0][pos2+co*3]);
                tmp.push_back(paths_host[0][pos2+co*3+1]);
                tmp.push_back(paths_host[0][pos2+co*3+2]);
              }
            }
            m_save_paths.push_back(tmp);
          }
          pos=pos+2;
        }
      }
      if(opts.matrix3out.value()){
	      write_mask3(THREADS_STREAM,mat_crossed_host[0],mat_numcrossed_host[0],max_per_jump_mat,
		    lrmat_crossed_host[0],lrmat_numcrossed_host[0],max_per_jump_lrmat,ConMat3,ConMat3b);
      }
      if(opts.matrix1out.value()||opts.matrix2out.value()){
	      write_mask1(data_host,(offset_SLs-THREADS_STREAM),THREADS_STREAM,
		    lrmat_crossed_host[0],lrmat_numcrossed_host[0],max_per_jump_lrmat,ConMat1,ConMat1b);
      }
    }

    offset_SLs+=THREADS_STREAM;
  }
  // end iterations

  if((niters)%2){
    PTR_lengths_gpuB=lengths_gpu[0];					//here for copying
    PTR_paths_gpuB=paths_gpu[0];
    PTR_mat_crossed_gpuB=mat_crossed_gpu[0];
    PTR_mat_numcrossed_gpuB=mat_numcrossed_gpu[0];
    PTR_lrmat_crossed_gpuB=lrmat_crossed_gpu[0];
    PTR_lrmat_numcrossed_gpuB=lrmat_numcrossed_gpu[0];
  }else{
    PTR_lengths_gpuB=&lengths_gpu[0][THREADS_STREAM*2];
    PTR_paths_gpuB=&paths_gpu[0][THREADS_STREAM*data_host.nsteps*3];
    PTR_mat_crossed_gpuB=&mat_crossed_gpu[0][size_mat_cross];
    PTR_mat_numcrossed_gpuB=&mat_numcrossed_gpu[0][THREADS_STREAM];
    PTR_lrmat_crossed_gpuB=&lrmat_crossed_gpu[0][size_lrmat_cross];
    PTR_lrmat_numcrossed_gpuB=&lrmat_numcrossed_gpu[0][THREADS_STREAM];
  }

  checkCuda(cudaStreamSynchronize(streams[0]));

  if(opts.matrix3out.value()){
    checkCuda(cudaMemcpyAsync(*mat_crossed_host,PTR_mat_crossed_gpuB,size_mat_cross*sizeof(float3),cudaMemcpyDeviceToHost,streams[1]));
    checkCuda(cudaMemcpyAsync(*mat_numcrossed_host,PTR_mat_numcrossed_gpuB,num_threads*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
  }
  if(opts.matrix1out.value()||opts.matrix2out.value()||opts.lrmask3.value()!=""){
    checkCuda(cudaMemcpyAsync(*lrmat_crossed_host,PTR_lrmat_crossed_gpuB,size_lrmat_cross*sizeof(float3),cudaMemcpyDeviceToHost,streams[1]));
    checkCuda(cudaMemcpyAsync(*lrmat_numcrossed_host,PTR_lrmat_numcrossed_gpuB,num_threads*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
  }
  checkCuda(cudaMemcpyAsync(*lengths_host,PTR_lengths_gpuB,num_threads*2*sizeof(int),cudaMemcpyDeviceToHost,streams[1]));
  if(opts.save_paths.value()){
    checkCuda(cudaMemcpyAsync(*paths_host,PTR_paths_gpuB,THREADS_STREAM*data_host.nsteps*3*sizeof(float),cudaMemcpyDeviceToHost,streams[1]));
  }

  checkCuda(cudaStreamSynchronize(streams[1]));

  // HOST work
  // Update keeptotal
  size_t pos=0;
  if(!opts.network.value()){
    for(int i=0;i<last_iter;i++){
      if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
	      keeptotal[0]++;	// Reduction ...maybe better in GPU and avoid memcpy
      }
      pos=pos+2;
    }
  }else{
    // Network mode
    size_t aux=0;
    for(size_t i=0;i<last_iter;i++){
      size_t numseed=(offset_SLs-THREADS_STREAM+aux)/data_host.nparticles;
      if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
	      keeptotal[data_host.seeds_ROI[numseed]]++;
      }
      aux++;
      pos=pos+2;
    }
  }
  if(opts.save_paths.value()){
    // save coordinates
    pos=0;
    for(size_t i=0;i<last_iter;i++){
      if(lengths_host[0][pos]>0||lengths_host[0][pos+1]>0){
        vector<float> tmp;
        if(lengths_host[0][pos]>0){
          size_t posSEED=i*data_host.nsteps*3;
          size_t posCURRENT=0;
          for(;posCURRENT<lengths_host[0][pos];posCURRENT++){
            tmp.push_back(paths_host[0][posSEED+posCURRENT*3]);
            tmp.push_back(paths_host[0][posSEED+posCURRENT*3+1]);
            tmp.push_back(paths_host[0][posSEED+posCURRENT*3+2]);
          }
        }
	      if(lengths_host[0][pos+1]>0){
          size_t pos2=i*data_host.nsteps*3+((data_host.nsteps/2)*3);
          size_t co=0;
          for(;co<lengths_host[0][pos+1];co++){
            tmp.push_back(paths_host[0][pos2+co*3]);
            tmp.push_back(paths_host[0][pos2+co*3+1]);
            tmp.push_back(paths_host[0][pos2+co*3+2]);
          }
	      }
	      m_save_paths.push_back(tmp);
      }
      pos=pos+2;
    }
  }
  if(opts.matrix3out.value()||opts.matrix3out.value()){
    write_mask3(last_iter,mat_crossed_host[0],mat_numcrossed_host[0],max_per_jump_mat,
		lrmat_crossed_host[0],lrmat_numcrossed_host[0],max_per_jump_lrmat,ConMat3,ConMat3b);
  }
  if(opts.matrix1out.value()||opts.matrix2out.value()){
    write_mask1(data_host,(offset_SLs-THREADS_STREAM),last_iter,
		lrmat_crossed_host[0],lrmat_numcrossed_host[0],max_per_jump_lrmat,ConMat1,ConMat1b);
  }

  if(opts.simpleout.value()){
    checkCuda(cudaMemcpy(*mprob_host,*mprob_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyDeviceToHost));
    size_t position=0;
    for(int z=0;z<data_host.Ssizes[2];z++){
      for(int y=0;y<data_host.Ssizes[1];y++){
	      for(int x=0;x<data_host.Ssizes[0];x++){
	        mprob[0](x,y,z)=mprob_host[0][position];
	        position++;
	      }
      }
    }
  } // Maybe I can change the pointer and avoid the copy !!!
  if(opts.simpleout.value()&&opts.omeanpathlength.value()){
    checkCuda(cudaMemcpy(*mprob2_host,*mprob2_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyDeviceToHost));
    size_t position=0;
    for(int z=0;z<data_host.Ssizes[2];z++){
      for(int y=0;y<data_host.Ssizes[1];y++){
	      for(int x=0;x<data_host.Ssizes[0];x++){
          mprob2[0](x,y,z)=mprob2_host[0][position];
          position++;
	      }
      }
    }
  }
  if(opts.opathdir.value()){
    checkCuda(cudaMemcpy(*mlocaldir_host,*mlocaldir_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*6*sizeof(float),cudaMemcpyDeviceToHost));
    size_t position=0;
    for(int z=0;z<data_host.Ssizes[2];z++){
      for(int y=0;y<data_host.Ssizes[1];y++){
	      for(int x=0;x<data_host.Ssizes[0];x++){
	        for(int v=0;v<6;v++){
            mlocaldir[0](x,y,z,v)=mlocaldir_host[0][position];
            position++;
	        }
	      }
      }
    }
  }

  if(opts.network.value()){
    int size_ConNet=(data_host.network.NVols+data_host.network.NSurfs)*(data_host.network.NVols+data_host.network.NSurfs);
    checkCuda(cudaMemcpy(*ConNet,*ConNet_gpu,size_ConNet*sizeof(float),cudaMemcpyDeviceToHost));
    if(opts.omeanpathlength.value()){
      checkCuda(cudaMemcpy(*ConNetb,*ConNetb_gpu,size_ConNet*sizeof(float),cudaMemcpyDeviceToHost));
    }
  }

  if(opts.s2tout.value()){
    long total_s2targets=data_host.nseeds*(data_host.targets.NVols+data_host.targets.NSurfs);
    checkCuda(cudaMemcpy(m_s2targets,*s2targets_gpu,total_s2targets*sizeof(float),cudaMemcpyDeviceToHost));
    if(opts.omeanpathlength.value()){
      checkCuda(cudaMemcpy(m_s2targetsb,*s2targetsb_gpu,total_s2targets*sizeof(float),cudaMemcpyDeviceToHost));
    }
  }

  //destroy streams
  for(int i=0;i<NSTREAMS;i++){
    checkCuda(cudaStreamDestroy(streams[i]));
  }
  checkCuda(cudaDeviceReset());
}

bool compare_Vertices(const float3 &a, const float3 &b){
  if(a.x<b.x) return true;
  if(a.x>b.x) return false;

  return (a.y<b.y);
}

void make_unique(vector<float3>& conns){
  //int 3 (x:id, y: triangle, z: value)
  sort(conns.begin(),conns.end(),compare_Vertices);
  vector<float3> conns2;
  for(unsigned int i=0;i<conns.size();i++){
    if(i>0){
      if( conns[i].x==conns[i-1].x && conns[i].y==conns[i-1].y) continue;
    }
    conns2.push_back(conns[i]);
  }
  conns=conns2;
}


////////////////////////
///// WRITE MASK  3 ////
////////////////////////
void write_mask3(
      unsigned long long 	nstreamlines,
			float3*			        mat_crossed_host,
			int* 			          mat_numcrossed_host,
			int			            max_per_jump_mat,
			float3*			        lrmat_crossed_host,
			int* 			          lrmat_numcrossed_host,
			int			            max_per_jump_lrmat,
			// Output
			float**			        ConMat3,
			float**			        ConMat3b)

{

  probtrackxOptions& opts =probtrackxOptions::getInstance();

  int nsteps=opts.nsteps.value();

  // Check if LR Matrix
  if(opts.lrmask3.value()==""){
    vector< float3 > inmask;
    vector< int > mytrianglesi; // List with the roi and triangles of an individual vertex i
    vector< int > mytrianglesj; // List with the roi and triangles of an individual vertex j
    float3 mytruple;
    for(unsigned long long sl=0;sl<nstreamlines;sl++){
      inmask.clear();
      for(int c=0; c<mat_numcrossed_host[sl];c++){
        mytruple.x=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].x;
        mytruple.y=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].y;  // Triangle id
        mytruple.z=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].z;  // Value
        inmask.push_back(mytruple);
      }
      make_unique(inmask);

      for(unsigned int i=0;i<inmask.size();i++){
        mytrianglesi.clear();
        int index=inmask[i].x;
        mytrianglesi.push_back(inmask[i].y);
        for(;(i+1)<inmask.size() && inmask[i+1].x==index;i++){  // Same vertix - different roi-triangle
          mytrianglesi.push_back(inmask[i+1].y);
        }
        unsigned int j=i+1;
        for(;j<inmask.size();j++){
          mytrianglesj.clear();
          index=inmask[j].x;
          mytrianglesj.push_back(inmask[j].y);
          for(;(j+1)<inmask.size() && inmask[j+1].x==index;j++){  // Same vertix - different roi-triangle
            mytrianglesj.push_back(inmask[j+1].y);
          }
          bool connect=false;
          for(unsigned int ii=0;ii<mytrianglesi.size()&&!connect;ii++){
            for(unsigned int jj=0;jj<mytrianglesj.size()&&!connect;jj++){
              if(mytrianglesi[ii]!=mytrianglesj[jj] || mytrianglesi[ii]==-1 || mytrianglesj[jj]==-1){
                // If is -1 is because it is not a vertex, it is a voxel
                connect=true;
	            }
	          }
	        }
          if(connect){
            if(opts.pathdist.value()||opts.omeanpathlength.value()){
              float val = fabs(inmask[i].z-inmask[j].z);
              ConMat3[(int)inmask[i].x][(int)inmask[j].x]=ConMat3[(int)inmask[i].x][(int)inmask[j].x]+val;
            }else{
              ConMat3[(int)inmask[i].x][(int)inmask[j].x]=ConMat3[(int)inmask[i].x][(int)inmask[j].x]+1;
            }
            if(opts.omeanpathlength.value()){
              ConMat3b[(int)inmask[i].x][(int)inmask[j].x]=ConMat3b[(int)inmask[i].x][(int)inmask[j].x]+1;
            }
	        }
	      }
      }
    }
  }else{
    vector< int > mytrianglesi; // List with the roi and triangles of an individual vertex i
    vector< int > mytrianglesj; // List with the roi and triangles of an individual vertex j
    float3 mytruple;
    for(unsigned long long sl=0;sl<nstreamlines;sl++){
      vector< float3 > inmask;
      for(int c=0; c<mat_numcrossed_host[sl];c++){
        mytruple.x=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].x;
        mytruple.y=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].y;  // Triangle id
        mytruple.z=mat_crossed_host[sl*nsteps*max_per_jump_mat+c].z;  // Value
	      inmask.push_back(mytruple);
      }
      make_unique(inmask);

      vector< float3 > inlrmask;
      for(int c=0; c<lrmat_numcrossed_host[sl];c++){
        mytruple.x=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].x;
        mytruple.y=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].y;  // Triangle id
        mytruple.z=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].z;  // Value
        inlrmask.push_back(mytruple);
      }
      make_unique(inlrmask);

      for(unsigned int i=0;i<inmask.size();i++){
        mytrianglesi.clear();
        int index=inmask[i].x;
        mytrianglesi.push_back(inmask[i].y);
	      for(;(i+1)<inmask.size() && inmask[i+1].x==index;i++){  // Same vertix - different roi-triangle
	        mytrianglesi.push_back(inmask[i+1].y);
	      }
	      for(unsigned j=0;j<inlrmask.size();j++){
          mytrianglesj.clear();
          index=inlrmask[j].x;
          mytrianglesj.push_back(inlrmask[j].y);
          for(;(j+1)<inlrmask.size() && inlrmask[j+1].x==index;j++){  // Same vertix - different roi-triangle
            mytrianglesj.push_back(inlrmask[j+1].y);
          }
          bool connect=false;
          for(unsigned int ii=0;ii<mytrianglesi.size()&&!connect;ii++){
            for(unsigned int jj=0;jj<mytrianglesj.size()&&!connect;jj++){
              if(mytrianglesi[ii]!=mytrianglesj[jj] || mytrianglesi[ii]==-1 || mytrianglesj[jj]==-1){
                // If is -1 is because is not a vertex, it is a voxel
		            connect=true;
              }
            }
          }
	        if(connect){
	          if(opts.pathdist.value()||opts.omeanpathlength.value()){
	            float val = fabs(inmask[i].z-inlrmask[j].z);
	            ConMat3[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat3[(int)inmask[i].x][(int)inlrmask[j].x]+val;
            }else{
              ConMat3[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat3[(int)inmask[i].x][(int)inlrmask[j].x]+1;
            }
            if(opts.omeanpathlength.value()){
              ConMat3b[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat3b[(int)inmask[i].x][(int)inlrmask[j].x]+1;
            }
	        }
	      }
      }
    }
  }
}

////////////////////////
///// WRITE MASK  1 ////
////////////////////////
void write_mask1(	tractographyData&	data_host,
			long long 		offset_SLs,
			unsigned long long 	nstreamlines,
			float3*			lrmat_crossed_host,
			int* 			lrmat_numcrossed_host,
			int			max_per_jump_lrmat,
			// Output
			float**			ConMat1,
			float**			ConMat1b)

{

  probtrackxOptions& opts =probtrackxOptions::getInstance();
  int nsteps=opts.nsteps.value();

  vector< int > mytrianglesi; // List with the roi and triangles of an individual vertex i
  vector< int > mytrianglesj; // List with the roi and triangles of an individual vertex j
  float3 mytruple;

  vector< float3 > inmask;  	// is only 1 loc (seed) but maybe different triangles
  vector< float3 > inlrmask;

  for(unsigned long long sl=0;sl<nstreamlines;sl++){
    size_t numseed = (offset_SLs+sl)/(data_host.nparticles);
    inmask.clear();
    for(int c=0; c<data_host.matrix1_Ntri[numseed];c++){
      mytruple.x=data_host.matrix1_locs[MAX_TRI_SEED*numseed+c];	// Loc
      mytruple.y=data_host.matrix1_idTri[MAX_TRI_SEED*numseed+c];  	// Triangle id
      mytruple.z=0;  							// Value
      inmask.push_back(mytruple);
    }
    //make_unique(inmask); // not needed here

    inlrmask.clear();
    //printf("%i crossed\n",lrmat_numcrossed_host[sl]);
    for(int c=0; c<lrmat_numcrossed_host[sl];c++){
      mytruple.x=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].x;  // Loc
      mytruple.y=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].y;  // Triangle id
      mytruple.z=lrmat_crossed_host[sl*nsteps*max_per_jump_lrmat+c].z;  // Value
      inlrmask.push_back(mytruple);
    }
    make_unique(inlrmask);

    for(unsigned int i=0;i<inmask.size();i++){
      mytrianglesi.clear();
      int index=inmask[i].x;
      mytrianglesi.push_back(inmask[i].y);
      for(;(i+1)<inmask.size() && inmask[i+1].x==index;i++){  // Same vertix - different roi-triangle
	      mytrianglesi.push_back(inmask[i+1].y);
      }
      for(unsigned j=0;j<inlrmask.size();j++){
        if(!opts.matrix2out.value() && inmask[i].x==inlrmask[j].x) continue; // Diagonal
        mytrianglesj.clear();
        index=inlrmask[j].x;
        mytrianglesj.push_back(inlrmask[j].y);
        for(;(j+1)<inlrmask.size() && inlrmask[j+1].x==index;j++){  // Same vertix - different roi-triangle
          mytrianglesj.push_back(inlrmask[j+1].y);
        }
        bool connect=false;
        for(unsigned int ii=0;ii<mytrianglesi.size()&&!connect;ii++){
          for(unsigned int jj=0;jj<mytrianglesj.size()&&!connect;jj++){
            if(mytrianglesi[ii]!=mytrianglesj[jj] || mytrianglesi[ii]==-1 || mytrianglesj[jj]==-1){
              // If is -1 is because is not a vertex, it is a voxel
              connect=true;
            }
          }
        }
        if(connect){
          if(opts.pathdist.value()||opts.omeanpathlength.value()){
            float val = fabs(inmask[i].z-inlrmask[j].z);
            ConMat1[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat1[(int)inmask[i].x][(int)inlrmask[j].x]+val;
          }else{
            //printf("CONN %i-%i\n",(int)inmask[i].x,(int)inlrmask[j].x);
            ConMat1[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat1[(int)inmask[i].x][(int)inlrmask[j].x]+1;
	        }
          if(opts.omeanpathlength.value()){
            ConMat1b[(int)inmask[i].x][(int)inlrmask[j].x]=ConMat1b[(int)inmask[i].x][(int)inlrmask[j].x]+1;
          }
	      }
      }
    }
  }
}
