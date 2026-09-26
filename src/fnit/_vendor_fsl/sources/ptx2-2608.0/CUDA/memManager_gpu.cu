/*  memManager_gpu.cu

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include "cuda.h"

#include "vector_types.h"

#include "newimage/newimage.h"

#include "probtrackxOptions.h"

using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;

cudaError_t checkCuda(cudaError_t result, const char *msg=NULL){
  if (result != cudaSuccess) {
    if (msg) fprintf(stderr, "Error: %s\n", msg);
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    exit(1);
  }
  return result;
}

void init_gpu(){
  int *q;
  checkCuda(cudaMalloc((void **)&q, sizeof(int)));
  checkCuda(cudaFree(q));

  int device;
  checkCuda(cudaGetDevice(&device));
  printf ("\n...................Allocated GPU %d...................\n", device);
  checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
}

void allocate_host_mem(
  // Input
  tractographyData&		data_host,
  long&                 MAX_SLs,           // MAX streamlines -  calculated here
  long&                 THREADS_STREAM,    // calculated here
  // Input - Output allocated mem
  int**                 lengths_host,
  float**               paths_host,
  float**               mprob_host,
  float**               mprob2_host,
  float**               mlocaldir_host,
  //float**             targvalues_host,
  //float**             targvaluesB_host,
  float3**              mat_crossed_host,
  int**                 mat_numcrossed_host,
  long long&            size_mat_cross,
  int&                  max_per_jump_mat,
  float3**              lrmat_crossed_host,
  int**                 lrmat_numcrossed_host,
  long long&            size_lrmat_cross,
  int&                  max_per_jump_lrmat)
{
  probtrackxOptions& opts=probtrackxOptions::getInstance();

  // calculate the maximum number of streamlines that can be executed in parallel

  size_t free,total;
  cuMemGetInfo(&free,&total); // in bytes
  size_t bytes_per_sl_STREAM=0; // needed for each STREAM (twice)
  size_t bytes_per_sl_COMMON=0; // needed in common to all STREAMS
  // Clamp usable memory to 32 GB to avoid pinned memory overflow
  const size_t max_usable_mem = 32L * 1024 * 1024 * 1024;
  if (free > max_usable_mem)
    free = max_usable_mem;

  if(!opts.save_paths.value()){
    // only for threads in a STREAM (can discard the coordinates of finished streamlines)
    bytes_per_sl_COMMON+= data_host.nsteps*3*sizeof(float);    // paths_gpu (3 floats per step - MAX Nsteps)
  }else{
    // for all the streamlines allocated
    bytes_per_sl_STREAM+= data_host.nsteps*3*sizeof(float);    // paths_gpu (3 floats per step - MAX Nsteps
  }

  bytes_per_sl_STREAM+= 2*sizeof(int);                // lengths_gpu (2 directions)
  bytes_per_sl_STREAM+= sizeof(curandState);            // random seed

  if(opts.simpleout.value()){
    free=free-data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float);  // m_prob
    bytes_per_sl_COMMON+= (data_host.nsteps)*sizeof(int); // beenhere
  }
  if(opts.omeanpathlength.value()&opts.simpleout.value()){
    free=free-data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float);  // m_prob2
  }
  if(opts.opathdir.value()){
    free=free-data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*6*sizeof(float);  // m_localdir
  }
  if(opts.network.value()){
    int nROIS=data_host.network.NVols+data_host.network.NSurfs;
    free=free-(nROIS*nROIS*sizeof(float)); //ConNet
    if(opts.omeanpathlength.value()){
      free=free-(nROIS*nROIS*sizeof(float)); //ConNetb
    }
    if(nROIS>maxNetsInShared){
      // Cannot use Shared Memory if too many ROIS, need Global memory for flags
      bytes_per_sl_COMMON+=(2*nROIS)*sizeof(float);
    }
  }
  if(opts.s2tout.value()){
    int nROIS=data_host.targets.NVols+data_host.targets.NSurfs;
    long total_s2targets=data_host.nseeds*nROIS;
    free=free-(total_s2targets*sizeof(float)); //matrix with results
    if(opts.omeanpathlength.value()){
      free=free-(total_s2targets*sizeof(float)); //s2targetsb
    }
    if(nROIS>maxTargsInShared){
      // Cannot use Shared Memory if too many ROIS, need Global memory for flags
      bytes_per_sl_COMMON+=(nROIS)*sizeof(float);
    }
  }
  if(opts.loopcheck.value()){
    bytes_per_sl_COMMON+= (data_host.nsteps/5)*sizeof(int);    // loopcheckkeys_gpu
    bytes_per_sl_COMMON+= (data_host.nsteps/5)*sizeof(float3); // loopcheckdirs_gpu
  }
  if(opts.matrix3out.value()){
    bytes_per_sl_STREAM+= 3*data_host.nsteps*sizeof(float3);   // mat_crossed_gpu
    //max is 3 by num_steps ... but it will never happens
    bytes_per_sl_STREAM+= sizeof(int);                   // mat_numcrossed_gpu
    if(opts.lrmask3.value()!=""){
      bytes_per_sl_STREAM+= 3*data_host.nsteps*sizeof(float3); // lrmat_crossed_gpu
      //3-> ... never will happens
      bytes_per_sl_STREAM+= sizeof(int);               // lrmat_numcrossed_gpu
    }
  }else if(opts.matrix1out.value()||opts.matrix2out.value()){
    bytes_per_sl_STREAM+= 3*data_host.nsteps*sizeof(float3);    // lrmat_crossed_gpu
    //3
    bytes_per_sl_STREAM+= sizeof(int);                // lrmat_numcrossed_gpu
  }
  free=free*FREEPERCENTAGE; // 80% defined in options.h
  MAX_SLs=(long)(free/(bytes_per_sl_STREAM+(bytes_per_sl_COMMON/NSTREAMS)));

  if(MAX_SLs%2) MAX_SLs++;
  unsigned long long totalSLs = (unsigned long long)data_host.nseeds*data_host.nparticles;
  if(totalSLs<MAX_SLs){
    MAX_SLs=totalSLs;
  }
  printf("Running %i streamlines in parallel using 2 STREAMS\n",MAX_SLs);
  THREADS_STREAM=MAX_SLs/NSTREAMS;   // paths_gpu just need to be a single structure if not save_paths (take a look !!)

  // Allocate in HOST
  checkCuda(cudaMallocHost((void**)lengths_host,2*THREADS_STREAM*sizeof(float))); // 2 paths per sample
  if(opts.save_paths.value()){ // if not.. discard it when finished streamline
    size_t num = THREADS_STREAM*data_host.nsteps*3;
    checkCuda(cudaMallocHost((void**)paths_host, num*sizeof(float)));
  }
  if(opts.simpleout.value())
    checkCuda(cudaMallocHost((void**)mprob_host,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
  if(opts.omeanpathlength.value()&opts.simpleout.value())
    checkCuda(cudaMallocHost((void**)mprob2_host,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
  if(opts.opathdir.value())
    checkCuda(cudaMallocHost((void**)mlocaldir_host,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*6*sizeof(float)));
  if(opts.matrix3out.value()){
    // If volumes overlap, it is possible to have more than 1 crossing voxel per jump
    if(data_host.matrix3.NSurfs){
      size_mat_cross=3*THREADS_STREAM*data_host.nsteps; // 3 vertices per jump (is this the maximum?)
      max_per_jump_mat=3;
    }else{
      //size_mat_cross=THREADS_STREAM*data_host.nsteps;
      //max_per_jump_mat=1;
      size_mat_cross=3*THREADS_STREAM*data_host.nsteps;
      max_per_jump_mat=3;
    }
    checkCuda(cudaMallocHost((void**)mat_crossed_host,size_mat_cross*sizeof(float3)));
    checkCuda(cudaMallocHost((void**)mat_numcrossed_host,THREADS_STREAM*sizeof(int)));
    if(opts.lrmask3.value()!=""){
      if(data_host.matrix3.NSurfs){
        size_lrmat_cross=3*THREADS_STREAM*data_host.nsteps; // 3 vertices per jump (is this the maximum?)
        max_per_jump_lrmat=3;
      }else{
        //size_lrmat_cross=THREADS_STREAM*data_host.nsteps;
        //max_per_jump_lrmat=1;
        size_lrmat_cross=3*THREADS_STREAM*data_host.nsteps;
        max_per_jump_lrmat=3;
      }
      checkCuda(cudaMallocHost((void**)lrmat_crossed_host,size_lrmat_cross*sizeof(float3)));
      checkCuda(cudaMallocHost((void**)lrmat_numcrossed_host,THREADS_STREAM*sizeof(int)));
    }
  }else if(opts.matrix1out.value()||opts.matrix2out.value()){
    if(data_host.lrmatrix1.NSurfs){
      size_lrmat_cross=3*THREADS_STREAM*data_host.nsteps; // 3 vertices per jump (is this the maximum?)
      max_per_jump_lrmat=3;
    }else{
      //size_lrmat_cross=THREADS_STREAM*data_host.nsteps;
      //max_per_jump_lrmat=1;
      size_lrmat_cross=3*THREADS_STREAM*data_host.nsteps;
      max_per_jump_lrmat=3;
    }
    checkCuda(cudaMallocHost((void**)lrmat_crossed_host,size_lrmat_cross*sizeof(float3)));
    checkCuda(cudaMallocHost((void**)lrmat_numcrossed_host,THREADS_STREAM*sizeof(int)));
  }
}

void allocate_gpu_mem(    tractographyData&     data_host,
                        long&            MAX_SLs,
                        long            THREADS_STREAM,
                        // Output
                        float**            mprob_gpu,
                        float**            mprob2_gpu,
                        float**            mlocaldir_gpu,
                        int**             beenhere_gpu,
                        float**            ConNet_gpu,
                        float**            ConNetb_gpu,
                        bool&            net_flags_in_shared,
                        float**            net_flags_gpu,
                        float**            net_values_gpu,
                        float**            s2targets_gpu,
                        float**            s2targetsb_gpu,
                        bool&            targ_flags_in_shared,
                        float**            targ_flags_gpu,
                        float**            paths_gpu,
                        int**             lengths_gpu,
                        // Loopcheck
                        int**            loopcheckkeys_gpu,
                        float3**        loopcheckdirs_gpu,
                        // Matrix
                        float3**         mat_crossed_gpu,
                        int**             mat_numcrossed_gpu,
                        int            size_mat_cross,
                        float3**         lrmat_crossed_gpu,
                        int**             lrmat_numcrossed_gpu,
                        int            size_lrmat_cross)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  int nsteps=opts.nsteps.value();

  // coordinate visited
  long long nbytes;
  if(!opts.save_paths.value()){
    // only for threads in a STREAM (can discard the coordinates of finished streamlines)
    nbytes=THREADS_STREAM*data_host.nsteps;
    nbytes*=3;
    nbytes*=sizeof(float);
    checkCuda(cudaMalloc((void**)paths_gpu,nbytes));
  }else{
    // for all the streamlines allocated
    nbytes=MAX_SLs*data_host.nsteps;
    nbytes*=3;
    nbytes*=sizeof(float);
    checkCuda(cudaMalloc((void**)paths_gpu,nbytes));
  }
  // path lenghts
  checkCuda(cudaMalloc((void**)lengths_gpu,MAX_SLs*2*sizeof(int)));

  // Map probabilities
  if(opts.simpleout.value()){
    checkCuda(cudaMalloc((void**)mprob_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemset(*mprob_gpu,0,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    // beenhere: to avoid 2 updates in same voxel
    long long size_beenhere = THREADS_STREAM;
    size_beenhere*=data_host.nsteps;
    checkCuda(cudaMalloc((void**)beenhere_gpu,size_beenhere*sizeof(int)));
  }
  if(opts.omeanpathlength.value()&&opts.simpleout.value()){
    checkCuda(cudaMalloc((void**)mprob2_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemset(*mprob2_gpu,0,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
  }
  // Map with average local tract orientations
  if(opts.opathdir.value()){
    checkCuda(cudaMalloc((void**)mlocaldir_gpu,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*6*sizeof(float)));
    checkCuda(cudaMemset(*mlocaldir_gpu,0,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*6*sizeof(float)));
  }
  if(opts.network.value()){
    // Network Matrix
    int nROIS=data_host.network.NVols+data_host.network.NSurfs;
    checkCuda(cudaMalloc((void**)ConNet_gpu,nROIS*nROIS*sizeof(float)));
    checkCuda(cudaMemset(*ConNet_gpu,0,nROIS*nROIS*sizeof(float)));
    if(opts.omeanpathlength.value()){
      checkCuda(cudaMalloc((void**)ConNetb_gpu,nROIS*nROIS*sizeof(float)));
      checkCuda(cudaMemset(*ConNetb_gpu,0,nROIS*nROIS*sizeof(float)));
    }
    // int maxNetsInShared= (24576-(6*THREADS_BLOCK)*sizeof(float))/(THREADS_BLOCK*2*sizeof(float));
    // [24KBytes (out of 48KB)   6 floats already allocated (coordinates)  2arrays (values & flags)
    // set to 8 in options.h
    if(nROIS>maxNetsInShared){
      net_flags_in_shared=false;
      // Flags for each thread to check if visited
      checkCuda(cudaMalloc((void**)net_flags_gpu,THREADS_STREAM*nROIS*sizeof(float)));
      checkCuda(cudaMalloc((void**)net_values_gpu,THREADS_STREAM*nROIS*sizeof(float)));
      checkCuda(cudaMemset(*net_flags_gpu,0,THREADS_STREAM*nROIS*sizeof(float)));
      checkCuda(cudaMemset(*net_values_gpu,0,THREADS_STREAM*nROIS*sizeof(float)));
    }else{
      net_flags_in_shared=true;
    }
  }
  // Seed to targets: this is for s2astext
  if(opts.s2tout.value()){
    int nROIS=data_host.targets.NVols+data_host.targets.NSurfs;
    long total_s2targets=data_host.nseeds*nROIS;
    checkCuda(cudaMalloc((void**)s2targets_gpu,total_s2targets*sizeof(float)));
    checkCuda(cudaMemset(*s2targets_gpu,0,total_s2targets*sizeof(float)));
    if(opts.omeanpathlength.value()){
      checkCuda(cudaMalloc((void**)s2targetsb_gpu,total_s2targets*sizeof(float)));
      checkCuda(cudaMemset(*s2targetsb_gpu,0,total_s2targets*sizeof(float)));
    }
    if(nROIS>maxTargsInShared){
      targ_flags_in_shared=false;
      // Flags for each thread to check if visited
      checkCuda(cudaMalloc((void**)targ_flags_gpu,THREADS_STREAM*nROIS*sizeof(float)));
      checkCuda(cudaMemset(*targ_flags_gpu,0,THREADS_STREAM*nROIS*sizeof(float)));
    }else{
      targ_flags_in_shared=true;
    }
  }

  if(opts.loopcheck.value()){
    checkCuda(cudaMalloc((void**)loopcheckkeys_gpu,(THREADS_STREAM*nsteps/5)*sizeof(int)));
    checkCuda(cudaMalloc((void**)loopcheckdirs_gpu,(THREADS_STREAM*nsteps/5)*sizeof(float3)));
  }

  // Connectivity Matrices
  if(opts.matrix3out.value()){
    checkCuda(cudaMalloc((void**)mat_crossed_gpu,NSTREAMS*size_mat_cross*sizeof(float3)));
    checkCuda(cudaMalloc((void**)mat_numcrossed_gpu,MAX_SLs*sizeof(int)));
    if(opts.lrmask3.value()!=""){
      checkCuda(cudaMalloc((void**)lrmat_crossed_gpu,NSTREAMS*size_lrmat_cross*sizeof(float3)));
      checkCuda(cudaMalloc((void**)lrmat_numcrossed_gpu,MAX_SLs*sizeof(int)));
    }
  }else if(opts.matrix1out.value()||opts.matrix2out.value()){
    checkCuda(cudaMalloc((void**)lrmat_crossed_gpu,NSTREAMS*size_lrmat_cross*sizeof(float3)));
    checkCuda(cudaMalloc((void**)lrmat_numcrossed_gpu,MAX_SLs*sizeof(int)));
  }
}

void copy_ToConstantMemory(tractographyData&    data_host)
{
  probtrackxOptions& opts=probtrackxOptions::getInstance();

  checkCuda(cudaMemcpyToSymbol(C_vox2mm,data_host.vox2mm,12*sizeof(float)));

  checkCuda(cudaMemcpyToSymbol(C_steplength,&(data_host.steplength),sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_distthresh,&(data_host.distthresh),sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_curv_thr,&(data_host.curv_thr),sizeof(float)));
  //cudaMemcpyToSymbol(C_fibthresh,&(data_host.fibthresh),sizeof(float));

  checkCuda(cudaMemcpyToSymbol(C_Sdims,data_host.Sdims,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_Ddims,data_host.Ddims,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_Wsampling_S2D_I,data_host.Wsampling_S2D_I,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_Wsampling_D2S_I,data_host.Wsampling_D2S_I,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_SsamplingI,data_host.SsamplingI,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_DsamplingI,data_host.DsamplingI,3*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_Seeds_to_DTI,data_host.Seeds_to_DTI,12*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_DTI_to_Seeds,data_host.DTI_to_Seeds,12*sizeof(float)));
  //checkCuda(cudaMemcpyToSymbol(C_Seeds_to_M2,data_host.Seeds_to_M2,12*sizeof(float)));
  checkCuda(cudaMemcpyToSymbol(C_Ssizes,data_host.Ssizes,3*sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(C_Dsizes,data_host.Dsizes,3*sizeof(int)));
  //checkCuda(cudaMemcpyToSymbol(C_M2sizes,data_host.M2sizes,3*sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(C_Warp_S2D_sizes,data_host.Warp_S2D_sizes,3*sizeof(int)));
  checkCuda(cudaMemcpyToSymbol(C_Warp_D2S_sizes,data_host.Warp_D2S_sizes,3*sizeof(int)));
  if(data_host.lrmatrix1.NVols){
    if(opts.matrix2out.value()){
      checkCuda(cudaMemcpyToSymbol(C_Seeds_to_M2,data_host.Seeds_to_M2,12*sizeof(float)));
      checkCuda(cudaMemcpyToSymbol(C_M2sizes,data_host.M2sizes,3*sizeof(int)));
    }
  }
}

void copy_ToTextureMemory(tractographyData& data_host)
{
  probtrackxOptions& opts=probtrackxOptions::getInstance();

  cudaArray *d_volumeArray1,*d_volumeArray2,*d_volumeArray3;
  cudaArray *d_volumeArray4,*d_volumeArray5,*d_volumeArray6;

  cudaResourceDesc  rdesc;
  cudaTextureDesc   tdesc;
  cudaMemcpy3DParms copyParams;

  memset(&rdesc,      0, sizeof(rdesc));
  memset(&tdesc,      0, sizeof(tdesc));
  memset(&copyParams, 0, sizeof(copyParams));

  rdesc.resType        = cudaResourceTypeArray;
  tdesc.readMode       = cudaReadModeElementType;
  tdesc.filterMode     = cudaFilterModePoint;
  tdesc.addressMode[0] = cudaAddressModeClamp;
  tdesc.addressMode[1] = cudaAddressModeClamp;
  tdesc.addressMode[2] = cudaAddressModeClamp;

  if(opts.seeds_to_dti.value()!="" && fsl_imageexists(opts.seeds_to_dti.value())){
    long size_warp=data_host.Warp_S2D_sizes[0]*data_host.Warp_S2D_sizes[1]*data_host.Warp_S2D_sizes[2];
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    const cudaExtent volumeSize= make_cudaExtent(data_host.Warp_S2D_sizes[0],data_host.Warp_S2D_sizes[1],data_host.Warp_S2D_sizes[2]);
    checkCuda(cudaMalloc3DArray(&d_volumeArray1,&channelDesc,volumeSize));
    checkCuda(cudaMalloc3DArray(&d_volumeArray2,&channelDesc,volumeSize));
    checkCuda(cudaMalloc3DArray(&d_volumeArray3,&channelDesc,volumeSize));

    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;

    rdesc.res.array.array = d_volumeArray1;
    copyParams.dstArray   = d_volumeArray1;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)data_host.SeedDTIwarp, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_SeedDTIwarp.x, &rdesc, &tdesc, NULL));

    rdesc.res.array.array = d_volumeArray2;
    copyParams.dstArray   = d_volumeArray2;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)&data_host.SeedDTIwarp[size_warp], volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_SeedDTIwarp.y, &rdesc, &tdesc, NULL));

    rdesc.res.array.array = d_volumeArray3;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)&data_host.SeedDTIwarp[2*size_warp], volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray   = d_volumeArray3;
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_SeedDTIwarp.z, &rdesc, &tdesc, NULL));
  }
  if(opts.dti_to_seeds.value()!="" && fsl_imageexists(opts.dti_to_seeds.value())){
    long size_warp=data_host.Warp_D2S_sizes[0]*data_host.Warp_D2S_sizes[1]*data_host.Warp_D2S_sizes[2];
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    const cudaExtent volumeSize2= make_cudaExtent(data_host.Warp_D2S_sizes[0],data_host.Warp_D2S_sizes[1],data_host.Warp_D2S_sizes[2]);
    checkCuda(cudaMalloc3DArray(&d_volumeArray4,&channelDesc,volumeSize2));
    checkCuda(cudaMalloc3DArray(&d_volumeArray5,&channelDesc,volumeSize2));
    checkCuda(cudaMalloc3DArray(&d_volumeArray6,&channelDesc,volumeSize2));

    copyParams.extent = volumeSize2;
    copyParams.kind   = cudaMemcpyHostToDevice;

    rdesc.res.array.array = d_volumeArray4;
    copyParams.dstArray   = d_volumeArray4;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)data_host.DTISeedwarp, volumeSize2.width*sizeof(float), volumeSize2.width, volumeSize2.height);
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_DTISeedwarp.x, &rdesc, &tdesc, NULL));

    rdesc.res.array.array = d_volumeArray5;
    copyParams.dstArray   = d_volumeArray5;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)&data_host.DTISeedwarp[size_warp], volumeSize2.width*sizeof(float), volumeSize2.width, volumeSize2.height);
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_DTISeedwarp.y, &rdesc, &tdesc, NULL));

    rdesc.res.array.array = d_volumeArray6;
    copyParams.dstArray   = d_volumeArray6;
    copyParams.srcPtr     = make_cudaPitchedPtr((void*)&data_host.DTISeedwarp[2*size_warp], volumeSize2.width*sizeof(float), volumeSize2.width, volumeSize2.height);
    checkCuda(cudaMemcpy3D(&copyParams));
    checkCuda(cudaCreateTextureObject(&T_DTISeedwarp.z, &rdesc, &tdesc, NULL));
  }
}

size_t calculate_mem_required(tractographyData&    data_host){
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  size_t total_mem_required;
  total_mem_required = sizeof(tractographyData);
  total_mem_required += data_host.nseeds*3*sizeof(float); // seeds
  if(opts.network.value()) total_mem_required += data_host.nseeds*sizeof(float); // network
  total_mem_required += 2*data_host.Dsizes[0]*data_host.Dsizes[1]*data_host.Dsizes[2]*sizeof(float); //mask & lut_vol2mat
  total_mem_required += 3*data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float); // samples

  size_t seedVolSize=data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float);
  size_t voxFacesIndexSize = (data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int);
  if(data_host.avoid.NVols) total_mem_required += seedVolSize;
  if(data_host.avoid.NSurfs){
    size_t surfSize = size_t(data_host.avoid.sizesStr[1]*sizeof(float)) +  data_host.avoid.sizesStr[2]*sizeof(int)
      + data_host.avoid.sizesStr[3]*sizeof(int) + voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.stop.NVols) total_mem_required += seedVolSize;
  if(data_host.stop.NSurfs){
    size_t surfSize = size_t(data_host.stop.sizesStr[1]*sizeof(float)) +  data_host.stop.sizesStr[2]*sizeof(int)
      + data_host.stop.sizesStr[3]*sizeof(int) + voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.wtstop.NVols)  total_mem_required += data_host.wtstop.NVols * seedVolSize;
  if(data_host.wtstop.NSurfs){
    size_t surfSize = size_t(data_host.wtstop.sizesStr[1]*sizeof(float)) +  data_host.wtstop.sizesStr[2]*sizeof(int)
      + data_host.wtstop.sizesStr[3]*sizeof(int) + data_host.wtstop.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.network.NVols) total_mem_required += data_host.network.NVols * seedVolSize;
  if(data_host.network.NSurfs){
    size_t surfSize = size_t(data_host.network.sizesStr[1]*sizeof(float)) +  data_host.network.sizesStr[2]*sizeof(int)
      + data_host.network.sizesStr[3]*sizeof(int) + data_host.network.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.network.NVols||data_host.network.NSurfs){
    total_mem_required += data_host.network.NVols+data_host.network.NSurfs*sizeof(int);
  }
  if(data_host.networkREF.NVols) total_mem_required += seedVolSize;
  if(data_host.networkREF.NSurfs){
    size_t surfSize = size_t(data_host.networkREF.sizesStr[1]*sizeof(float)) +  data_host.networkREF.sizesStr[2]*sizeof(int)
      + data_host.networkREF.sizesStr[3]*sizeof(int) + voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.waypoint.NVols) total_mem_required += data_host.waypoint.NVols * seedVolSize;
  if(data_host.waypoint.NSurfs){
    size_t surfSize = size_t(data_host.waypoint.sizesStr[1]*sizeof(float)) +  data_host.waypoint.sizesStr[2]*sizeof(int)
      + data_host.waypoint.sizesStr[3]*sizeof(int) + data_host.waypoint.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.waypoint.NVols||data_host.waypoint.NSurfs){
    total_mem_required += data_host.waypoint.NVols+data_host.waypoint.NSurfs*sizeof(int);
  }
  if(data_host.targets.NVols) total_mem_required += data_host.targets.NVols * seedVolSize;
  if(data_host.targets.NSurfs){
    size_t surfSize = size_t(data_host.targets.sizesStr[1]*sizeof(float)) +  data_host.targets.sizesStr[2]*sizeof(int)
      + data_host.targets.sizesStr[3]*sizeof(int) + data_host.targets.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.targets.NVols||data_host.targets.NSurfs){
    total_mem_required += data_host.targets.NVols+data_host.targets.NSurfs*sizeof(int);
  }
  if(data_host.targetsREF.NVols) total_mem_required += seedVolSize;
  if(data_host.targetsREF.NSurfs){
    size_t surfSize = size_t(data_host.targetsREF.sizesStr[1]*sizeof(float)) +  data_host.targetsREF.sizesStr[2]*sizeof(int)
      + data_host.targetsREF.sizesStr[3]*sizeof(int) + voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.lrmatrix1.NVols){
    if(opts.matrix2out.value())
      total_mem_required += data_host.lrmatrix1.NVols*data_host.M2sizes[0]*data_host.M2sizes[1]*data_host.M2sizes[2]*sizeof(float);
    else
      total_mem_required += data_host.lrmatrix1.NVols * seedVolSize;
  }
  if(data_host.lrmatrix1.NSurfs){
    size_t surfSize = size_t(data_host.lrmatrix1.sizesStr[0]*sizeof(int))
      + data_host.lrmatrix1.sizesStr[1]*sizeof(float) +  data_host.lrmatrix1.sizesStr[2]*sizeof(int)
      + data_host.lrmatrix1.sizesStr[3]*sizeof(int) + data_host.lrmatrix1.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.matrix3.NVols) total_mem_required += data_host.matrix3.NVols * seedVolSize;
  if(data_host.matrix3.NSurfs){
    size_t surfSize = size_t(data_host.matrix3.sizesStr[0]*sizeof(int))
      + data_host.matrix3.sizesStr[1]*sizeof(float) +  data_host.matrix3.sizesStr[2]*sizeof(int)
      + data_host.matrix3.sizesStr[3]*sizeof(int) + data_host.matrix3.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  if(data_host.lrmatrix3.NVols) total_mem_required += data_host.lrmatrix3.NVols * seedVolSize;
  if(data_host.lrmatrix3.NSurfs){
    size_t surfSize = size_t(data_host.lrmatrix3.sizesStr[0]*sizeof(int))
      + data_host.lrmatrix3.sizesStr[1]*sizeof(float) +  data_host.lrmatrix3.sizesStr[2]*sizeof(int)
      + data_host.lrmatrix3.sizesStr[3]*sizeof(int) + data_host.lrmatrix3.NSurfs*voxFacesIndexSize;
    total_mem_required += surfSize;
  }
  return total_mem_required;
}


void copy_to_gpu(     tractographyData&    data_host,
                    tractographyData*&    data_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  size_t total_mem_required = calculate_mem_required(data_host);
  cout << "Memory required for allocating data (MB): "<< total_mem_required/1048576 << endl;

  size_t free,total;
  cudaMemGetInfo(&free,&total);
  if(total_mem_required > free){
    cout << "Not enough Memory available on device. Exiting ..." << endl;
    terminate();
  }

  checkCuda(cudaMalloc((void**)&data_gpu,sizeof(tractographyData)));
  checkCuda(cudaMemcpy(data_gpu,&data_host,sizeof(tractographyData),cudaMemcpyHostToDevice));

  int* auxI;
  float* auxF;

  // seeds
  checkCuda(cudaMalloc((void**)&auxF,data_host.nseeds*3*sizeof(float)), "Allocating seeds");
  checkCuda(cudaMemcpy(auxF,data_host.seeds,data_host.nseeds*3*sizeof(float),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->seeds,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  if(opts.network.value()){
    checkCuda(cudaMalloc((void**)&auxF,data_host.nseeds*sizeof(float)), "Allocating Network ROIs");
    checkCuda(cudaMemcpy(auxF,data_host.seeds_ROI,data_host.nseeds*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->seeds_ROI,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  // mask
  checkCuda(cudaMalloc((void**)&auxF,data_host.Dsizes[0]*data_host.Dsizes[1]*data_host.Dsizes[2]*sizeof(float)), "Allocating mask");
  checkCuda(cudaMemcpy(auxF,data_host.mask,data_host.Dsizes[0]*data_host.Dsizes[1]*data_host.Dsizes[2]*sizeof(float),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->mask,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  // th_samples
  checkCuda(cudaMalloc((void**)&auxF,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float)), "Allocating th_samples");
  checkCuda(cudaMemcpy(auxF,data_host.thsamples,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->thsamples,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  // ph_samples
  checkCuda(cudaMalloc((void**)&auxF,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float)), "Allocating ph_samples");
  checkCuda(cudaMemcpy(auxF,data_host.phsamples,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->phsamples,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  // f_samples
  checkCuda(cudaMalloc((void**)&auxF,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float)), "Allocating f_samples");
  checkCuda(cudaMemcpy(auxF,data_host.fsamples,data_host.nfibres*data_host.nsamples*data_host.nvoxels*sizeof(float),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->fsamples,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  // lut_vol2mat
  checkCuda(cudaMalloc((void**)&auxI,data_host.Dsizes[0]*data_host.Dsizes[1]*data_host.Dsizes[2]*sizeof(int)), "Allocating lut_vol2mat");
  checkCuda(cudaMemcpy(auxI,data_host.lut_vol2mat,data_host.Dsizes[0]*data_host.Dsizes[1]*data_host.Dsizes[2]*sizeof(int),cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(&data_gpu->lut_vol2mat,&auxI,sizeof(int*),cudaMemcpyHostToDevice));

  //Avoid mask
  if(data_host.avoid.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.avoid.volume,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->avoid.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.avoid.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.avoid.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.avoid.vertices,data_host.avoid.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->avoid.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.avoid.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.avoid.faces,data_host.avoid.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->avoid.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.avoid.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.avoid.VoxFaces,data_host.avoid.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->avoid.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.avoid.VoxFacesIndex,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->avoid.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Stop mask
  if(data_host.stop.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.stop.volume,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->stop.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.stop.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.stop.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.stop.vertices,data_host.stop.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->stop.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.stop.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.stop.faces,data_host.stop.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->stop.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.stop.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.stop.VoxFaces,data_host.stop.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->stop.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.stop.VoxFacesIndex,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->stop.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Wtstop mask
  if(data_host.wtstop.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.wtstop.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.wtstop.volume,data_host.wtstop.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->wtstop.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.wtstop.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.wtstop.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.wtstop.vertices,data_host.wtstop.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->wtstop.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.wtstop.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.wtstop.faces,data_host.wtstop.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->wtstop.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.wtstop.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.wtstop.VoxFaces,data_host.wtstop.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->wtstop.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.wtstop.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.wtstop.VoxFacesIndex,(data_host.wtstop.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->wtstop.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Network mask
  if(data_host.network.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.network.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.network.volume,data_host.network.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.network.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.network.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.network.vertices,data_host.network.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.network.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.network.faces,data_host.network.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.network.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.network.VoxFaces,data_host.network.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.network.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.network.VoxFacesIndex,(data_host.network.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  if(data_host.network.NVols||data_host.network.NSurfs){
    int totalrois=data_host.network.NVols+data_host.network.NSurfs;
    checkCuda(cudaMalloc((void**)&auxI,totalrois*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.network.IndexRoi,totalrois*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->network.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Reference Network mask
  if(data_host.networkREF.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.networkREF.volume,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->networkREF.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.networkREF.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.networkREF.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.networkREF.vertices,data_host.networkREF.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->networkREF.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.networkREF.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.networkREF.faces,data_host.networkREF.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->networkREF.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.networkREF.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.networkREF.VoxFaces,data_host.networkREF.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->networkREF.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.networkREF.VoxFacesIndex,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->networkREF.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }

  // Waypoints mask
  if(data_host.waypoint.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.waypoint.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.waypoint.volume,data_host.waypoint.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.waypoint.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.waypoint.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.waypoint.vertices,data_host.waypoint.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.waypoint.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.waypoint.faces,data_host.waypoint.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.waypoint.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.waypoint.VoxFaces,data_host.waypoint.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.waypoint.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.waypoint.VoxFacesIndex,(data_host.waypoint.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  if(data_host.waypoint.NVols||data_host.waypoint.NSurfs){
    int totalrois=data_host.waypoint.NVols+data_host.waypoint.NSurfs;
    checkCuda(cudaMalloc((void**)&auxI,totalrois*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.waypoint.IndexRoi,totalrois*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->waypoint.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Target mask
  if(data_host.targets.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.targets.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.targets.volume,data_host.targets.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.targets.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.targets.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.targets.vertices,data_host.targets.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.targets.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targets.faces,data_host.targets.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.targets.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targets.VoxFaces,data_host.targets.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.targets.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targets.VoxFacesIndex,(data_host.targets.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  if(data_host.targets.NVols||data_host.targets.NSurfs){
    int totalrois=data_host.targets.NVols+data_host.targets.NSurfs;
    checkCuda(cudaMalloc((void**)&auxI,totalrois*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targets.IndexRoi,totalrois*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targets.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Reference Targets mask
  if(data_host.targetsREF.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.targetsREF.volume,data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targetsREF.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.targetsREF.NSurfs){
    checkCuda(cudaMalloc((void**)&auxF,data_host.targetsREF.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.targetsREF.vertices,data_host.targetsREF.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targetsREF.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.targetsREF.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targetsREF.faces,data_host.targetsREF.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targetsREF.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.targetsREF.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targetsREF.VoxFaces,data_host.targetsREF.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targetsREF.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.targetsREF.VoxFacesIndex,(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1)*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->targetsREF.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
  }
  // Matrix 1
  // LRMatrix 1
  if(data_host.lrmatrix1.NVols){
    if(opts.matrix2out.value()){
      checkCuda(cudaMalloc((void**)&auxF,data_host.lrmatrix1.NVols*data_host.M2sizes[0]*data_host.M2sizes[1]*data_host.M2sizes[2]*sizeof(float)));
      checkCuda(cudaMemcpy(auxF,data_host.lrmatrix1.volume,data_host.lrmatrix1.NVols*data_host.M2sizes[0]*data_host.M2sizes[1]*data_host.M2sizes[2]*sizeof(float),cudaMemcpyHostToDevice));
      checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    }else{
      checkCuda(cudaMalloc((void**)&auxF,data_host.lrmatrix1.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
      checkCuda(cudaMemcpy(auxF,data_host.lrmatrix1.volume,data_host.lrmatrix1.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
      checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    }
  }
  if(data_host.lrmatrix1.NSurfs){
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix1.sizesStr[0]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix1.locs,data_host.lrmatrix1.sizesStr[0]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.locs,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxF,data_host.lrmatrix1.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.lrmatrix1.vertices,data_host.lrmatrix1.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix1.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix1.faces,data_host.lrmatrix1.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix1.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix1.VoxFaces,data_host.lrmatrix1.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.lrmatrix1.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix1.VoxFacesIndex,(data_host.lrmatrix1.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix1.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    //cudaMalloc((void**)&auxI,data_host.lrmatrix1.sizesStr[4]*sizeof(int));
    //cudaMemcpy(auxI,data_host.lrmatrix1.IndexRoi,data_host.lrmatrix1.sizesStr[4]*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(&data_gpu->lrmatrix1.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice);
  }
  // Matrix 3
  if(data_host.matrix3.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.matrix3.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.matrix3.volume,data_host.matrix3.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.matrix3.NSurfs){
    checkCuda(cudaMalloc((void**)&auxI,data_host.matrix3.sizesStr[0]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.matrix3.locs,data_host.matrix3.sizesStr[0]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.locs,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxF,data_host.matrix3.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.matrix3.vertices,data_host.matrix3.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.matrix3.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.matrix3.faces,data_host.matrix3.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.matrix3.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.matrix3.VoxFaces,data_host.matrix3.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.matrix3.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.matrix3.VoxFacesIndex,(data_host.matrix3.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->matrix3.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    //cudaMalloc((void**)&auxI,data_host.matrix3.sizesStr[4]*sizeof(int));
    //cudaMemcpy(auxI,data_host.matrix3.IndexRoi,data_host.matrix3.sizesStr[4]*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(&data_gpu->matrix3.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice);
  }
  // LRMatrix 3
  if(data_host.lrmatrix3.NVols){
    checkCuda(cudaMalloc((void**)&auxF,data_host.lrmatrix3.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.lrmatrix3.volume,data_host.lrmatrix3.NVols*data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.volume,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
  }
  if(data_host.lrmatrix3.NSurfs){
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix3.sizesStr[0]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix3.locs,data_host.lrmatrix3.sizesStr[0]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.locs,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxF,data_host.lrmatrix3.sizesStr[1]*sizeof(float)));
    checkCuda(cudaMemcpy(auxF,data_host.lrmatrix3.vertices,data_host.lrmatrix3.sizesStr[1]*sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.vertices,&auxF,sizeof(float*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix3.sizesStr[2]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix3.faces,data_host.lrmatrix3.sizesStr[2]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.faces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,data_host.lrmatrix3.sizesStr[3]*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix3.VoxFaces,data_host.lrmatrix3.sizesStr[3]*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.VoxFaces,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc((void**)&auxI,(data_host.lrmatrix3.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int)));
    checkCuda(cudaMemcpy(auxI,data_host.lrmatrix3.VoxFacesIndex,(data_host.lrmatrix3.NSurfs*(data_host.Ssizes[0]*data_host.Ssizes[1]*data_host.Ssizes[2]+1))*sizeof(int),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(&data_gpu->lrmatrix3.VoxFacesIndex,&auxI,sizeof(int*),cudaMemcpyHostToDevice));
    //cudaMalloc((void**)&auxI,data_host.lrmatrix3.sizesStr[4]*sizeof(int));
    //cudaMemcpy(auxI,data_host.lrmatrix3.IndexRoi,data_host.lrmatrix3.sizesStr[4]*sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(&data_gpu->lrmatrix3.IndexRoi,&auxI,sizeof(int*),cudaMemcpyHostToDevice);
  }
}
