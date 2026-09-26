/*  tractography_CallKernels.cu

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

void initialise_SeedsGPU(
  curandState*& devStates,
  int           THREADS_STREAM)
{
  int blocks_Rand = THREADS_STREAM/THREADS_BLOCK_RAND;
  if(THREADS_STREAM%THREADS_BLOCK_RAND) blocks_Rand++;
  checkCuda(cudaMalloc(&devStates,blocks_Rand*THREADS_BLOCK_RAND*sizeof(curandState)));
  dim3 Dim_Grid_Rand(blocks_Rand,1);
  dim3 Dim_Block_Rand(THREADS_BLOCK_RAND,1);
  setup_randoms_kernel <<<Dim_Grid_Rand,Dim_Block_Rand>>>(devStates,rand());
}

void calculate_path(
  cudaStream_t      stream,
  tractographyData* data_gpu,
  int               num_threads,
  curandState*      devStates,
  long long         offset_SLs,
  int**             loopcheckkeys_gpu,
  float3**          loopcheckdirs_gpu,
  //OUTPUT
  float*            paths_gpu,
  int*              lengths_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  if(!opts.modeuler.value()){
    if(opts.randfib.value()==0){
      if(opts.loopcheck.value()){
        get_path_kernel<0,true,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      }else{
        get_path_kernel<0,false,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      }
    }else if(opts.randfib.value()==1){
      if(opts.loopcheck.value())
        get_path_kernel<1,true,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<1,false,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else if(opts.randfib.value()==2){
      if(opts.loopcheck.value())
        get_path_kernel<2,true,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<2,false,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else if(opts.randfib.value()==3){
      if(opts.loopcheck.value())
        get_path_kernel<3,true,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<3,false,false> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else{
      printf("RANDFIB cannot be higher than 3\n"); exit(1);
    }
  }else{
    if(opts.randfib.value()==0){
      if(opts.loopcheck.value()){
        get_path_kernel<0,true,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      }else{
        get_path_kernel<0,false,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      }
    }else if(opts.randfib.value()==1){
      if(opts.loopcheck.value())
        get_path_kernel<1,true,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<1,false,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else if(opts.randfib.value()==2){
      if(opts.loopcheck.value())
        get_path_kernel<2,true,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<2,false,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else if(opts.randfib.value()==3){
      if(opts.loopcheck.value())
        get_path_kernel<3,true,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
      else
        get_path_kernel<3,false,true> <<<nblocks,THREADS_BLOCK,0,stream>>>
          (data_gpu,num_threads,devStates,offset_SLs,
           *loopcheckkeys_gpu,*loopcheckdirs_gpu,
           T_SeedDTIwarp,
           T_DTISeedwarp,
           paths_gpu,lengths_gpu);
    }else{
      printf("RANDFIB cannot be higher than 3\n"); exit(1);
    }
  }
}

void stop_mask(
  cudaStream_t            stream,
  tractographyData&    data_host,
  tractographyData*    data_gpu,
  int                                num_threads,
  float*                        paths_gpu,
  int*                            lengths_gpu)
{
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  if(data_host.stop.NVols&&data_host.stop.NSurfs){
    stop_masks_kernel<true,true><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.stop.NVols){
    stop_masks_kernel<true,false><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.stop.NSurfs){
    stop_masks_kernel<false,true><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }
}

void wtstop_masks(
  cudaStream_t            stream,
  tractographyData&    data_host,
  tractographyData*    data_gpu,
  int                                num_threads,
  float*                        paths_gpu,
  int*                            lengths_gpu)
{
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;
  int shared = THREADS_BLOCK*(data_host.wtstop.NVols+data_host.wtstop.NSurfs)*sizeof(char) + (6*THREADS_BLOCK)*sizeof(float);

  if(data_host.wtstop.NVols && data_host.wtstop.NSurfs){
    wtstop_masks_kernel<true,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.wtstop.NVols){
    wtstop_masks_kernel<true,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.wtstop.NSurfs){
    wtstop_masks_kernel<false,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }
}

void avoid_mask(
  cudaStream_t            stream,
  tractographyData&    data_host,
  tractographyData*    data_gpu,
  int                                num_threads,
  float*                        paths_gpu,
  int*                            lengths_gpu)
{
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  if(data_host.avoid.NVols&&data_host.avoid.NSurfs){
    avoid_masks_kernel<true,true><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.avoid.NVols){
    avoid_masks_kernel<true,false><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.avoid.NSurfs){
    avoid_masks_kernel<false,true><<<nblocks,THREADS_BLOCK,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }
}

void way_masks(
  cudaStream_t                stream,
  tractographyData&        data_host,
  tractographyData*        data_gpu,
  int                                    num_threads,
  float*                            paths_gpu,
  int*                                lengths_gpu)
{
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;
  int shared = THREADS_BLOCK*(data_host.waypoint.NVols + data_host.waypoint.NSurfs)*sizeof(char) + (6*THREADS_BLOCK)*sizeof(float);

  if(data_host.waypoint.NVols && data_host.waypoint.NSurfs){
    way_masks_kernel<true,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.waypoint.NVols){
    way_masks_kernel<true,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }else if(data_host.waypoint.NSurfs){
    way_masks_kernel<false,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu);
  }
}

void net_masks(
  cudaStream_t                stream,
  tractographyData&        data_host,
  tractographyData*        data_gpu,
  int                                    num_threads,
  long long                        offset_SLs,
  float*                            paths_gpu,
  int*                                lengths_gpu,
  float**                            ConNet_gpu,
  float**                            ConNetb_gpu,
  bool                                net_flags_in_shared,
  float**                         net_flags_gpu,
  // Maybe not allocated memory, only if not enough space in Shared
  float**                            net_values_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  int shared=0;

  // If enough space in shared, then allocate flags there, otherwise, allocate memory in Global memory
  if(net_flags_in_shared){
    shared = THREADS_BLOCK*2*(data_host.network.NVols + data_host.network.NSurfs)*sizeof(float)+ (6*THREADS_BLOCK)*sizeof(float);
  }else{
    shared = (6*THREADS_BLOCK)*sizeof(float);
  }
  int savelength=0;
  if(opts.omeanpathlength.value()) savelength=2;
  else if(opts.pathdist.value()) savelength=1;

  if(net_flags_in_shared){
    if(savelength==0){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }else if(savelength==1){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }else if(savelength==2){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }
  }else{
    if(savelength==0){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }else if(savelength==1){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }else if(savelength==2){
      if(data_host.network.NVols && data_host.network.NSurfs){
        net_masks_kernel<true,true,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NVols){
        net_masks_kernel<true,false,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }else if(data_host.network.NSurfs){
        net_masks_kernel<false,true,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>
          (data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*ConNet_gpu,*ConNetb_gpu,*net_flags_gpu,*net_values_gpu);
      }
    }
  }
}

void targets_masks(
  cudaStream_t                        stream,
  tractographyData&                data_host,
  tractographyData*                data_gpu,
  int                                            num_threads,
  long long                                offset_SLs,
  float*                                    paths_gpu,
  int*                                        lengths_gpu,
  float**                                    s2targets_gpu,
  float**                                    s2targetsb_gpu,
  bool                                        targ_flags_in_shared,
  // Maybe not allocated memory, only if not enough space in Shared
  float**                                 targ_flags_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  int shared = 0;
  // If enough space in shared, then allocate flags there, otherwise, allocate memory in Global memory
  if(targ_flags_in_shared){
    shared = THREADS_BLOCK*(data_host.targets.NVols + data_host.targets.NSurfs)*sizeof(float)+ (6*THREADS_BLOCK)*sizeof(float);
  }else{
    shared = (6*THREADS_BLOCK)*sizeof(float);
  }

  int savelength=0;
  if(opts.omeanpathlength.value()) savelength=2;
  else if(opts.pathdist.value()) savelength=1;

  if(targ_flags_in_shared){
    if(savelength==0){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,0,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }else if(savelength==1){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,1,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }else if(savelength==2){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,2,true><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }
  }else{
    //!targ_flags_in_shared
    if(savelength==0){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,0,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }else if(savelength==1){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,1,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }else if(savelength==2){
      if(data_host.targets.NVols && data_host.targets.NSurfs){
        targets_masks_kernel<true,true,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NVols){
        targets_masks_kernel<true,false,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }else if(data_host.targets.NSurfs){
        targets_masks_kernel<false,true,2,false><<<nblocks,THREADS_BLOCK,shared,stream>>>(data_gpu,num_threads,offset_SLs,paths_gpu,lengths_gpu,*s2targets_gpu,*s2targetsb_gpu,*targ_flags_gpu);
      }
    }
  }
}

void update_path(
  cudaStream_t            stream,
  tractographyData*    data_gpu,
  int                                num_threads,
  float*                        paths_gpu,
  int*                            lengths_gpu,
  int**                         beenhere_gpu,
  int                             update_upper_limit,
  float**                     mprob_gpu,
  float**                     mprob2_gpu,
  float**                        mlocaldir_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  int nblocks=num_threads/THREADS_BLOCK_UPDATE;
  if(num_threads%THREADS_BLOCK_UPDATE) nblocks++;

  if(!opts.opathdir.value()){
    if(!opts.omeanpathlength.value()){
      if(!opts.pathdist.value()){
        update_path_kernel<false,false,false><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                         *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }else{
        update_path_kernel<true,false,false><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                        *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }
    }else{
      if(!opts.pathdist.value()){
        update_path_kernel<false,true,false><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                        *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }else{
        update_path_kernel<true,true,false><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                       *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }
    }
  }else{
    if(!opts.omeanpathlength.value()){
      if(!opts.pathdist.value()){
        update_path_kernel<false,false,true><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                        *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }else{
        update_path_kernel<true,false,true><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                       *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }
    }else{
      if(!opts.pathdist.value()){
        update_path_kernel<false,true,true><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                       *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }else{
        update_path_kernel<true,true,true><<<nblocks,THREADS_BLOCK_UPDATE,0,stream>>>(data_gpu,num_threads,paths_gpu,lengths_gpu,
                                                                                      *beenhere_gpu,update_upper_limit,*mprob_gpu,*mprob2_gpu,*mlocaldir_gpu);
      }
    }
  }
}

void matrix1(
  cudaStream_t                stream,
  tractographyData&        data_host,
  tractographyData*        data_gpu,
  int                                    num_threads,
  float*                            paths_gpu,
  int*                                lengths_gpu,
  float3*                         mat_crossed_gpu,
  int*                                 mat_numcrossed_gpu,
  float3*                         lrmat_crossed_gpu,
  int*                                 lrmat_numcrossed_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  // LRMATRIX 1: but not Matrix 2 with a different resolution
  if(opts.matrix2out.value()){
    if(data_host.lrmatrix1.NVols && data_host.lrmatrix1.NSurfs){
      matrix_kernel<true,true,true><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix1.NVols){
      matrix_kernel<true,false,true><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix1.NSurfs){
      matrix_kernel<false,true,true><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }
  }else{
    if(data_host.lrmatrix1.NVols && data_host.lrmatrix1.NSurfs){
      matrix_kernel<true,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix1.NVols){
      matrix_kernel<true,false,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix1.NSurfs){
      matrix_kernel<false,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix1,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }
  }
}

void matrix3(
  cudaStream_t           stream,
  tractographyData&        data_host,
  tractographyData*        data_gpu,
  int                            num_threads,
  float*                    paths_gpu,
  int*                        lengths_gpu,
  float3*                         mat_crossed_gpu,
  int*                         mat_numcrossed_gpu,
  float3*                         lrmat_crossed_gpu,
  int*                         lrmat_numcrossed_gpu)
{
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  int nblocks=num_threads/THREADS_BLOCK;
  if(num_threads%THREADS_BLOCK) nblocks++;

  if(data_host.matrix3.NVols && data_host.matrix3.NSurfs){
    matrix_kernel<true,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
      (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
       &data_gpu->matrix3,
       mat_crossed_gpu,mat_numcrossed_gpu);
  }else if(data_host.matrix3.NVols){
    matrix_kernel<true,false,false><<<nblocks,THREADS_BLOCK,0,stream>>>
      (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
       &data_gpu->matrix3,
       mat_crossed_gpu,mat_numcrossed_gpu);
  }else if(data_host.matrix3.NSurfs){
    matrix_kernel<false,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
      (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
       &data_gpu->matrix3,
       mat_crossed_gpu,mat_numcrossed_gpu);
  }

  // LRMATRIX 3
  if(opts.lrmask3.value()!=""){
    if(data_host.lrmatrix3.NVols && data_host.lrmatrix3.NSurfs){
      matrix_kernel<true,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix3,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix3.NVols){
      matrix_kernel<true,false,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix3,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }else if(data_host.lrmatrix3.NSurfs){
      matrix_kernel<false,true,false><<<nblocks,THREADS_BLOCK,0,stream>>>
        (data_gpu,num_threads,paths_gpu,lengths_gpu,opts.pathdist.value(),opts.omeanpathlength.value(),
         &data_gpu->lrmatrix3,
         lrmat_crossed_gpu,lrmat_numcrossed_gpu);
    }
  }
}
