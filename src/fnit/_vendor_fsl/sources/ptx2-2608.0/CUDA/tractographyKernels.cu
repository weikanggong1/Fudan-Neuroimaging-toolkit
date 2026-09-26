/*  tractographyKernels.cu

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <CUDA/streamlinesDevice.cu>
#include <CUDA/intersectionsDevice.cu>
#include <CUDA/options/options.h>

extern "C" __global__ void setup_randoms_kernel(curandState* state, double seed){
  int id = blockIdx.x*THREADS_BLOCK_RAND+threadIdx.x;
  curand_init(seed,id,0,&state[id]);
}

template <int randfib,bool loopcheck,bool modeuler>
__global__ void get_path_kernel(
  tractographyData* data_gpu,
  const int         maxThread,
  //essential
  curandState*      state,
  const long long   offset,
  //loopcheck
  int*              loopcheckkeys,
  float3*           loopcheckdirs,
  // warps
  WarpTexture       s2d,
  WarpTexture       d2s,
  //OUTPUT
  float*            path,
  int*              lengths)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;
  ///// TEST fibst : yo creo que esta mal en el caso de que randfib=1,2, o 3

  float3 part_init;
  part_init.x=0;
  part_init.y=0;
  part_init.z=0;

  int numseed = (offset+id)/data_gpu->nparticles;
  curandState localState = state[id];
  bool part_has_jumped=false;

  ////// Shared memory ////////
  // in order to have 32 Warps per SM, Max 12 arrays x 64 threads = 3072 (*32=48KB)
  __shared__ float partCx[THREADS_BLOCK]; //coordinates in DTI space
  __shared__ float partCy[THREADS_BLOCK]; //coordinates in DTI space
  __shared__ float partCz[THREADS_BLOCK]; //coordinates in DTI space

  __shared__ float memSH_a[THREADS_BLOCK]; // common space in shared memory
  __shared__ float memSH_b[THREADS_BLOCK]; // common space in shared memory
  __shared__ float memSH_c[THREADS_BLOCK]; // common space in shared memory
  // samples (th,ph,f) are stored here

  __shared__ float memSH_d[THREADS_BLOCK];
  __shared__ float memSH_e[THREADS_BLOCK];
  __shared__ float memSH_f[THREADS_BLOCK];

  __shared__ float partRx[THREADS_BLOCK]; //rotation
  __shared__ float partRy[THREADS_BLOCK]; //rotation
  __shared__ float partRz[THREADS_BLOCK]; //rotation


  // Use path to store my intial coordinates
  // We want to start at the same exact point, even if sampvox is activated
  uint offset_path_fw = id*data_gpu->nsteps*3;
  uint offset_path_bw = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  path[offset_path_fw]= data_gpu->seeds[numseed*3];
  path[offset_path_fw+1]= data_gpu->seeds[numseed*3+1];
  path[offset_path_fw+2]= data_gpu->seeds[numseed*3+2];

  path[offset_path_bw]= data_gpu->seeds[numseed*3];
  path[offset_path_bw+1]= data_gpu->seeds[numseed*3+1];
  path[offset_path_bw+2]= data_gpu->seeds[numseed*3+2];


  if(data_gpu->sampvox>0){
    float r2=data_gpu->sampvox*data_gpu->sampvox;
    bool rej=true;
    float dx,dy,dz;
    while(rej){
      dx=2.0f*data_gpu->sampvox*(curand_uniform(&localState)-0.5f);
      dy=2.0f*data_gpu->sampvox*(curand_uniform(&localState)-0.5f);
      dz=2.0f*data_gpu->sampvox*(curand_uniform(&localState)-0.5f);
      if( dx*dx+dy*dy+dz*dz <= r2 )
        rej=false;
    }

    path[offset_path_fw]+=dx/C_Sdims[0];
    path[offset_path_fw+1]+=dy/C_Sdims[1];
    path[offset_path_fw+2]+=dz/C_Sdims[2];

    path[offset_path_bw]+=dx/C_Sdims[0];
    path[offset_path_bw+1]+=dy/C_Sdims[1];
    path[offset_path_bw+2]+=dz/C_Sdims[2];
  }
  // track in one direction
  lengths[id*2]=streamline<randfib,loopcheck,modeuler>(
    data_gpu,
    localState,
    &loopcheckkeys[int((id*data_gpu->nsteps)/5)],
    &loopcheckdirs[int((id*data_gpu->nsteps)/5)],
    &s2d,
    &d2s,
    &partCx[threadIdx.x],&partCy[threadIdx.x],&partCz[threadIdx.x],
    &partRx[threadIdx.x],&partRy[threadIdx.x],&partRz[threadIdx.x],
    &memSH_a[threadIdx.x],&memSH_b[threadIdx.x],&memSH_c[threadIdx.x],
    &memSH_d[threadIdx.x],&memSH_e[threadIdx.x],&memSH_f[threadIdx.x],
    &path[offset_path_fw],part_init,part_has_jumped);

  // track in the other direction
  lengths[id*2+1]=streamline<randfib,loopcheck,modeuler>(
    data_gpu,
    localState,
    &loopcheckkeys[int((id*data_gpu->nsteps)/5)],
    &loopcheckdirs[int((id*data_gpu->nsteps)/5)],
    &s2d,
    &d2s,
    &partCx[threadIdx.x],&partCy[threadIdx.x],&partCz[threadIdx.x],
    &partRx[threadIdx.x],&partRy[threadIdx.x],&partRz[threadIdx.x],
    &memSH_a[threadIdx.x],&memSH_b[threadIdx.x],&memSH_c[threadIdx.x],
    &memSH_d[threadIdx.x],&memSH_e[threadIdx.x],&memSH_f[threadIdx.x],
    &path[offset_path_bw],part_init,part_has_jumped);

  state[id]=localState; // save state, otherwise random numbers will be repeated (start at the same point)
}

/////////////////////////
/////// AVOID MASK ///////
/////////////////////////
template <bool avoidVol,bool avoidSurf>
__global__ void avoid_masks_kernel( tractographyData*   data_gpu,
                                    const int           maxThread,
                                    //INPUT-OUTPUT
                                    float*              paths,
                                    int*                lengths)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  __shared__ float segmentAx[THREADS_BLOCK];
  __shared__ float segmentAy[THREADS_BLOCK];
  __shared__ float segmentAz[THREADS_BLOCK];
  __shared__ float segmentBx[THREADS_BLOCK];
  __shared__ float segmentBy[THREADS_BLOCK];
  __shared__ float segmentBz[THREADS_BLOCK];

  ///////////////////////
  ////// ONE WAY ////////
  ///////////////////////
  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];
  int2 rejflag;

  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];

  rejflag.x=0;
  int pos=3;
  if(data_gpu->forcefirststep){
    segmentAx[threadIdx.x]=mypath[3];
    segmentAy[threadIdx.x]=mypath[4];
    segmentAz[threadIdx.x]=mypath[5];
    pos=6;
  }
  if(avoidVol){
    if(has_crossed_volume(data_gpu->avoid.volume,        // seed is inside exclusion volume
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      rejflag.x=1;
    }
  }
  if(rejflag.x==0){
    for(;pos<mylength*3;pos=pos+3){
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
      if(avoidVol){
        if(has_crossed_volume(data_gpu->avoid.volume,
                              &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
          rejflag.x=1;
          break;
        }
      }
      if(avoidSurf){
        bool r=has_crossed_surface(data_gpu->avoid.vertices,data_gpu->avoid.faces,data_gpu->avoid.VoxFaces,data_gpu->avoid.VoxFacesIndex,
                                   &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                   &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          rejflag.x=1;
          break;
        }
        segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
        segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
        segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
      }
    }
  }
  if(rejflag.x==1){
    lengths[id*2]=-1;
  }
  ///////////////////////
  ////// OTHER WAY /////
  ///////////////////////
  rejflag.y=0;
  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];

  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];
  pos=3;
  if(data_gpu->forcefirststep){
    segmentAx[threadIdx.x]=mypath[3];
    segmentAy[threadIdx.x]=mypath[4];
    segmentAz[threadIdx.x]=mypath[5];
    pos=6;
  }
  if(avoidVol){
    if(has_crossed_volume(data_gpu->avoid.volume,
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      rejflag.y=1;
    }
  }
  if(rejflag.y==0){
    for(;pos<mylength*3;pos=pos+3){
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
      if(avoidVol){
        if(has_crossed_volume(data_gpu->avoid.volume,
                              &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
          rejflag.y=1;
          break;
        }
      }
      if(avoidSurf){
        bool r = has_crossed_surface(data_gpu->avoid.vertices,data_gpu->avoid.faces,data_gpu->avoid.VoxFaces,data_gpu->avoid.VoxFacesIndex,
                                     &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                     &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          rejflag.y=1;
          break;
        }
        segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
        segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
        segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
      }
    }
  }

  if(rejflag.y>0&&rejflag.x>0){
    lengths[id*2]=-1;
  }
  if(rejflag.y>0){
    lengths[id*2+1]=-1;
  }
}


/////////////////////////
/////// STOP MASK ///////
/////////////////////////
template <bool stopVol,bool stopSurf>
__global__ void stop_masks_kernel(  tractographyData*   data_gpu,
                                    const int           maxThread,
                                    // INPUT-OUTPUT
                                    float*              paths,
                                    int*                lengths)    // num of coordinates
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  __shared__ float segmentAx[THREADS_BLOCK];
  __shared__ float segmentAy[THREADS_BLOCK];
  __shared__ float segmentAz[THREADS_BLOCK];
  __shared__ float segmentBx[THREADS_BLOCK];
  __shared__ float segmentBy[THREADS_BLOCK];
  __shared__ float segmentBz[THREADS_BLOCK];


  ///////////////////////
  ////// ONE WAY ////////
  ///////////////////////
  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];
  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];

  int pos=3;
  if(data_gpu->forcefirststep){
    segmentAx[threadIdx.x]=mypath[3];
    segmentAy[threadIdx.x]=mypath[4];
    segmentAz[threadIdx.x]=mypath[5];
    pos=6;
  }
  bool goLoop=true;
  if(stopVol){
    if(has_crossed_volume(data_gpu->stop.volume,
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2]=-1;
      // here is not +1 because length is not increased with seed coordinates
      else lengths[id*2]=int(pos/3)+1;  // +1 because the current position is counted (num of coordinates)
      goLoop=false;
    }
  }
  if(goLoop){
    for(;pos<mylength*3;pos=pos+3){
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
      if(stopVol){
        if(has_crossed_volume(data_gpu->stop.volume,
                              &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
          if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2]=-1;
          // here is not +1 because length is not increased with seed coordinates
          else lengths[id*2]=int(pos/3)+1;  // +1 because the current position is counted (num of coordinates)
          break;
        }
      }
      if(stopSurf){
        bool r=has_crossed_surface(data_gpu->stop.vertices,data_gpu->stop.faces,data_gpu->stop.VoxFaces,data_gpu->stop.VoxFacesIndex,
                                   &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                   &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2]=-1;
          else lengths[id*2]=int(pos/3)+1;
          break;
        }
        segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
        segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
        segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
      }
    }
  }
  ///////////////////////
  ////// OTHER WAY /////
  ///////////////////////
  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];

  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];
  pos=3;
  if(data_gpu->forcefirststep){
    segmentAx[threadIdx.x]=mypath[3];
    segmentAy[threadIdx.x]=mypath[4];
    segmentAz[threadIdx.x]=mypath[5];
    pos=6;
  }
  goLoop=true;
  if(stopVol){
    if(has_crossed_volume(data_gpu->stop.volume,
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2+1]=-1;
      else lengths[id*2+1]=int(pos/3)+1;
      goLoop=false;
    }
  }
  if(goLoop){
    for(;pos<mylength*3;pos=pos+3){
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
      if(stopVol){
        if(has_crossed_volume(data_gpu->stop.volume,
                              &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
          if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2+1]=-1;
          else lengths[id*2+1]=int(pos/3)+1;
          break;
        }
      }
      if(stopSurf){
        bool r = has_crossed_surface(data_gpu->stop.vertices,data_gpu->stop.faces,data_gpu->stop.VoxFaces,data_gpu->stop.VoxFacesIndex,
                                     &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                     &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          if(((pos/3))*data_gpu->steplength < data_gpu->distthresh) lengths[id*2+1]=-1;
          else lengths[id*2+1]=int(pos/3)+1;
          break;
        }
        segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
        segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
        segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
      }
    }
  }
}



////////////////////////////
/////// WTSTOP MASK ////////
////////////////////////////
// stop when a the streamline is going out of a roi of this mask
// this GPU version solves the problem of overlapped volumes in the CPU version
// for volumes, if the seed is inside the volume, we allow to go out ant then crosss anothetone
// for surfaces, if one is crossed twice, then the stralines stops
// we cannot identify in or out for a surface

// ignoring forcefirststep ... if seed is inside wtstop: is treated

template <bool wtstopVol,bool wtstopSurf>
__global__ void wtstop_masks_kernel(  tractographyData*   data_gpu,
                                      const int           maxThread,
                                      // INPUT-OUTPUT
                                      float*              paths,
                                      int*                lengths)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  extern __shared__ float shared[];
  float* segmentAx = (float*)shared;                 // THREADS_BLOCK;
  float* segmentAy = (float*)&segmentAx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentAz = (float*)&segmentAy[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBx = (float*)&segmentAz[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBy = (float*)&segmentBx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBz = (float*)&segmentBy[THREADS_BLOCK];;         // THREADS_BLOCK;
  char* wtstop_flags = (char*)&segmentBz[THREADS_BLOCK];         // num of wtstop rois (vols+surfs) * THREADS_BLOCK
  // wtstop_flags:
  // 0-> seed is inside the roi, let it go out
  // 1-> still not in roi
  // 2-> in the roi
  // 3-> going out the roi ... STOP

  /////////////////
  //// ONE WAY ////
  /////////////////
  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];
  bool wtstop=false;
  // set flags to 1 (still not in roi)
  for(int j=0;j<(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs);j++)
    wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=1;

  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];
  for(int j=0;j<data_gpu->wtstop.NVols;j++){
    if(has_crossed_volume(&data_gpu->wtstop.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      // seed is inside a roi (only volumes) !!!     should also check Surfaces -> MATRIX 1 ?
      wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=0;
    }
  }

  for(int pos=3;pos<mylength*3;pos=pos+3){
    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];
    if(wtstopVol){
      for(int j=0;j<data_gpu->wtstop.NVols;j++){
        bool r=has_crossed_volume(&data_gpu->wtstop.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r==true && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==1){
          // going into the roi
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=2;
        }else if(r==false && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==2){
          // going outside the roi --- stop
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=3;
          wtstop=true;
        }else if(r==false && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==0){
          // going outside the roi --- but seed was inside the roi
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=1;
        }
      }
    }
    if(wtstopSurf){
      for(int j=0;j<data_gpu->wtstop.NSurfs;j++){
        bool r=has_crossed_surface(data_gpu->wtstop.vertices,data_gpu->wtstop.faces,
                                   data_gpu->wtstop.VoxFaces,
                                   &data_gpu->wtstop.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                   &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                   &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+data_gpu->wtstop.NVols+j]++;
          if(wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+data_gpu->wtstop.NVols+j]==3)
            wtstop=true;
        }
      }
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    if(wtstop){
      if(((pos/3)-1)*data_gpu->steplength < data_gpu->distthresh) lengths[id*2]=-1;
      else lengths[id*2]=int(pos/3);
      // remove last path point? Yes, so that counters don't count 2nd crossings (not +1)
      break;
    }
  }
  ////////////////////
  //// OTHER WAY /////
  ////////////////////
  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];
  wtstop=false;
  // set flags to 1 (still not in roi)
  for(int j=0;j<(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs);j++)
    wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=1;

  segmentAx[threadIdx.x]=mypath[0];
  segmentAy[threadIdx.x]=mypath[1];
  segmentAz[threadIdx.x]=mypath[2];
  for(int j=0;j<data_gpu->wtstop.NVols;j++){
    if(has_crossed_volume(&data_gpu->wtstop.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                          &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x])){
      // seed is inside a roi (only volumes)
      wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=0;
    }
  }

  for(int pos=3;pos<mylength*3;pos=pos+3){
    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];
    if(wtstopVol){
      for(int j=0;j<data_gpu->wtstop.NVols;j++){
        bool r=has_crossed_volume(&data_gpu->wtstop.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r==true && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==1){
          // going into the roi
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=2;
        }else if(r==false && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==2){
          // going outside the roi --- stop
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=3;
          wtstop=true;
        }else if(r==false && wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]==0){
          // going outside the roi --- but seed was inside the roi
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+j]=1;
        }
      }
    }
    if(wtstopSurf){
      for(int j=0;j<data_gpu->wtstop.NSurfs;j++){
        bool r=has_crossed_surface(data_gpu->wtstop.vertices,data_gpu->wtstop.faces,
                                   data_gpu->wtstop.VoxFaces,
                                   &data_gpu->wtstop.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                   &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                   &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
        if(r){
          wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+data_gpu->wtstop.NVols+j]++;
          if(wtstop_flags[threadIdx.x*(data_gpu->wtstop.NVols+data_gpu->wtstop.NSurfs)+data_gpu->wtstop.NVols+j]==3)
            wtstop=true;
        }
      }
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    if(wtstop){
      if(((pos/3)-1)*data_gpu->steplength < data_gpu->distthresh) lengths[id*2+1]=-1;
      else lengths[id*2+1]=int(pos/3);
      // remove last path point? Yes, so that counters don't count 2nd crossings (not +1)
      break;
    }
  }
}

///////////////////////////////
/////// WAYPOINTS MASK ////////
///////////////////////////////
template <bool wayVol,bool waySurf>
__global__ void way_masks_kernel( tractographyData*     data_gpu,
                                  const int              maxThread,
                                  // INNPUT-OUTPUT
                                  float*                paths,
                                  int*                  lengths)
{
  ///// DYNAMIC SHARED MEMORY /////
  extern __shared__ float shared[];
  float* segmentAx = (float*)shared;                 // THREADS_BLOCK;
  float* segmentAy = (float*)&segmentAx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentAz = (float*)&segmentAy[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBx = (float*)&segmentAz[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBy = (float*)&segmentBx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBz = (float*)&segmentBy[THREADS_BLOCK];;         // THREADS_BLOCK;
  char* way_flags = (char*)&segmentBz[THREADS_BLOCK];    //num of way_points rois: (NwayVols + NwaySurfs) * THREADS_BLOCK

  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];

  int numpassed=0;
  int totalRois=data_gpu->waypoint.NVols+data_gpu->waypoint.NSurfs;

  for(int i=0;i<totalRois;i++){
    way_flags[threadIdx.x*totalRois+i]=0;
  }

  // rejflag = 0 (accept), 1 (reject) or 2 (wait for second direction)
  int2 rejflag;
  bool order=true;

  if(waySurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  /////////////////
  //// ONE WAY ////
  /////////////////
  rejflag.x=0;
  for(int pos=3;pos<mylength*3;pos=pos+3){
    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];
    if(wayVol){
      for(int j=0;j<data_gpu->waypoint.NVols;j++){
        if(!way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[j]]){
          if(has_crossed_volume(&data_gpu->waypoint.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
            way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[j]]=1;
            numpassed++;
          }
        }
      }
    }
    if(waySurf){
      for(int j=0;j<data_gpu->waypoint.NSurfs;j++){
        if(!way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[data_gpu->waypoint.NVols+j]]){
          bool r=    has_crossed_surface(data_gpu->waypoint.vertices,data_gpu->waypoint.faces,
                                         data_gpu->waypoint.VoxFaces,
                                         &data_gpu->waypoint.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                         &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                         &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
          if(r){
            way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[data_gpu->waypoint.NVols+j]]=1;
            numpassed++;
          }
        }
      }
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    // check if order is respected (each iteration)
    if(data_gpu->waycond && data_gpu->wayorder){
      for(int i=0;i<totalRois;i++){
        if(way_flags[threadIdx.x*totalRois+i]){
          for(int k=0;k<i;k++){
            if(!way_flags[threadIdx.x*totalRois+k]){
              order=false;
              break;
            }
          }
        }
        if(!order) break;
      }
    }
    if(!order) break;
  }
  if(numpassed==0) rejflag.x=1;    // not any roi: reject
  else if(numpassed<(totalRois)){    // only some rois
    if(data_gpu->waycond){    // AND condition
      if(data_gpu->oneway) // oneway...so each part individually
        rejflag.x=1; // not all rois: reject
      else{        // second part of the path...maybe
        if(data_gpu->wayorder){
          if(order) rejflag.x=2;     // if order...wait to second part
          else rejflag.x=1;       // nor order: reject
        }else{
          rejflag.x=2;        // order does not matter: wait
        }
      }
    }else{
      rejflag.x=0;  // OR condition: accept (order does not matter)
    }
  }else{ // all rois crossed
    if(data_gpu->waycond && data_gpu->wayorder){
      if(order) rejflag.x=0;        // accept
      else rejflag.x=1;        // not correct order: rejecy
    }else{
      rejflag.x=0;    // all conditions ok and order does not matter
    }
  }
  if(rejflag.x==1){
    lengths[id*2]=-1;
  }

  ///////////////////
  //// OTHER WAY ////
  ///////////////////
  rejflag.y=0;
  if(data_gpu->oneway){
    for(int i=0;i<totalRois;i++){
      way_flags[threadIdx.x*totalRois+i]=0;
    }
    numpassed=0;
    order=true;
  }
  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];

  if(waySurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  for(int pos=3;pos<mylength*3;pos=pos+3){
    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];

    if(wayVol){
      for(int j=0;j<data_gpu->waypoint.NVols;j++){
        if(!way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[j]]){
          if(has_crossed_volume(&data_gpu->waypoint.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
            way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[j]]=1;
            numpassed++;
          }
        }
      }
    }
    if(waySurf){
      for(int j=0;j<data_gpu->waypoint.NSurfs;j++){
        if(!way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[data_gpu->waypoint.NVols+j]]){
          bool r=    has_crossed_surface(data_gpu->waypoint.vertices,data_gpu->waypoint.faces,
                                         data_gpu->waypoint.VoxFaces,
                                         &data_gpu->waypoint.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                         &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                         &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
          if(r){
            way_flags[threadIdx.x*totalRois+data_gpu->waypoint.IndexRoi[data_gpu->waypoint.NVols+j]]=1;
            numpassed++;
          }
        }
      }
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    // check if order is respected
    if(data_gpu->waycond && data_gpu->wayorder){
      for(int i=0;i<(totalRois);i++){
        if(way_flags[threadIdx.x*totalRois+i]){
          for(int k=0;k<i;k++){
            if(!way_flags[threadIdx.x*totalRois+k]){
              order=false;
              break;
            }
          }
        }
        if(!order) break;
      }
    }
    if(!order) break;
  }
  if(numpassed==0) rejflag.y=1;    // not any roi
  else if(numpassed<(totalRois)){    // only some rois
    if(data_gpu->waycond){
      if(data_gpu->oneway)
        rejflag.y=1;
      else{
        if(data_gpu->wayorder){
          if(order) rejflag.y=2;
          else rejflag.y=1;
        }else{
          rejflag.y=2;
        }
      }
    }else{
      rejflag.y=0;
    }
  }else{
    if(data_gpu->waycond && data_gpu->wayorder){
      if(order) rejflag.y=0;
      else rejflag.y=1;
    }else{
      rejflag.y=0;    // ALL CONDITIONS
    }
  }
  if(rejflag.y>0&&rejflag.x>0){
    lengths[id*2]=-1;
  }
  if(rejflag.y>0){
    //printf("NO WAYYY y\n");
    lengths[id*2+1]=-1;
  }
}

/////////////////////////////
/////// NETWORK MASK ////////
/////////////////////////////
template <bool netVol,bool netSurf,int savelength,bool flags_in_shared>
// savelength 0: no --pd, nor --ompl | 1: --pd | 2: --ompl (ConNet pathlengths, ConNetb binary hits, and later calculates mean)
__global__ void net_masks_kernel( tractographyData*   data_gpu,
                                  const int           maxThread,
                                  const long long     offset,
                                  // INNPUT-OUTPUT
                                  float*              paths,
                                  int*                lengths,
                                  float*              ConNet,
                                  float*              ConNetb,
                                  // To use in case too many Net ROIs
                                  float*              net_flags_Global,
                                  float*              net_values_Global)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  int totalRois=data_gpu->network.NVols+data_gpu->network.NSurfs;
  ///// DYNAMIC SHARED MEMORY /////
  extern __shared__ float shared[];
  float* segmentAx = (float*)shared;              // THREADS_BLOCK;
  float* segmentAy = (float*)&segmentAx[THREADS_BLOCK];; // THREADS_BLOCK;
  float* segmentAz = (float*)&segmentAy[THREADS_BLOCK];; // THREADS_BLOCK;
  float* segmentBx = (float*)&segmentAz[THREADS_BLOCK];; // THREADS_BLOCK;
  float* segmentBy = (float*)&segmentBx[THREADS_BLOCK];; // THREADS_BLOCK;
  float* segmentBz = (float*)&segmentBy[THREADS_BLOCK];; // THREADS_BLOCK;

  float* net_flags;
  float* net_values;
  if(flags_in_shared){
    // Using Shared Memory
    net_values = (float*)&segmentBz[THREADS_BLOCK];
    // num of net_points rois: (NnetVols + NnetSurfs) * THREADS_BLOCK
    net_flags = (float*)&net_values[THREADS_BLOCK*totalRois];
    // num of net_points rois: (NnetVols + NnetSurfs) * THREADS_BLOCK

    //eah thread of the block:
    net_values = (float*)&net_values[threadIdx.x*totalRois];
    net_flags = (float*)&net_flags[threadIdx.x*totalRois];
  }else{
    net_flags = &net_flags_Global[id*totalRois];
    net_values = &net_values_Global[id*totalRois];
  }

  int numseed = (offset+id)/data_gpu->nparticles;
  int ROI = data_gpu->seeds_ROI[numseed];

  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];
  int numpassed=1; // count my own ROI

  for(int i=0;i<totalRois;i++){
    net_values[i]=0.0f;
    net_flags[i]=0.0f;
  }
  net_flags[ROI]=1.0f; // my own ROI

  if(netSurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  /////////////////
  //// ONE WAY ////
  /////////////////
  float pathlength=data_gpu->steplength;
  float pathlength2=data_gpu->steplength; //compiler (6.5) fails if I use pathlength in the second direction

  for(int pos=3;pos<mylength*3;pos=pos+3){
    bool checkMASKs=0; //  Avoid to check all the ROIs in case not any cross

    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];

    if(data_gpu->networkREF.NVols){
      if(has_crossed_volume(data_gpu->networkREF.volume,
                            &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
        checkMASKs=1;
      }
    }
    if(data_gpu->networkREF.NSurfs){
      bool r=has_crossed_surface(data_gpu->networkREF.vertices,data_gpu->networkREF.faces,data_gpu->networkREF.VoxFaces,data_gpu->networkREF.VoxFacesIndex,
                                 &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                 &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
      if(r){
        checkMASKs=1;
      }
    }

    if(checkMASKs){
      if(netVol){
        for(int j=0;j<data_gpu->network.NVols;j++){
          if(!net_flags[data_gpu->network.IndexRoi[j]]){
            if(has_crossed_volume(&data_gpu->network.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
              net_flags[data_gpu->network.IndexRoi[j]]=pathlength;
              numpassed++;
            }
          }
        }
      }
      if(netSurf){
        for(int j=0;j<data_gpu->network.NSurfs;j++){
          if(!net_flags[data_gpu->network.IndexRoi[data_gpu->network.NVols+j]]){
            bool r=has_crossed_surface(data_gpu->network.vertices,data_gpu->network.faces,
                                       data_gpu->network.VoxFaces,
                                       &data_gpu->network.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                       &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                       &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
            if(r){
              net_flags[data_gpu->network.IndexRoi[data_gpu->network.NVols+j]]=pathlength;
              numpassed++;
            }
          }
        }
      }
    }
    if(netSurf){
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    pathlength+=data_gpu->steplength;
  }
  if(numpassed<=1){
    lengths[id*2]=-1; // not any roi (only mine): reject
  }else{
    // at least one crossed
    for(int i=0;i<totalRois;i++){
      if(i!=ROI){
        if(net_flags[i]){
          net_values[i]=net_flags[i];
        }
      }
    }
  }

  ///////////////////
  //// OTHER WAY ////
  ///////////////////
  for(int i=0;i<totalRois;i++){
    net_flags[i]=0;
  }
  net_flags[ROI]=1; // my own ROI
  numpassed=1; // count my own ROI

  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];

  if(netSurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  for(int pos=3;pos<mylength*3;pos=pos+3){
    bool checkMASKs=0; //  Avoid to check all the ROIs in case not any cross

    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];


    if(data_gpu->networkREF.NVols){
      if(has_crossed_volume(data_gpu->networkREF.volume,
                            &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
        checkMASKs=1;
      }
    }
    if(data_gpu->networkREF.NSurfs){
      bool r=has_crossed_surface(data_gpu->networkREF.vertices,data_gpu->networkREF.faces,
                                 data_gpu->networkREF.VoxFaces,data_gpu->networkREF.VoxFacesIndex,
                                 &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                 &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
      if(r){
        checkMASKs=1;
      }
    }

    if(checkMASKs){
      if(netVol){
        for(int j=0;j<data_gpu->network.NVols;j++){
          if(!net_flags[data_gpu->network.IndexRoi[j]]){
            if(has_crossed_volume(&data_gpu->network.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
              net_flags[data_gpu->network.IndexRoi[j]]=pathlength2;
              numpassed++;
            }
          }
        }
      }
      if(netSurf){
        for(int j=0;j<data_gpu->network.NSurfs;j++){
          if(!net_flags[data_gpu->network.IndexRoi[data_gpu->network.NVols+j]]){
            bool r=has_crossed_surface(data_gpu->network.vertices,data_gpu->network.faces,
                                       data_gpu->network.VoxFaces,
                                       &data_gpu->network.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                       &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                       &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
            if(r){
              net_flags[data_gpu->network.IndexRoi[data_gpu->network.NVols+j]]=pathlength2;
              numpassed++;
            }
          }
        }
      }
    }
    if(netSurf){
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    pathlength2+=data_gpu->steplength;
  }
  if(numpassed<=1){
    lengths[id*2+1]=-1; // not any roi (only mine)
  }else{
    // at least one crossed
    for(int i=0;i<totalRois;i++){
      if(i!=ROI){
        if(net_flags[i]){
          if(net_values[i]==0){
            net_values[i]=net_flags[i];
          }else{
            // if already hit by the other direction, we store the mean of the length of both directions
            net_values[i]=(net_values[i]+net_flags[i])/2;
          }
        }
      }
    }
  }
  for(int i=0;i<totalRois;i++){
    if(i!=ROI){
      if(net_values[i]){
        if(savelength==0)
          atomicAdd(&ConNet[ROI*totalRois+i],1);
        if(savelength==1)
          atomicAdd(&ConNet[ROI*totalRois+i],net_values[i]);
        if(savelength==2){
          atomicAdd(&ConNet[ROI*totalRois+i],net_values[i]);
          atomicAdd(&ConNetb[ROI*totalRois+i],1);
        }
      }
    }
  }
}

/////////////////////////////
/////// TARGETS MASK ////////
/////////////////////////////
template <bool targVol,bool targSurf,int savelength, bool flags_in_shared>
// savelength 0: no --pd or --ompl | 1: --pd | 2: --ompl
__global__ void targets_masks_kernel( tractographyData*     data_gpu,
                                      const int             maxThread,
                                      const long long       offset,
                                      // INNPUT-OUTPUT
                                      float*                paths,
                                      int*                  lengths,
                                      float*                s2targets_gpu,        // (Nseeds x NTargets)
                                      float*                s2targetsb_gpu,
                                      // To use in case too many Net ROIs
                                      float*                targ_flags_Global)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  int totalTargets=data_gpu->targets.NVols+data_gpu->targets.NSurfs;
  ///// DYNAMIC SHARED MEMORY /////
  extern __shared__ float shared[];
  float* segmentAx = (float*)shared;                 // THREADS_BLOCK;
  float* segmentAy = (float*)&segmentAx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentAz = (float*)&segmentAy[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBx = (float*)&segmentAz[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBy = (float*)&segmentBx[THREADS_BLOCK];;         // THREADS_BLOCK;
  float* segmentBz = (float*)&segmentBy[THREADS_BLOCK];;         // THREADS_BLOCK;

  float* targ_flags;
  if(flags_in_shared){
    // Using Shared Memory
    targ_flags = (float*)&segmentBz[THREADS_BLOCK];
    // num of target rois: (NtarVols + NtarSurfs) * THREADS_BLOCK
    // eah thread of the block:
    targ_flags = (float*)&targ_flags[threadIdx.x*totalTargets];
  }else{
    targ_flags = &targ_flags_Global[id*totalTargets];
  }

  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  int mylength=lengths[id*2];

  for(int i=0;i<totalTargets;i++){
    targ_flags[i]=0.0f;
  }

  if(targSurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  /////////////////
  //// ONE WAY ////
  /////////////////
  float pathlength=data_gpu->steplength;

  for(int pos=3;pos<mylength*3;pos=pos+3){
    bool checkMASKs=0; //  Avoid to check all the ROIs in case not any cross

    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];

    if(data_gpu->targetsREF.NVols){
      if(has_crossed_volume(data_gpu->targetsREF.volume,
                            &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
        checkMASKs=1;
      }
    }
    if(data_gpu->targetsREF.NSurfs){
      bool r=has_crossed_surface(data_gpu->targetsREF.vertices,data_gpu->targetsREF.faces,data_gpu->targetsREF.VoxFaces,data_gpu->targetsREF.VoxFacesIndex,
                                 &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                 &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
      if(r){
        checkMASKs=1;
      }
    }

    if(checkMASKs){
      if(targVol){
        for(int j=0;j<data_gpu->targets.NVols;j++){
          if(!targ_flags[data_gpu->targets.IndexRoi[j]]){
            if(has_crossed_volume(&data_gpu->targets.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
              targ_flags[data_gpu->targets.IndexRoi[j]]=pathlength;
            }
          }
        }
      }
      if(targSurf){
        for(int j=0;j<data_gpu->targets.NSurfs;j++){
          if(!targ_flags[data_gpu->targets.IndexRoi[data_gpu->targets.NVols+j]]){
            bool r= has_crossed_surface(data_gpu->targets.vertices,data_gpu->targets.faces,
                                        data_gpu->targets.VoxFaces,
                                        &data_gpu->targets.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                        &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                        &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
            if(r){
              targ_flags[data_gpu->targets.IndexRoi[data_gpu->targets.NVols+j]]=pathlength;
            }
          }
        }
      }
    }
    if(targSurf){
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    pathlength+=data_gpu->steplength;
  }

  ///////////////////
  //// OTHER WAY ////
  ///////////////////
  pathlength=data_gpu->steplength;
  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];

  if(targSurf){
    segmentAx[threadIdx.x]=mypath[0];
    segmentAy[threadIdx.x]=mypath[1];
    segmentAz[threadIdx.x]=mypath[2];
  }
  for(int pos=3;pos<mylength*3;pos=pos+3){
    bool checkMASKs=0; //  Avoid to check all the ROIs in case not any cross

    segmentBx[threadIdx.x]=mypath[pos];
    segmentBy[threadIdx.x]=mypath[pos+1];
    segmentBz[threadIdx.x]=mypath[pos+2];

    if(data_gpu->targetsREF.NVols){
      if(has_crossed_volume(data_gpu->targetsREF.volume,
                            &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
        checkMASKs=1;
      }
    }
    if(data_gpu->targetsREF.NSurfs){
      bool r=has_crossed_surface(data_gpu->targetsREF.vertices,data_gpu->targetsREF.faces,
                                 data_gpu->targetsREF.VoxFaces,data_gpu->targetsREF.VoxFacesIndex,
                                 &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                 &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
      if(r){
        checkMASKs=1;
      }
    }

    if(checkMASKs){
      if(targVol){
        for(int j=0;j<data_gpu->targets.NVols;j++){
          if(!targ_flags[data_gpu->targets.IndexRoi[j]]){
            if(has_crossed_volume(&data_gpu->targets.volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                  &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x])){
              targ_flags[data_gpu->targets.IndexRoi[j]]=pathlength;
            }
          }
        }
      }
      if(targSurf){
        for(int j=0;j<data_gpu->targets.NSurfs;j++){
          if(!targ_flags[data_gpu->targets.IndexRoi[data_gpu->targets.NVols+j]]){
            bool r=has_crossed_surface(data_gpu->targets.vertices,data_gpu->targets.faces,
                                       data_gpu->targets.VoxFaces,
                                       &data_gpu->targets.VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                       &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                       &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
            if(r){
              targ_flags[data_gpu->targets.IndexRoi[data_gpu->targets.NVols+j]]=pathlength;
            }
          }
        }
      }
    }
    if(targSurf){
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    pathlength+=data_gpu->steplength;
  }

  int numseed = (offset+id)/data_gpu->nparticles;
  int totalSeeds = data_gpu->nseeds;
  for(int i=0;i<totalTargets;i++){
    if(targ_flags[i]){
      if(savelength==0){
        atomicAdd(&s2targets_gpu[i*totalSeeds+numseed],1);
      }if(savelength==1){
        atomicAdd(&s2targets_gpu[i*totalSeeds+numseed],targ_flags[i]);
      }if(savelength==2){
        atomicAdd(&s2targets_gpu[i*totalSeeds+numseed],targ_flags[i]);
        atomicAdd(&s2targetsb_gpu[i*totalSeeds+numseed],1);
      }
    }
  }
}

///////////////////////////////
/////// MATRIX  MASKs /////////
///////////////////////////////
template <bool HVols,bool HSurfs, bool M2> // M2 is for Matrix2: it can be defined in a different space
__global__ void matrix_kernel(  tractographyData*     data_gpu,
                                const int             maxThread,
                                float*                paths,
                                int*                  lengths,
                                bool                  pathdist,
                                bool                  omeanpathlength,
                                MaskData*             matrixData, // info vols & surfs
                                // OUTPUT
                                float3*               crossed,
                                int*                   numcrossed)
{
  unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  __shared__ float segmentAx[THREADS_BLOCK];
  __shared__ float segmentAy[THREADS_BLOCK];
  __shared__ float segmentAz[THREADS_BLOCK];
  __shared__ float segmentBx[THREADS_BLOCK];
  __shared__ float segmentBy[THREADS_BLOCK];
  __shared__ float segmentBz[THREADS_BLOCK];

  //int max_per_jump=1;  // I changed this, see mem file
  int max_per_jump=3;  // Change THIs, should pass from min routine
  float pathlength=1.0f;
  if(pathdist||omeanpathlength) pathlength=data_gpu->steplength; // it starts with the second coordinate of the path

  // if path shorter than threshold, then ignore it
  int mylength=lengths[id*2];
  float length=0;
  length=lengths[id*2+1];
  if(length<=0) length=0; // it can be -1: rejected
  if(length && mylength>0) length=length-1;    // do not count seed point twice
  if(mylength>0) length+=mylength;
  length=length*data_gpu->steplength;
  if(length<matrixData->distthresh){
    numcrossed[id]=0;
    return;
  }

  /////////////////
  //// ONE WAY ////
  /////////////////
  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath=&paths[offset_path];
  if(HSurfs){
    if(M2){
      vox_to_vox_S2M2(mypath,&segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x]);
    }else{
      segmentAx[threadIdx.x]=mypath[0];
      segmentAy[threadIdx.x]=mypath[1];
      segmentAz[threadIdx.x]=mypath[2];
    }
    max_per_jump=3;
  }

  int pos=3;
  int mynumcrossed=0;
  for(;pos<mylength*3;pos=pos+3){
    if(M2){
      vox_to_vox_S2M2(&mypath[pos],&segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
    }else{
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
    }
    if(HVols){
      for(int j=0;j<matrixData->NVols;j++){
        int loc=0;
        if(has_crossed_volume_loc<M2>(&matrixData->volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                      &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x],loc)){
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].x=loc;
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].y=-1; // id triangle...not here (only surfaces)
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].z=pathlength;
          mynumcrossed++;
        }
      }
    }
    if(HSurfs){
      for(int j=0;j<matrixData->NSurfs;j++){
        has_crossed_surface_loc<M2>(matrixData->vertices,matrixData->locs,
                                    matrixData->faces,matrixData->VoxFaces,
                                    &matrixData->VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                    &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                    &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x],
                                    &crossed[id*data_gpu->nsteps*3],mynumcrossed,pathlength);
      }

      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    if(pathdist||omeanpathlength) pathlength+=data_gpu->steplength;
  }

  ///////////////////
  //// OTHER WAY ////
  ///////////////////
  if(pathdist||omeanpathlength) pathlength=-data_gpu->steplength; // it starts with the second coordinate of the path
  // reverse, m_tracksign !! . If different directions when crossing 2 nodes, then the path distance is longer.

  offset_path = id*data_gpu->nsteps*3+((data_gpu->nsteps/2)*3);
  mypath=&paths[offset_path];
  mylength=lengths[id*2+1];
  if(HSurfs){
    if(M2){
      vox_to_vox_S2M2(mypath,&segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x]);
    }else{
      segmentAx[threadIdx.x]=mypath[0];
      segmentAy[threadIdx.x]=mypath[1];
      segmentAz[threadIdx.x]=mypath[2];
    }
  }
  pos=3;

  for(;pos<mylength*3;pos=pos+3){
    if(M2){
      vox_to_vox_S2M2(&mypath[pos],&segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x]);
    }else{
      segmentBx[threadIdx.x]=mypath[pos];
      segmentBy[threadIdx.x]=mypath[pos+1];
      segmentBz[threadIdx.x]=mypath[pos+2];
    }
    if(HVols){
      for(int j=0;j<matrixData->NVols;j++){
        int loc=0;
        if (has_crossed_volume_loc<M2>(&matrixData->volume[j*C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]],
                                       &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x],loc)){
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].x=loc;
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].y=-1; // id triangle...not here
          crossed[id*data_gpu->nsteps*max_per_jump+mynumcrossed].z=pathlength;
          mynumcrossed++;
        }
      }
    }
    if(HSurfs){
      for(int j=0;j<matrixData->NSurfs;j++){
        has_crossed_surface_loc<M2>(matrixData->vertices,matrixData->locs,
                                    matrixData->faces,matrixData->VoxFaces,
                                    &matrixData->VoxFacesIndex[j*(C_Ssizes[0]*C_Ssizes[1]*C_Ssizes[2]+1)],
                                    &segmentAx[threadIdx.x],&segmentAy[threadIdx.x],&segmentAz[threadIdx.x],
                                    &segmentBx[threadIdx.x],&segmentBy[threadIdx.x],&segmentBz[threadIdx.x],
                                    &crossed[id*data_gpu->nsteps*3],mynumcrossed,pathlength);
      }
      segmentAx[threadIdx.x]=segmentBx[threadIdx.x];
      segmentAy[threadIdx.x]=segmentBy[threadIdx.x];
      segmentAz[threadIdx.x]=segmentBz[threadIdx.x];
    }
    if(pathdist||omeanpathlength) pathlength-=data_gpu->steplength;
  }

  numcrossed[id]=mynumcrossed;
}

/////////////////////////////////////
///////// UPDATE PATHS VOLUME ///////
/////////////////////////////////////
template <bool pathdist, bool omeanpathlength, bool opathdir>
__global__ void update_path_kernel( tractographyData*     data_gpu,
                                    const int             maxThread,
                                    float*                path,
                                    int*                  lengths,
                                    int*                  beenhere,
                                    const int             upper_limit,
                                    // OUTPUT
                                    float*                m_prob,
                                    float*                m_prob2,    // for omeanpathlength
                                    float*                m_localdir)    // for opathdir
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if(id>=maxThread) return;

  uint offset_path = id*data_gpu->nsteps*3;
  float* mypath = &path[offset_path];
  int mylength = lengths[id*2];
  int* m_beenhere = &beenhere[id*(data_gpu->nsteps)];
  int coordinatex,coordinatey,coordinatez;
  int Tcoordinate;
  float coordinatex2,coordinatey2,coordinatez2;
  float pathlength=0.0f;
  bool update;
  int length_beenhere=0;
  float steplength = data_gpu->steplength;
  int factor_posY=C_Ssizes[0];
  int factor_posZ=C_Ssizes[0]*C_Ssizes[1];
  //float3 pre_path;
  // one way

  for(int i=0;i<mylength;i++){
    // if(i>0 && (m_path[i]-m_path[0]).MaximumAbsoluteValue()==0){
    // pathlength=0;
    // What about this for Matrix 3 !!!

    coordinatex2 = mypath[(i)*3];    // coordinates for next iteration
    coordinatey2 = mypath[(i)*3+1];    // prefetching ???  ... take a lookkk
    coordinatez2 = mypath[(i)*3+2];

    update=true;
    coordinatex = (int)(rintf(coordinatex2));
    coordinatey = (int)(rintf(coordinatey2));
    coordinatez = (int)(rintf(coordinatez2));

    if(coordinatex<0||coordinatex>(C_Ssizes[0]-1)
       ||coordinatey<0||coordinatey>(C_Ssizes[1]-1)
       ||coordinatez<0||coordinatez>(C_Ssizes[2]-1)) continue;

    Tcoordinate = coordinatex+coordinatey*factor_posY+coordinatez*factor_posZ;

    for(int j=0;j<length_beenhere;j++){
      if(Tcoordinate==m_beenhere[j]){
        update=false;
        break;
      }
    }
    if(Tcoordinate>0 && Tcoordinate<upper_limit&& update && update){
      if(opathdir){
        float3 v;
        int Tcoordinate2=(coordinatex+coordinatey*factor_posY+coordinatez*factor_posZ)*6;
        if(i==0){
          //if(mylength>1){
          v.x=mypath[3]-coordinatex2;
          v.y=mypath[4]-coordinatey2;
          v.z=mypath[5]-coordinatez2;
          //}else{
          //    printf("AddA 0 in %i %i %i\n",coordinatex,coordinatey,coordinatez);
          //    v.x=0;v.y=0;v.z=0;
          //}
        }else{
          v.x=coordinatex2-mypath[(i-1)*3];
          v.y=coordinatey2-mypath[(i-1)*3+1];
          v.z=coordinatez2-mypath[(i-1)*3+2];
        }
        float ss=v.x*v.x+v.y*v.y+v.z*v.z;
        if(ss>0){
          ss=sqrtf(ss);
          v.x=v.x/ss; v.y=v.y/ss; v.z=v.z/ss;
        }
        // Add direction (needs to account for the current direction and flip if necessary)
        // lower diagonal rows (because of the way SymmetricMatrix works)
        atomicAdd(&m_localdir[Tcoordinate2],v.x*v.x);
        atomicAdd(&m_localdir[Tcoordinate2+1],v.x*v.y);
        atomicAdd(&m_localdir[Tcoordinate2+2],v.y*v.y);
        atomicAdd(&m_localdir[Tcoordinate2+3],v.x*v.z);
        atomicAdd(&m_localdir[Tcoordinate2+4],v.y*v.z);
        atomicAdd(&m_localdir[Tcoordinate2+5],v.z*v.z);
      }
      if(!omeanpathlength){
        if(!pathdist){
          atomicAdd(&m_prob[Tcoordinate],1);
        }else{
          atomicAdd(&m_prob[Tcoordinate],pathlength);
        }
      }else{
        if(!pathdist){
          atomicAdd(&m_prob[Tcoordinate],1);
          atomicAdd(&m_prob2[Tcoordinate],pathlength);
        }else{
          atomicAdd(&m_prob[Tcoordinate],pathlength);
          atomicAdd(&m_prob2[Tcoordinate],1);
        }
      }
      m_beenhere[length_beenhere]=Tcoordinate;
      length_beenhere++;

      ///BENHERE VERY SLOW
    }
    pathlength+=steplength;
  }

  // other way
  offset_path = id*data_gpu->nsteps*3+(data_gpu->nsteps/2)*3;
  mypath = &path[offset_path];
  mylength = lengths[id*2+1];
  pathlength=0.0f;

  for(int i=0;i<mylength;i++){    // Start at second position, seed has already been updated  ???
    //if(i>0 && (m_path[i]-m_path[0]).MaximumAbsoluteValue()==0){
    //pathlength=0;
    coordinatex2 = mypath[(i)*3];    // coordinates for next iteration
    coordinatey2 = mypath[(i)*3+1];    // prefetching ???  ... take a lookkk
    coordinatez2 = mypath[(i)*3+2];

    update=true;
    coordinatex = (int)(rintf(coordinatex2));
    coordinatey = (int)(rintf(coordinatey2));
    coordinatez = (int)(rintf(coordinatez2));

    if(coordinatex<0||coordinatex>(C_Ssizes[0]-1)
       ||coordinatey<0||coordinatey>(C_Ssizes[1]-1)
       ||coordinatez<0||coordinatez>(C_Ssizes[2]-1)) continue;

    Tcoordinate = coordinatex+coordinatey*factor_posY+coordinatez*factor_posZ;
    for(int j=0;j<length_beenhere;j++){
      if(Tcoordinate==m_beenhere[j]){
        update=false;
        break;
      }
    }
    if(Tcoordinate>0 && Tcoordinate<upper_limit && update){
      if(opathdir){
        float3 v;
        int Tcoordinate2=(coordinatex+coordinatey*factor_posY+coordinatez*factor_posZ)*6;
        if(i==0){
          //if(mylength>1){
          v.x=mypath[3]-coordinatex2;
          v.y=mypath[4]-coordinatey2;
          v.z=mypath[5]-coordinatez2;
          //}else{
          //    printf("AddB 0 in %i %i %i\n",coordinatex,coordinatey,coordinatez);
          //    v.x=0;v.y=0;v.z=0;
          //}
        }else{
          v.x=coordinatex2-mypath[(i-1)*3];
          v.y=coordinatey2-mypath[(i-1)*3+1];
          v.z=coordinatez2-mypath[(i-1)*3+2];
        }
        float ss=v.x*v.x+v.y*v.y+v.z*v.z;
        if(ss>0){
          ss=sqrtf(ss);
          v.x=v.x/ss; v.y=v.y/ss; v.z=v.z/ss;
        }
        // Add direction (needs to account for the current direction and flip if necessary)
        // lower diagonal rows (because of the way SymmetricMatrix works)
        atomicAdd(&m_localdir[Tcoordinate2],v.x*v.x);
        atomicAdd(&m_localdir[Tcoordinate2+1],v.x*v.y);
        atomicAdd(&m_localdir[Tcoordinate2+2],v.y*v.y);
        atomicAdd(&m_localdir[Tcoordinate2+3],v.x*v.z);
        atomicAdd(&m_localdir[Tcoordinate2+4],v.y*v.z);
        atomicAdd(&m_localdir[Tcoordinate2+5],v.z*v.z);
      }
      if(!omeanpathlength){
        if(!pathdist){
          atomicAdd(&m_prob[Tcoordinate],1);
        }else{
          atomicAdd(&m_prob[Tcoordinate],pathlength);
        }
      }else{
        if(!pathdist){
          atomicAdd(&m_prob[Tcoordinate],1);
          atomicAdd(&m_prob2[Tcoordinate],pathlength);
        }else{
          atomicAdd(&m_prob[Tcoordinate],pathlength);
          atomicAdd(&m_prob2[Tcoordinate],1);
        }
      }
      m_beenhere[length_beenhere]=Tcoordinate;
      length_beenhere++;
      ///BENHERE VERY SLOW
    }
    pathlength+=steplength;
  }
}
