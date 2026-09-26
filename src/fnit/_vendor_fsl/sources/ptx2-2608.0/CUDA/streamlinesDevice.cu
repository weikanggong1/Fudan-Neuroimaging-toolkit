/*  streamlines_kernel.cu

    Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#define DIVRND 1.00000001f
#define LCRAT 5.0f    //box for loopcheck is five times smaller than brain mask

#include <CUDA/transformationsDevice.cu>
#include <CUDA/tractographyData.h>
#include <curand_kernel.h>

__constant__ float C_steplength[1];
__constant__ float C_distthresh[1];
__constant__ float C_curv_thr[1];
//__constant__ float C_fibthresh[1];

__device__ inline bool inMask(    tractographyData*    data_gpu,
                                  float*             coordx,
                                  float*             coordy,
                                  float*             coordz)
{
  if(rintf(coordx[0])<0 || rintf(coordx[0])>=C_Dsizes[0]) return false;
  if(rintf(coordy[0])<0 || rintf(coordy[0])>=C_Dsizes[1]) return false;
  if(rintf(coordz[0])<0 || rintf(coordz[0])>=C_Dsizes[2]) return false;

  return(data_gpu->mask[(int)rintf(coordz[0])*C_Dsizes[0]*C_Dsizes[1]+
                        (int)rintf(coordy[0])*C_Dsizes[0]+(int)rintf(coordx[0])]!=0);
}

__device__ inline void jump(    tractographyData*    data_gpu,
                                float*             new_rx,
                                float*            new_ry,
                                float*            new_rz,
                                float*            partCx,
                                float*            partCy,
                                float*            partCz,
                                float*             partRx,
                                float*            partRy,
                                float*            partRz,
                                curandState&         localState)
{
  //float rx_new=cos(sample_ph[0])*sin(sample_th[0]);
  //float ry_new=sin(sample_ph[0])*sin(sample_th[0]);
  //float rz_new=cos(sample_th[0]);
  float sign=1.0f;

  //if(!m_simdiff){  always false
  sign=((new_rx[0]*partRx[0] + new_ry[0]*partRy[0] + new_rz[0]*partRz[0])>0) ? 1.0f:-1.0f;

  partCx[0] += sign*(*C_steplength)/C_Ddims[0]*new_rx[0];
  partCy[0] += sign*(*C_steplength)/C_Ddims[1]*new_ry[0];
  partCz[0] += sign*(*C_steplength)/C_Ddims[2]*new_rz[0];

  partRx[0]=sign*new_rx[0];
  partRy[0]=sign*new_ry[0];
  partRz[0]=sign*new_rz[0];
}

__device__ inline void testjump(tractographyData*    data_gpu,
                                float*             new_rx,
                                float*            new_ry,
                                float*            new_rz,
                                float&            partCx,
                                float&            partCy,
                                float&            partCz,
                                float*             partRx,
                                float*            partRy,
                                float*            partRz,
                                curandState&         localState)
{
  float sign=1.0f;
  sign=((new_rx[0]*partRx[0] + new_ry[0]*partRy[0] + new_rz[0]*partRz[0])>0) ? 1.0f:-1.0f;

  partCx += sign*(*C_steplength)/C_Ddims[0]*new_rx[0];
  partCy += sign*(*C_steplength)/C_Ddims[1]*new_ry[0];
  partCz += sign*(*C_steplength)/C_Ddims[2]*new_rz[0];
}

__device__ inline void first_jump(    tractographyData*    data_gpu,
                                      float*             new_rx,
                                      float*            new_ry,
                                      float*            new_rz,
                                      float*            partCx,
                                      float*            partCy,
                                      float*            partCz,
                                      float*             partRx,
                                      float*            partRy,
                                      float*            partRz,
                                      float3&         part_init,
                                      bool&            part_has_jumped,
                                      curandState&         localState)
{
  float sign=1.0f;
  bool init=false;
  //if(!m_simdiff){  always false
  if(part_has_jumped){
    sign=((new_rx[0]*partRx[0] + new_ry[0]*partRy[0] + new_rz[0]*partRz[0])>0) ? 1.0f:-1.0f;
  }else{
    sign=(curand_uniform(&localState)>0.5f)?1.0f:-1.0f;
    //jumpsign=sign;  NOT used never
    part_has_jumped=true;
    init=true;
  }

  partCx[0] += sign*(*C_steplength)/C_Ddims[0]*new_rx[0];
  partCy[0] += sign*(*C_steplength)/C_Ddims[1]*new_ry[0];
  partCz[0] += sign*(*C_steplength)/C_Ddims[2]*new_rz[0];

  partRx[0]=sign*new_rx[0];
  partRy[0]=sign*new_ry[0];
  partRz[0]=sign*new_rz[0];

  if(init){
    part_init.x=partRx[0];        // save first orientation
    part_init.y=partRy[0];        // next part will have
    part_init.z=partRz[0];        // opposite orientation
  }
}
__device__ inline void first_testjump(    tractographyData*    data_gpu,
                                          float*             new_rx,
                                          float*            new_ry,
                                          float*            new_rz,
                                          float&            partCx,        // new coordinates after jumping
                                          float&            partCy,
                                          float&            partCz,
                                          float*             partRx,
                                          float*            partRy,
                                          float*            partRz,
                                          bool&            part_has_jumped,
                                          curandState&         localState)
{
  float sign=1.0f;
  if(part_has_jumped){
    sign=((new_rx[0]*partRx[0] + new_ry[0]*partRy[0] + new_rz[0]*partRz[0])>0) ? 1.0f:-1.0f;
  }else{
    sign=(curand_uniform(&localState)>0.5f)?1.0f:-1.0f;
  }
  partCx += sign*(*C_steplength)/C_Ddims[0]*new_rx[0];
  partCy += sign*(*C_steplength)/C_Ddims[1]*new_ry[0];
  partCz += sign*(*C_steplength)/C_Ddims[2]*new_rz[0];
}

__device__ inline bool check_dir(    tractographyData*    data_gpu,
                                     float*             memSH_a,    // input: sample_th | output: new_rx
                                     float*            memSH_b,    // input: sample_ph | output: new_ry
                                     float*            memSH_c,    // input: sample_f | output: new_rz
                                     float*             memSH_d,
                                     float*            memSH_e,
                                     float*            memSH_f,
                                     float*             partRx,
                                     float*            partRy,
                                     float*            partRz,
                                     curandState&         localState)
{
  // volume fraction criterion....from fsamples
  // or outside mask when Probabilistic interpolation, then memSH_c (sample_f) is -1
  if(memSH_c[0]<=curand_uniform(&localState)) return false;

  // direction=0 ?
  if (memSH_a[0]==0 || memSH_b[0]==0) return false;

  // I use memSH to store new x,y,z directions.
  // That way I do not have to recompute them when jump

  //memSH_c cos(th)
  //memSH_d sin(th)
  //memSH_e sin(ph)
  //memSH_f cos(ph)
  sincosf(memSH_a[0],memSH_d,memSH_c);    // new rz = cos(th) -> memSH_c            // !!!  SINGLE PRECISION !!
  sincosf(memSH_b[0],memSH_e,memSH_f);

  memSH_a[0]=memSH_f[0]*memSH_d[0];      // new rx = cos(ph)*sin(th)
  memSH_b[0]=memSH_e[0]*memSH_d[0];     // new ry = sin(ph)*sin(th)

  /*memSH_c[0]=cos(memSH_b[0])*sin(memSH_a[0]);      // new rx = cos(ph)*sin(th)
    memSH_b[0]=sin(memSH_b[0])*sin(memSH_a[0]);     // new ry = sin(ph)*sin(th)
    memSH_a[0]=cos(memSH_a[0]);            // new rz = cos(th)    */

  // check curvature threshold
  if(fabsf(memSH_a[0]*partRx[0] + memSH_b[0]*partRy[0] + memSH_c[0]*partRz[0])>(*C_curv_thr))
    return true;
  else
    return false;
}

// check curvature threshold
__device__ inline bool first_check_dir(    tractographyData*    data_gpu,
                                           float*             memSH_a,    // input: sample_th | output: new_rx
                                           float*            memSH_b,    // input: sample_ph | output: new_ry
                                           float*            memSH_c,    // input: sample_f | output: new_rz
                                           float*             memSH_d,
                                           float*            memSH_e,
                                           float*            memSH_f,
                                           float*             partRx,
                                           float*            partRy,
                                           float*            partRz,
                                           const bool        part_has_jumped,
                                           curandState&         localState)
{
  // volume fraction criterion....from fsamples
  // or outside mask when Probabilistic interpolation
  if(memSH_c[0]<=curand_uniform(&localState)) return false;

  // direction=0 ?
  if (memSH_a[0]==0 || memSH_b[0]==0) return false;

  //memSH_c cos(th)
  //memSH_d sin(th)
  //memSH_e sin(ph)
  //memSH_f cos(ph)
  sincosf(memSH_a[0],memSH_d,memSH_c);        // new rz = cos(th) -> memSH_c
  sincosf(memSH_b[0],memSH_e,memSH_f);

  memSH_a[0]=memSH_f[0]*memSH_d[0];      // new rx = cos(ph)*sin(th)
  memSH_b[0]=memSH_e[0]*memSH_d[0];     // new ry = sin(ph)*sin(th)


  if(part_has_jumped){
    if(fabsf(memSH_a[0]*partRx[0] + memSH_b[0]*partRy[0] + memSH_c[0]*partRz[0])>(*C_curv_thr))
      return true;
    else
      return false;
  }else return true;
}


template<int mode>
__device__ inline int sample_fibre(    tractographyData*    data_gpu,
                                       const int         samp,
                                       const int         col,
                                       curandState&         localState)

{
  if(mode==0){
    return 0;
  }else if(mode==3){//sample all
    return int(floorf(data_gpu->nfibres*(curand_uniform(&localState)/DIVRND)));
  }else if(mode==1){//sample all>thresh
    int numfibs=0;
    int selection=0;
    for(int fib=0;fib<data_gpu->nfibres;fib++){
      if(data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col]>data_gpu->fibthresh){
        numfibs++;
      }
    }
    if(numfibs==0){
      return 0;
    }else{
      selection=floorf((curand_uniform(&localState)/DIVRND)*numfibs);
    }
    numfibs=0;
    /// NOT A BETTER WAY ??????????
    for(int fib=0;fib<data_gpu->nfibres;fib++){
      float f = data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
      if(f>data_gpu->fibthresh){
        if(numfibs==selection) return fib;
        numfibs++;
      }
    }
    return 0;
  }else if(mode==2){//sample all>thresh in proportion of f (default)
    float fsumtmp=0;
    for(int fib=0;fib<data_gpu->nfibres;fib++){
      float ft=data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
      if(ft>data_gpu->fibthresh){
        fsumtmp+=ft;  //count total weight of f in this voxel.
      }
    }
    if(fsumtmp==0){
      return(0);
    }else{
      float ft,fsumtmp2=0;
      float rtmp=fsumtmp * curand_uniform(&localState);
      for(int fib=0;fib<data_gpu->nfibres;fib++){
        ft=data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
        if(ft>data_gpu->fibthresh)
          fsumtmp2 += ft;
        if(rtmp<=fsumtmp2){
          return(fib);
        }
      }
      return(data_gpu->nfibres-1);  // just in case fsumtmp*rnd lose precision
    }
  }
  // }else{
  // ERROR
  // }

  return 0;
}

__device__ inline void sampleN(    tractographyData*    data_gpu,
                                   curandState&         localState,
                                   float*            partCx,
                                   float*            partCy,
                                   float*            partCz,
                                   float*             partRx,
                                   float*            partRy,
                                   float*            partRz,
                                   // OUTPUT
                                   float*             sample_th,
                                   float*            sample_ph,
                                   float*            sample_f,        //shared
                                   // to use
                                   float*            memSH_d,
                                   float*            memSH_e,
                                   float*            memSH_f)
{
  //////// Probabilistic interpolation
  /// if we are not in the middle of a voxel,
  // we want interpolate an orientation taking account the orientations of 2 voxels
  // instead of interpolating the orientation (in the case of perpendicular orientations
  // is not a good idea), we interpolate the coordinate, we go to one of the voxels and then
  // take its orientation. Imagine we are just in the border of two voxels, then same probability
  // to go to one than to other. Other wise, higher probability of taking orientation from to the closest
  // Real interpolation ?
  int3 newC;
  newC.x = curand_uniform(&localState)>(partCx[0]-floorf(partCx[0]))?floorf(partCx[0]):ceilf(partCx[0]);
  newC.y = curand_uniform(&localState)>(partCy[0]-floorf(partCy[0]))?floorf(partCy[0]):ceilf(partCy[0]);
  newC.z = curand_uniform(&localState)>(partCz[0]-floorf(partCz[0]))?floorf(partCz[0]):ceilf(partCz[0]);
  ////////////////////////////////////

  int col;
  if( (newC.z<0) || (newC.y<0) || (newC.x<0) ||
      (newC.z>=C_Dsizes[2]) || (newC.y>=C_Dsizes[1]) || (newC.x>=C_Dsizes[0]))
    col=-1;
  // what voxel? vol2mat tell us the index of the voxel inside the vector (vector does not include data outside the mask)
  // lut_vol2mat: extrapolation method is zeropad, i.e, if any coordinate is not inbounds, then value is 0.
  else
    col = data_gpu->lut_vol2mat[newC.z*C_Dsizes[0]*C_Dsizes[1]+newC.y*C_Dsizes[0]+newC.x]-1;
  sample_th[0]=0;
  sample_ph[0]=0;

  if(col==-1){ // outside brain mask. Will exit in check_dir
    sample_f[0]=-1.0f;
    return;
  }

  int samp=rintf(curand_uniform(&localState)*(data_gpu->nsamples-1));

  int fibind=0;

  if(data_gpu->nfibres>1){
    // if(sample_fib>0){ // pick specified fibre             // only happens with prefdir and it is not activated in this version
    //    fibind=sample_fibre(fsamples,samp,col,localState,nfibres,nsamples,nvoxels,sample_fib,fibthresh);
    //    sample[0].x=thsamples[fibind*nsamples*nvoxels+samp*nvoxels+col];
    //    sample[0].y=phsamples[fibind*nsamples*nvoxels+samp*nvoxels+col];
    // }else{
    // prefdirfile; option deleted
    // int locrule=0; option deleted locfibchoice

    // pick closest direction
    float dotmax=0.0f;
    for(int fib=0;fib<data_gpu->nfibres;fib++){
      if(data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col]>data_gpu->fibthresh){
        float thtmp=data_gpu->thsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
        float phtmp=data_gpu->phsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];

        sincosf(thtmp,memSH_d,memSH_e);    // use shared to store temporal results
        sincosf(phtmp,memSH_f,sample_f);// use shared to store temporal results
        // !!! OJO SINGLE PRECISION !!

        //float dottmp=fabs(sin(thtmp)*(cos(phtmp)*partRx[0] + sin(phtmp)*partRy[0]) + cos(thtmp)*partRz[0]);
        float dottmp=fabsf(memSH_d[0]*(sample_f[0]*partRx[0] + memSH_f[0]*partRy[0]) + memSH_e[0]*partRz[0]);

        if(dottmp>dotmax){
          dotmax=dottmp;
          sample_th[0]=thtmp;
          sample_ph[0]=phtmp;
          fibind=fib;
        }
      }
    }
    if(dotmax==0.0f){
      sample_th[0]=data_gpu->thsamples[samp*data_gpu->nvoxels+col];
      sample_ph[0]=data_gpu->phsamples[samp*data_gpu->nvoxels+col];
      fibind=0;
    }
    // }
  }else{
    sample_th[0]=data_gpu->thsamples[samp*data_gpu->nvoxels+col];
    sample_ph[0]=data_gpu->phsamples[samp*data_gpu->nvoxels+col];
  }

  if(data_gpu->usef){
    sample_f[0] = data_gpu->fsamples[fibind*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
  }else{
    sample_f[0]=1.0f;
  }
}
__device__ inline void sampleN_test(    tractographyData*    data_gpu,
                                        curandState&         localState,
                                        float&            partCx,        // store here results for the test
                                        float&            partCy,
                                        float&            partCz,
                                        float*             partRx,
                                        float*            partRy,
                                        float*            partRz,
                                        // to use
                                        float*            memSH_d,
                                        float*            memSH_e,
                                        float*            memSH_f)
{
  int3 newC;
  newC.x = curand_uniform(&localState)>(partCx-floorf(partCx))?floorf(partCx):ceilf(partCx);
  newC.y = curand_uniform(&localState)>(partCy-floorf(partCy))?floorf(partCy):ceilf(partCy);
  newC.z = curand_uniform(&localState)>(partCz-floorf(partCz))?floorf(partCz):ceilf(partCz);
  ////////////////////////////////////

  int col;
  if( (newC.z<0) || (newC.y<0) || (newC.x<0) ||
      (newC.z>=C_Dsizes[2]) || (newC.y>=C_Dsizes[1]) || (newC.x>=C_Dsizes[0]))
    col=-1;
  // what voxel? vol2mat tell us the index of the voxel inside the vector (vector does not include data outside the mask)
  // lut_vol2mat: extrapolation method is zeropad, i.e, if any coordinate is not inbounds, then value is 0.
  else
    col = data_gpu->lut_vol2mat[newC.z*C_Dsizes[0]*C_Dsizes[1]+newC.y*C_Dsizes[0]+newC.x]-1;
  partCx=0.0f;
  partCy=0.0f;

  if(col==-1){ // outside brain mask. Return 0
    partCz=0.0f;
    return;
  }
  int samp=rintf(curand_uniform(&localState)*(data_gpu->nsamples-1));
  int fibind=0;
  if(data_gpu->nfibres>1){
    float dotmax=0.0f;
    for(int fib=0;fib<data_gpu->nfibres;fib++){
      if(data_gpu->fsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col]>data_gpu->fibthresh){
        float thtmp=data_gpu->thsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
        float phtmp=data_gpu->phsamples[fib*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];

        sincosf(thtmp,memSH_d,memSH_e);    // use shared to store temporal results
        sincosf(phtmp,memSH_f,&partCz);// use shared to store temporal results

        float dottmp=fabsf(memSH_d[0]*(partCz*partRx[0] + memSH_f[0]*partRy[0]) + memSH_e[0]*partRz[0]);

        if(dottmp>dotmax){
          dotmax=dottmp;
          partCx=thtmp;
          partCy=phtmp;
          fibind=fib;
        }
      }
    }
    if(dotmax==0.0f){
      partCx=data_gpu->thsamples[samp*data_gpu->nvoxels+col];
      partCy=data_gpu->phsamples[samp*data_gpu->nvoxels+col];
      fibind=0;
    }
    // }
  }else{
    partCx=data_gpu->thsamples[samp*data_gpu->nvoxels+col];
    partCy=data_gpu->phsamples[samp*data_gpu->nvoxels+col];
  }
  if(data_gpu->usef){
    partCz = data_gpu->fsamples[fibind*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
  }else{
    partCz=1.0f;
  }
}

template <int randfib>
__device__ inline void init_sampleN(    tractographyData*    data_gpu,
                                        curandState&         localState,
                                        float*            partCx,
                                        float*            partCy,
                                        float*            partCz,
                                        //OUTPUT
                                        float*             sample_th,
                                        float*            sample_ph,
                                        float*            sample_f)        //shared
{
  //////// Probabilistic interpolation
  int3 newC;
  newC.x = curand_uniform(&localState)>(partCx[0]-floorf(partCx[0]))?floorf(partCx[0]):ceilf(partCx[0]);
  newC.y = curand_uniform(&localState)>(partCy[0]-floorf(partCy[0]))?floorf(partCy[0]):ceilf(partCy[0]);
  newC.z = curand_uniform(&localState)>(partCz[0]-floorf(partCz[0]))?floorf(partCz[0]):ceilf(partCz[0]);
  ////////////////////////////////////

  int col;
  if( (newC.z<0) || (newC.y<0) || (newC.x<0) ||
      (newC.z>=C_Dsizes[2]) || (newC.y>=C_Dsizes[1]) || (newC.x>=C_Dsizes[0]))
    col=-1;
  // what voxel? vol2mat tell us the index of the voxel inside the vector (vector does not include data outside the mask)
  // lut_vol2mat: extrapolation method is zeropad, i.e, if any coordinate is not inbounds, then value is 0.
  else
    col = data_gpu->lut_vol2mat[newC.z*C_Dsizes[0]*C_Dsizes[1]+newC.y*C_Dsizes[0]+newC.x]-1;
  sample_th[0]=0;
  sample_ph[0]=0;

  if(col==-1){ // outside brain mask. Will exit in check_dir
    sample_f[0]=-1;
    return;
  }

  int samp=rintf(curand_uniform(&localState)*(data_gpu->nsamples-1));

  int fibind=0;

  if(data_gpu->nfibres>1){
    // go for the specified fibre on the first jump or generate at random
    int myfibst=data_gpu->fibst;
    if(myfibst==-1){   // not set
      myfibst=sample_fibre<randfib>(data_gpu,samp,col,localState);
    }
    sample_th[0]=data_gpu->thsamples[myfibst*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
    sample_ph[0]=data_gpu->phsamples[myfibst*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
  }else{
    sample_th[0]=data_gpu->thsamples[samp*data_gpu->nvoxels+col];
    sample_ph[0]=data_gpu->phsamples[samp*data_gpu->nvoxels+col];
  }

  if(data_gpu->usef){
    sample_f[0] = data_gpu->fsamples[fibind*data_gpu->nsamples*data_gpu->nvoxels+samp*data_gpu->nvoxels+col];
  }else{
    sample_f[0]=1;
  }
}

__device__ inline void cart2sph(    float*     dir1,
                                    float*     dir2,
                                    float*    dir3,
                                    float*     th,
                                    float*     ph)
{
  float mag=sqrtf(dir1[0]*dir1[0]+dir2[0]*dir2[0]+dir3[0]*dir3[0]);
  if(mag==0){
    *ph=M_PI/2;
    *th=M_PI/2;
  }else{
    if(dir1[0]==0 && dir2[0]>=0) *ph=M_PI/2;
    else if(dir1[0]==0 && dir2[0]<0) *ph=-M_PI/2;
    else if(dir1[0]>0) *ph=atanf(dir2[0]/dir1[0]);
    else if(dir2[0]>0) *ph=atanf(dir2[0]/dir1[0])+M_PI;
    else *ph=atanf(dir2[0]/dir1[0])-M_PI;

    if(dir3[0]==0) *th=M_PI/2;
    else if(dir3[0]>0) *th=atanf(sqrtf(dir1[0]*dir1[0]+dir2[0]*dir2[0])/dir3[0]);
    else *th=atanf(sqrtf(dir1[0]*dir1[0]+dir2[0]*dir2[0])/dir3[0])+M_PI;
  }
}

__device__ inline void mean_sph_pol(    float*    A1,
                                        float*    A2,
                                        float*    A3,
                                        float&    B1,
                                        float&    B2,
                                        float&    B3,
                                        float*    aux1,
                                        float*    aux2,
                                        float*    aux3)
{
  // A is in and B contain th, ph f.
  // But A is already in cartesian coordinates. B in spherical coordinates

  sincosf(B1,aux1,&B3); // B3=cos(B(1))
  sincosf(B2,aux2,aux3);
  B1=aux1[0]*aux3[0]; // B1=(sin(B(1))*cos(B(2)))
  B2=aux1[0]*aux2[0]; // B2=(sin(B(1))*sin(B(2)))
  // rB << (sin(B(1))*cos(B(2))) << (sin(B(1))*sin(B(2))) << (cos(B(1)));

  float sum;
  sum=A1[0]*B1+A2[0]*B2+A3[0]*B3;
  if(sum>0){
    aux1[0]=(A1[0]+B1)/2;
    aux2[0]=(A2[0]+B2)/2;
    aux3[0]=(A3[0]+B3)/2;
  }else{
    aux1[0]=(A1[0]-B1)/2;
    aux2[0]=(A2[0]-B2)/2;
    aux3[0]=(A3[0]-B3)/2;
  }
  // cart2sph(aux1,aux2,aux3,A1,A2);
  // I keep cartesian coordinates to use them directly when jumping
  A1[0]=aux1[0];
  A2[0]=aux2[0];
  A3[0]=aux3[0];
}

// random sampling .. or not
template <int randfib,bool loopcheck,bool modeuler>
__device__ inline int streamline(
  tractographyData* data_gpu,
  curandState&      localState,        // Random state for this thread
  int*              loopcheckkeys,
  float3*           loopcheckdirs,
  WarpTexture*      s2d,               // seed2diff warp
  WarpTexture*      d2s,               // diff2seed warp
  // PARTICLE
  float*            partCx,            // coordinates in mm in DTI space - SHARED
  float*            partCy,            // coordinates in mm in DTI space - SHARED
  float*            partCz,            // coordinates in mm in DTI space - SHARED
  float*            partRx,            // rotation - SHARED
  float*            partRy,            // rotation - SHARED
  float*            partRz,            // rotation - SHARED
  float*            memSH_a,        // SHARED to use
  float*            memSH_b,        // SHARED to use
  float*            memSH_c,        // SHARED to use
  float*            memSH_d,        // SHARED to use
  float*            memSH_e,        // SHARED to use
  float*            memSH_f,        // SHARED to use
  // OUTPUT
  float*            m_path,            // Main Memory
  float3&           part_init,
  bool&             part_has_jumped)
{
  //int sampled_fib=data_gpu->fibst;  // not used
  int numloopcheck=0;

  // if not jump yet, will be 0
  partRx[0]=-part_init.x;
  partRy[0]=-part_init.y;
  partRz[0]=-part_init.z;

  int cnt=0;            // counter: number of points, no number of jumps (jumps=cnt-1)

  // find xyz in dti space
  if(!data_gpu->IsNonlinXfm){
    vox_to_vox_S2D(m_path,partCx,partCy,partCz);
  }else{
    NewimageCoord2NewimageCoord_S2D(d2s,
                                    m_path,
                                    memSH_a,memSH_b,memSH_c,memSH_d,memSH_e,memSH_f,
                                    partCx,partCy,partCz);
  }
  if(!inMask(data_gpu,partCx,partCy,partCz)){
    // outside mask, reject
    return(-1);
  }

  ////////////////////////////////////////////////////////////
  // Do a first step outside the loop to reduce divergences //
  ////////////////////////////////////////////////////////////
  /////// loopchecking //////
  if(loopcheck){
    int key=rintf(partCz[0]/LCRAT)*C_Dsizes[1]*C_Dsizes[0]+rintf(partCy[0]/LCRAT)*C_Dsizes[0]+rintf(partCx[0]/LCRAT);
    loopcheckdirs[0].x=partRx[0];
    loopcheckdirs[0].y=partRy[0];
    loopcheckdirs[0].z=partRz[0];
    loopcheckkeys[0]=key;
    numloopcheck++;
  }
  ////// end looopchecking //////

  //already stored first coordinates in m_path[0]
  cnt++;
  // sample a new fibre orientation
  init_sampleN<randfib>(data_gpu,localState,partCx,partCy,partCz,memSH_a,memSH_b,memSH_c);
  //memSH -> samples (th,ph,f)

  // check new direction and jump
  if(!first_check_dir(data_gpu,memSH_a,memSH_b,memSH_c,memSH_d,memSH_e,memSH_f,partRx,partRy,partRz,part_has_jumped,localState)){
    // volume fraction thresholh, direction=0 (th or ph is 0), or curvature threshold
    if(((cnt-1)*(*C_steplength)) < (*C_distthresh)) return(-1);
    return cnt;
  }else{
    if (!modeuler){
      first_jump(data_gpu,memSH_a,memSH_b,memSH_c,partCx,partCy,partCz,partRx,partRy,partRz,part_init,part_has_jumped,localState);
    }else{
      // to save new test coordinates-results and new th-ph from test
      float m_testx=*partCx;
      float m_testy=*partCy;
      float m_testz=*partCz;
      first_testjump(data_gpu,memSH_a,memSH_b,memSH_c,m_testx,m_testy,m_testz,partRx,partRy,partRz,part_has_jumped,localState);
      // sample a new fibre orientation from test coordinates
      // m_testx: thsample, m_testy: phsample, m_testz fsample
      sampleN_test(data_gpu,localState,m_testx,m_testy,m_testz,partRx,partRy,partRz,memSH_d,memSH_e,memSH_f);
      // mean (memSH_a,memSH_b,memSH_c AND m_testx,m_testy,m_testz), stored in memSH_a,memSH_b,memSH_c
      mean_sph_pol(memSH_a,memSH_b,memSH_c,m_testx,m_testy,m_testz,memSH_d,memSH_e,memSH_f);
      // finally jump
      first_jump(data_gpu,memSH_a,memSH_b,memSH_c,partCx,partCy,partCz,partRx,partRy,partRz,part_init,part_has_jumped,localState);
    }
  }
  if(!inMask(data_gpu,partCx,partCy,partCz)){
    // outside mask
    if(((cnt-1)*(*C_steplength)) < (*C_distthresh)) return(-1);
    return cnt;
  }
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////

  for(int it=1;it<(data_gpu->nsteps/2);it++){    // seed is counted as 1 step, as least cpu is like that
    /////// loopchecking //////
    if(loopcheck){
      int key=rintf(partCz[0]/LCRAT)*C_Dsizes[1]*C_Dsizes[0]+rintf(partCy[0]/LCRAT)*C_Dsizes[0]+rintf(partCx[0]/LCRAT);
      bool found=false;
      int loc;
      for(int i=0;((i<numloopcheck)&&(!found));i++){
        if(loopcheckkeys[i]==key){
          found=true;
          loc=i;
        }
      }
      if(found){
        if(loopcheckdirs[loc].x*partRx[0]+loopcheckdirs[loc].y*partRy[0]+loopcheckdirs[loc].z*partRz[0]<0) break;
        else{
          loopcheckdirs[loc].x=partRx[0];
          loopcheckdirs[loc].y=partRy[0];
          loopcheckdirs[loc].z=partRz[0];
        }
      }else{
        loopcheckdirs[numloopcheck].x=partRx[0];
        loopcheckdirs[numloopcheck].y=partRy[0];
        loopcheckdirs[numloopcheck].z=partRz[0];
        loopcheckkeys[numloopcheck]=key;
        numloopcheck++;
      }
    }
    ////// end looopchecking //////

    // now find xyz in seeds space
    if(!data_gpu->IsNonlinXfm){
      vox_to_vox_D2S(partCx,partCy,partCz,&m_path[cnt*3]);
    }else{
      NewimageCoord2NewimageCoord_D2S(s2d,
                                      partCx,partCy,partCz,
                                      memSH_a,memSH_b,memSH_c,memSH_d,memSH_e,memSH_f,
                                      &m_path[cnt*3]);
    }
    cnt++;

    // sample a new fibre orientation
    // memSH_a: thsample, memSH_b: phsample, memSH_c fsample
    sampleN(data_gpu,localState,partCx,partCy,partCz,partRx,partRy,partRz,memSH_a,memSH_b,memSH_c,memSH_d,memSH_e,memSH_f);

    // check new direction
    if(!check_dir(data_gpu,memSH_a,memSH_b,memSH_c,memSH_d,memSH_e,memSH_f,partRx,partRy,partRz,localState)){
      // curvature threshold || direction=zero (th or ph are 0) || volume fraction criterion ||
      // outside mask when Probabilistic interpolation
      break;
    }

    // JUMP
    if (!modeuler){
      jump(data_gpu,memSH_a,memSH_b,memSH_c,partCx,partCy,partCz,partRx,partRy,partRz,localState);
    }else{
      // to save new test coordinates-results and new th-ph from test
      float m_testx=*partCx;
      float m_testy=*partCy;
      float m_testz=*partCz;
      testjump(data_gpu,memSH_a,memSH_b,memSH_c,m_testx,m_testy,m_testz,partRx,partRy,partRz,localState);
      // sample a new fibre orientation from test coordinates
      // m_testx: thsample, m_testy: phsample, m_testz fsample
      sampleN_test(data_gpu,localState,m_testx,m_testy,m_testz,partRx,partRy,partRz,memSH_d,memSH_e,memSH_f);
      // mean (memSH_a,memSH_b,memSH_c AND m_testx,m_testy,m_testz), stored in memSH_a,memSH_b,memSH_c
      mean_sph_pol(memSH_a,memSH_b,memSH_c,m_testx,m_testy,m_testz,memSH_d,memSH_e,memSH_f);
      // finally jump
      jump(data_gpu,memSH_a,memSH_b,memSH_c,partCx,partCy,partCz,partRx,partRy,partRz,localState);
    }

    if(!inMask(data_gpu,partCx,partCy,partCz)) break; // outside mask

  } // Close Step Number Loop (done tracking sample)
  if(((cnt-1)*(*C_steplength)) < (*C_distthresh)) return(-1);
  return cnt;
}
