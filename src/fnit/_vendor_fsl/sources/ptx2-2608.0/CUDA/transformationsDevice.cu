/*  streamlines_kernel.cu

    Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

__constant__ float C_Sdims[3];
__constant__ float C_Ddims[3];
__constant__ float C_Wsampling_S2D_I[3];
__constant__ float C_Wsampling_D2S_I[3];
__constant__ float C_SsamplingI[3];
__constant__ float C_DsamplingI[3];
__constant__ float C_Seeds_to_DTI[12];
__constant__ float C_DTI_to_Seeds[12];
__constant__ float C_Seeds_to_M2[12];
__constant__ int C_Ssizes[3];
__constant__ int C_Dsizes[3];
__constant__ int C_M2sizes[3];
__constant__ int C_Warp_S2D_sizes[3];
__constant__ int C_Warp_D2S_sizes[3];


typedef struct {
  cudaTextureObject_t x;
  cudaTextureObject_t y;
  cudaTextureObject_t z;
} WarpTexture;

WarpTexture T_SeedDTIwarp;
WarpTexture T_DTISeedwarp;

__device__ inline float Tinterpolate(
  cudaTextureObject_t T,
  float*              coordx, // shared memory
  float*              coordy, // shared memory
  float*              coordz) // shared memory
{
  // p_interpmethod=trilinear
  int ix,iy,iz;
  ix= floorf(coordx[0]);
  iy= floorf(coordy[0]);
  iz= floorf(coordz[0]);
  float dx,dy,dz;
  dx=coordx[0]-ix;
  dy=coordy[0]-iy;
  dz=coordz[0]-iz;

  float aux0=tex3D<float>(T, ix, iy+1, iz+1); //V011
  float aux1=(tex3D<float>(T, ix+1, iy+1,iz+1) - aux0)*dx + aux0; //V111,V011
  aux0=tex3D<float>(T, ix, iy, iz+1); //V001
  float aux2=(tex3D<float>(T, ix+1 ,iy, iz+1) - aux0)*dx + aux0; //V101,V001
  float temp6=(aux1 - aux2)*dy + aux2;

  aux0=tex3D<float>(T, ix, iy+1, iz); //V010
  aux1=(tex3D<float>(T, ix+1, iy+1, iz) - aux0)*dx + aux0; //V110,V010
  aux0=tex3D<float>(T, ix, iy, iz); //V000
  aux2=(tex3D<float>(T, ix+1, iy, iz) - aux0)*dx + aux0; //V100,V000
  float temp5 = (aux1 - aux2)*dy + aux2;

  return (temp6 - temp5)*dz + temp5;
}


// fsl/src/warpfns/warpfns.h
__device__ inline void NewimageCoord2NewimageCoord_S2D(
  WarpTexture*  tex,
  const float*  inCoords, // Global memory
  // shared memory to use
  float*        memSH_a,
  float*        memSH_b,
  float*        memSH_c,
  float*        memSH_d,
  float*        memSH_e,
  float*        memSH_f,
  // OUTPUT
  float*        outCoordx,
  float*        outCoordy,
  float*        outCoordz)
{
  // Do the conversion
  // raw_newimagecoord2newimagecoord(0,&warps,inv_flag,0,srcvol,destvol,tmp);
  // coord=src.sampling_mat*coord;
  memSH_d[0] = C_Sdims[0]*inCoords[0];
  memSH_e[0] = C_Sdims[1]*inCoords[1];
  memSH_f[0] = C_Sdims[2]*inCoords[2];

  memSH_a[0] = C_Wsampling_D2S_I[0]*memSH_d[0];
  memSH_b[0] = C_Wsampling_D2S_I[1]*memSH_e[0];
  memSH_c[0] = C_Wsampling_D2S_I[2]*memSH_f[0];

  memSH_d[0] += Tinterpolate(tex->x, memSH_a, memSH_b, memSH_c);
  memSH_e[0] += Tinterpolate(tex->y, memSH_a, memSH_b, memSH_c);
  memSH_f[0] += Tinterpolate(tex->z, memSH_a, memSH_b, memSH_c);

  outCoordx[0]=C_DsamplingI[0]*memSH_d[0];
  outCoordy[0]=C_DsamplingI[1]*memSH_e[0];
  outCoordz[0]=C_DsamplingI[2]*memSH_f[0];
  // END raw_newimagecoord2newimagecoord
}

__device__ inline void NewimageCoord2NewimageCoord_D2S(
  WarpTexture* tex,
  const float* inCoordx,
  const float* inCoordy,
  const float* inCoordz,
  // shared memory to use
  float*       memSH_a,
  float*       memSH_b,
  float*       memSH_c,
  float*       memSH_d,
  float*       memSH_e,
  float*       memSH_f,
  // OUTPUT
  float*       outCoords)    // Global memory
{
  // Do the conversion
  // raw_newimagecoord2newimagecoord(0,&warps,inv_flag,0,srcvol,destvol,tmp);
  // coord=src.sampling_mat*coord;
  // memSH=warps->sampling_mat.i()*coord;
  memSH_a[0] = C_Wsampling_S2D_I[0]*C_Ddims[0]*inCoordx[0];
  memSH_b[0] = C_Wsampling_S2D_I[1]*C_Ddims[1]*inCoordy[0];
  memSH_c[0] = C_Wsampling_S2D_I[2]*C_Ddims[2]*inCoordz[0];

  memSH_d[0] = C_Ddims[0]*inCoordx[0] + Tinterpolate(tex->x, memSH_a, memSH_b, memSH_c);
  memSH_e[0] = C_Ddims[1]*inCoordy[0] + Tinterpolate(tex->y, memSH_a,memSH_b,memSH_c);
  memSH_f[0] = C_Ddims[2]*inCoordz[0] + Tinterpolate(tex->z, memSH_a,memSH_b,memSH_c);

  // coord = trgt.sampling_mat().i()*coord;
  outCoords[0]=C_SsamplingI[0]*memSH_d[0];
  outCoords[1]=C_SsamplingI[1]*memSH_e[0];
  outCoords[2]=C_SsamplingI[2]*memSH_f[0];
  // END raw_newimagecoord2newimagecoord
}

// fsl/src/miscmaths/miscmaths.h
// precalculated
// A xyz1 3x1, B dims1 3x1, C dims2 3x1, D xfm 4x4, E&F aux 3x1, S xyz2 3x1
// E=A*B(mult elements)
// F=DxE
// F=F/F(4)
// S=F/C
// precompute B*D(elements)/(F4*C) and mult by A
__device__ inline void vox_to_vox_S2D(    const float*    inCoords, // Global memory
                                        //OUTPUT
                                        float*        x2,
                                        float*        y2,
                                        float*        z2)
{
  x2[0]=C_Seeds_to_DTI[0]*inCoords[0]+C_Seeds_to_DTI[1]*inCoords[1]+C_Seeds_to_DTI[2]*inCoords[2]+C_Seeds_to_DTI[3];
  y2[0]=C_Seeds_to_DTI[4]*inCoords[0]+C_Seeds_to_DTI[5]*inCoords[1]+C_Seeds_to_DTI[6]*inCoords[2]+C_Seeds_to_DTI[7];
  z2[0]=C_Seeds_to_DTI[8]*inCoords[0]+C_Seeds_to_DTI[9]*inCoords[1]+C_Seeds_to_DTI[10]*inCoords[2]+C_Seeds_to_DTI[11];
}

__device__ inline void vox_to_vox_D2S(    const float*    x1,
                                        const float*    y1,
                                        const float*    z1,
                                        //OUTPUT
                                        float*        outCoords) // Global memory
{
  outCoords[0]=C_DTI_to_Seeds[0]*x1[0]+C_DTI_to_Seeds[1]*y1[0]+C_DTI_to_Seeds[2]*z1[0]+C_DTI_to_Seeds[3];
  outCoords[1]=C_DTI_to_Seeds[4]*x1[0]+C_DTI_to_Seeds[5]*y1[0]+C_DTI_to_Seeds[6]*z1[0]+C_DTI_to_Seeds[7];
  outCoords[2]=C_DTI_to_Seeds[8]*x1[0]+C_DTI_to_Seeds[9]*y1[0]+C_DTI_to_Seeds[10]*z1[0]+C_DTI_to_Seeds[11];
}

// for Matrix2
__device__ inline void vox_to_vox_S2M2(    const float*    inCoords, // Global memory
                                        //OUTPUT
                                        float*        x2,
                                        float*        y2,
                                        float*        z2)
{
  x2[0]=C_Seeds_to_M2[0]*inCoords[0]+C_Seeds_to_M2[1]*inCoords[1]+C_Seeds_to_M2[2]*inCoords[2]+C_Seeds_to_M2[3];
  y2[0]=C_Seeds_to_M2[4]*inCoords[0]+C_Seeds_to_M2[5]*inCoords[1]+C_Seeds_to_M2[6]*inCoords[2]+C_Seeds_to_M2[7];
  z2[0]=C_Seeds_to_M2[8]*inCoords[0]+C_Seeds_to_M2[9]*inCoords[1]+C_Seeds_to_M2[10]*inCoords[2]+C_Seeds_to_M2[11];
  //printf("transforming %f %f %f    TO      %f %f %f \n",inCoords[0],inCoords[1],inCoords[2],x2[0],y2[0],z2[0]);
}
