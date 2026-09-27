// Author:  Frederik Lange
// Date:    01/09/2017
//
// These are the Kernels required to calculate the hessian for a single gradient image
#ifndef EDDY_HESSIAN_KERNELS_CUH
#define EDDY_HESSIAN_KERNELS_CUH

namespace MMORF
{
  __global__ void kernel_make_jtj_symmetrical(
      // Input
      unsigned int ima_sz_x,
      unsigned int ima_sz_y,
      unsigned int ima_sz_z,
      cudaTextureObject_t ima,
      const float* __restrict__ spl_x,
      const float* __restrict__ spl_y,
      const float* __restrict__ spl_z,
      unsigned int spl_ksp_x,
      unsigned int spl_ksp_y,
      unsigned int spl_ksp_z,
      unsigned int param_sz_x,
      unsigned int param_sz_y,
      unsigned int param_sz_z,
      const int* __restrict__ jtj_offsets,
      // Output
      float* __restrict__ jtj_values);

  __global__ void kernel_make_jtj_non_symmetrical(
      // Input
      unsigned int ima_sz_x,
      unsigned int ima_sz_y,
      unsigned int ima_sz_z,
      cudaTextureObject_t ima,
      const float* __restrict__ spl_x_1,
      const float* __restrict__ spl_y_1,
      const float* __restrict__ spl_z_1,
      const float* __restrict__ spl_x_2,
      const float* __restrict__ spl_y_2,
      const float* __restrict__ spl_z_2,
      unsigned int spl_ksp_x_1,
      unsigned int spl_ksp_y_1,
      unsigned int spl_ksp_z_1,
      unsigned int spl_ksp_x_2,
      unsigned int spl_ksp_y_2,
      unsigned int spl_ksp_z_2,
      unsigned int param_sz_x_1,
      unsigned int param_sz_y_1,
      unsigned int param_sz_z_1,
      unsigned int param_sz_x_2,
      unsigned int param_sz_y_2,
      unsigned int param_sz_z_2,
      const int* __restrict__ jtj_offsets,
      // Output
      float* __restrict__ jtj_values);

  __global__ void make_hessian(
                  // Input
                  unsigned int ima_szx, unsigned int ima_szy, unsigned int ima_szz,
                  cudaTextureObject_t ima1,
                  const float* __restrict__ spl_x,
                  const float* __restrict__ spl_y,
                  const float* __restrict__ spl_z,
                  unsigned int kspx, unsigned int kspy, unsigned int kspz,
                  unsigned int spl_szx, unsigned int spl_szy, unsigned int spl_szz,
                  // Output
                  float* __restrict__ hess_val);

  __global__ void sparse_matrix_vector_multiply(
                                                  // Input
                                                  const float *mat_vals,
                                                  const float *vec,
                                                  const unsigned int dimension,
                                                  const unsigned int mat_id,
                                                  const unsigned int vec_id,
                                                  const unsigned int result_id,
                                                  // Input/Output
                                                  float *result);

  __global__ void sparse_matrix_vector_multiply(
                                                  // Input
                                                  const float* __restrict__ mat_vals,
                                                  const unsigned int* __restrict__ mat_rows,
                                                  const float* __restrict__ vec,
                                                  const unsigned int dimension,
                                                  // Output
                                                  float* __restrict__ result);
 
}

#endif
