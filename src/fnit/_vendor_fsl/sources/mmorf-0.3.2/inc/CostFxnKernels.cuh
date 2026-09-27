//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief GPU kernels for spline based calculations
/// \details Functions useful for Gauss-Newton style optimisation strategies
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef COST_FXN_KERNELS_CUH
#define COST_FXN_KERNELS_CUH

namespace MMORF
{
  __global__ void kernel_jac_det(
      // Input
      const unsigned int jac_det_sz,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      // Output
      float* __restrict__ jac_det);

  __global__ void kernel_make_jte(
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
      // Output
      float* __restrict__ jte_values);

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

  __global__ void kernel_make_jtj_symmetrical_diag_hess_main(
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

  __global__ void kernel_make_jtj_symmetrical_diag_hess_off(
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
      float* __restrict__ jtj_values_1,
      float* __restrict__ jtj_values_2);

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

  __global__ void kernel_make_jtj_non_symmetrical_diag_hess_main(
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

  __global__ void kernel_make_jtj_non_symmetrical_diag_hess_off(
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
      float* __restrict__ jtj_values_1,
      float* __restrict__ jtj_values_2);

  __global__ void kernel_make_jtj_non_symmetrical_diag_hess_spred(
      // Input
      unsigned int ima_sz_x,
      unsigned int ima_sz_y,
      unsigned int ima_sz_z,
      const cudaTextureObject_t* __restrict__ ima,
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

  __global__ void kernel_make_bkk(
      // Input
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
      const bool twice_differentiated_spline,
      // Output
      float* __restrict__ jtj_values);

  __global__ void kernel_log_jacobian_cost_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      // Output
      float* __restrict__ cost);

  __global__ void kernel_log_jacobian_grad_jxx_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      // Output
      float* __restrict__ d_cost_d_j11,
      float* __restrict__ d_cost_d_j12,
      float* __restrict__ d_cost_d_j13,
      float* __restrict__ d_cost_d_j21,
      float* __restrict__ d_cost_d_j22,
      float* __restrict__ d_cost_d_j23,
      float* __restrict__ d_cost_d_j31,
      float* __restrict__ d_cost_d_j32,
      float* __restrict__ d_cost_d_j33);

  __global__ void kernel_log_jacobian_singular_values_cost_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      // Output
      float* __restrict__ cost);

  __global__ void kernel_log_jacobian_singular_values_grad_jxx_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      // Output
      float* __restrict__ d_cost_d_j11,
      float* __restrict__ d_cost_d_j12,
      float* __restrict__ d_cost_d_j13,
      float* __restrict__ d_cost_d_j21,
      float* __restrict__ d_cost_d_j22,
      float* __restrict__ d_cost_d_j23,
      float* __restrict__ d_cost_d_j31,
      float* __restrict__ d_cost_d_j32,
      float* __restrict__ d_cost_d_j33);

  __global__ void kernel_log_jacobian_singular_values_exact_cost_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      const float* __restrict__ u1,
      const float* __restrict__ u2,
      const float* __restrict__ u3,
      // Output
      float* __restrict__ cost);

  __global__ void kernel_log_jacobian_singular_values_exact_grad_jxx_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ j11,
      const float* __restrict__ j12,
      const float* __restrict__ j13,
      const float* __restrict__ j21,
      const float* __restrict__ j22,
      const float* __restrict__ j23,
      const float* __restrict__ j31,
      const float* __restrict__ j32,
      const float* __restrict__ j33,
      const float* __restrict__ u1,
      const float* __restrict__ u2,
      const float* __restrict__ u3,
      const float* __restrict__ v11,
      const float* __restrict__ v12,
      const float* __restrict__ v13,
      const float* __restrict__ v21,
      const float* __restrict__ v22,
      const float* __restrict__ v23,
      const float* __restrict__ v31,
      const float* __restrict__ v32,
      const float* __restrict__ v33,
      // Output
      float* __restrict__ d_cost_d_j11,
      float* __restrict__ d_cost_d_j12,
      float* __restrict__ d_cost_d_j13,
      float* __restrict__ d_cost_d_j21,
      float* __restrict__ d_cost_d_j22,
      float* __restrict__ d_cost_d_j23,
      float* __restrict__ d_cost_d_j31,
      float* __restrict__ d_cost_d_j32,
      float* __restrict__ d_cost_d_j33);

  __global__ void kernel_rotation_gxx_grad_jxx_3d(
      // Input
      const unsigned int        n_samples,
      const float* __restrict__ g11,
      const float* __restrict__ g12,
      const float* __restrict__ g13,
      const float* __restrict__ g22,
      const float* __restrict__ g23,
      const float* __restrict__ g33,
      const float* __restrict__ r11,
      const float* __restrict__ r12,
      const float* __restrict__ r13,
      const float* __restrict__ r21,
      const float* __restrict__ r22,
      const float* __restrict__ r23,
      const float* __restrict__ r31,
      const float* __restrict__ r32,
      const float* __restrict__ r33,
      const float* __restrict__ s11,
      const float* __restrict__ s12,
      const float* __restrict__ s13,
      const float* __restrict__ s22,
      const float* __restrict__ s23,
      const float* __restrict__ s33,
      // Output
      float* __restrict__       d_g11_d_j11,
      float* __restrict__       d_g11_d_j12,
      float* __restrict__       d_g11_d_j13,
      float* __restrict__       d_g11_d_j21,
      float* __restrict__       d_g11_d_j22,
      float* __restrict__       d_g11_d_j23,
      float* __restrict__       d_g11_d_j31,
      float* __restrict__       d_g11_d_j32,
      float* __restrict__       d_g11_d_j33,
      float* __restrict__       d_g12_d_j11,
      float* __restrict__       d_g12_d_j12,
      float* __restrict__       d_g12_d_j13,
      float* __restrict__       d_g12_d_j21,
      float* __restrict__       d_g12_d_j22,
      float* __restrict__       d_g12_d_j23,
      float* __restrict__       d_g12_d_j31,
      float* __restrict__       d_g12_d_j32,
      float* __restrict__       d_g12_d_j33,
      float* __restrict__       d_g13_d_j11,
      float* __restrict__       d_g13_d_j12,
      float* __restrict__       d_g13_d_j13,
      float* __restrict__       d_g13_d_j21,
      float* __restrict__       d_g13_d_j22,
      float* __restrict__       d_g13_d_j23,
      float* __restrict__       d_g13_d_j31,
      float* __restrict__       d_g13_d_j32,
      float* __restrict__       d_g13_d_j33,
      float* __restrict__       d_g22_d_j11,
      float* __restrict__       d_g22_d_j12,
      float* __restrict__       d_g22_d_j13,
      float* __restrict__       d_g22_d_j21,
      float* __restrict__       d_g22_d_j22,
      float* __restrict__       d_g22_d_j23,
      float* __restrict__       d_g22_d_j31,
      float* __restrict__       d_g22_d_j32,
      float* __restrict__       d_g22_d_j33,
      float* __restrict__       d_g23_d_j11,
      float* __restrict__       d_g23_d_j12,
      float* __restrict__       d_g23_d_j13,
      float* __restrict__       d_g23_d_j21,
      float* __restrict__       d_g23_d_j22,
      float* __restrict__       d_g23_d_j23,
      float* __restrict__       d_g23_d_j31,
      float* __restrict__       d_g23_d_j32,
      float* __restrict__       d_g23_d_j33,
      float* __restrict__       d_g33_d_j11,
      float* __restrict__       d_g33_d_j12,
      float* __restrict__       d_g33_d_j13,
      float* __restrict__       d_g33_d_j21,
      float* __restrict__       d_g33_d_j22,
      float* __restrict__       d_g33_d_j23,
      float* __restrict__       d_g33_d_j31,
      float* __restrict__       d_g33_d_j32,
      float* __restrict__       d_g33_d_j33);

} // namespace MMORF
#endif // COST_FXN_KERNELS_CUH
