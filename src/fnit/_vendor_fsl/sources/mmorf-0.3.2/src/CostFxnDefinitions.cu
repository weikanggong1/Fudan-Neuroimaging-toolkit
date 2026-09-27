//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Implementation of cuda-specific code
/// \details Specifically CostFxnHelpers and CostFxnKernels functions
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "CostFxnHelpers.cuh"
#include "CostFxnKernels.cuh"

#include <cuda.h>

#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////
// CostFxnHelpers Definitions
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
    /// Convert from volume coordinates to linear coordinates
    __host__ __device__ void index_vol_to_lin(
                                // Input
                                // Vol coords
                                unsigned int xind, unsigned int yind, unsigned int zind,
                                // Vol dims
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                // Output
                                // Index into linear array
                                unsigned int *lind)
    {
        *lind = zind*(szx*szy) + yind*(szx) + xind;
    }

    /// Convert from linear coordinates to volume coordinates
    __host__ __device__ void index_lin_to_vol(
                                // Input
                                // Index into linear array
                                unsigned int lind,
                                // Vol dims
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                 // Output
                                unsigned int *xind, unsigned int *yind, unsigned int *zind)
    {
        *zind = lind/(szx*szy);
        *yind = (lind - *zind*(szx*szy))/szx;
        *xind = lind - *zind*(szx*szy) - *yind*(szx);
    }

    /// Convert from linear coordinates back to linear coordinates of different spaces
    __host__ __device__ void index_lin_to_lin(
                                // Input
                                // Original linear index
                                unsigned int lind_in,
                                // size of original volume space
                                unsigned int lv_szx, unsigned int lv_szy, unsigned int lv_szz,
                                // size of new volume space
                                unsigned int vl_szx, unsigned int vl_szy, unsigned int vl_szz,
                                // Output
                                unsigned int *lind_out)
    {
        // Intermediate variables
        unsigned int xind = 0;
        unsigned int yind = 0;
        unsigned int zind = 0;

        // Convert to original volume space coordinates
        index_lin_to_vol(lind_in,lv_szx,lv_szy,lv_szz,&xind,&yind,&zind);
        index_vol_to_lin(xind,yind,zind,vl_szx,vl_szy,vl_szz,lind_out);
    }

    /// Calculate the start and end row & column for each diagonal in the Hessian
    __host__ __device__ void identify_diagonal(
                                // Input
                                // Index of current diagonal
                                unsigned int diag_ind,
                                // No. of overlapping splines in each direction
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                // Total no. of splines in each direction
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                // Output
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column)
    {
        // Reference variables
        unsigned int hess_side_length = spl_x*spl_y*spl_z;

        // Ensure the main diagonal value is valid
        unsigned int main_diag_lind = (rep_x*rep_y*rep_z-1)/2;
        unsigned int hess_main_diag_lind = 0;
        unsigned int hess_lind = 0;
        // Calculate linear index into hessian
        index_lin_to_lin(main_diag_lind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_main_diag_lind);
        index_lin_to_lin(diag_ind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_lind);
        // Deal with below main diagonal
        if (diag_ind < main_diag_lind)
        {
            *first_row = hess_main_diag_lind - hess_lind;
            *last_row = hess_side_length - 1;
            *first_column = 0;
            *last_column = hess_side_length - *first_row - 1;
        }

        // Deal with main diagonal and above
        else if (diag_ind >= main_diag_lind)
        {
            *first_column = hess_lind - hess_main_diag_lind;
            *last_column = hess_side_length - 1;
            *first_row = 0;
            *last_row = hess_side_length - *first_column - 1;
        }
    }

    /// Get the dimensions of the spline field coeffients based on warp parameterisaton and
    /// image size
    __host__ std::vector<unsigned int> get_spl_coef_dim(const std::vector<unsigned int>& ksp,
                                                        const std::vector<unsigned int>& isz)
    {
      std::vector<unsigned int> rval(ksp.size());
      for (unsigned int i=0; i<ksp.size(); i++)
      {
          rval[i] = static_cast<unsigned int>(std::ceil(float(isz[i]+1) / float(ksp[i]))) + 2;
      }
      return(rval);
    }
} // namespace MMORF

//////////////////////////////////////////////////////////////////////////////////////////////
// Functions purely for use in JtJ calculations
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
  __device__ void calculate_diagonal_range(
      // Input
      const int offset,
      const unsigned int n_rows,
      const unsigned int n_cols,
      // Output
      unsigned int *first_row,
      unsigned int *last_row,
      unsigned int *first_col,
      unsigned int *last_col)
  {
    // Below main diagonal
    if (offset < 0){
      *first_row = -offset;
      *last_row = n_rows -1;
      *first_col = 0;
      *last_col = n_cols + offset -1;
    }
    // On or above main diagonal
    else{
      *first_row = 0;
      *last_row = n_rows - offset -1;
      *first_col = offset;
      *last_col = n_cols -1;
    }
  }

  __device__ void calculate_overlap(
      // Input
      const int diff_mid,
      const unsigned int ksp,
      const int spl_order,
      // Output
      unsigned int *spl1_start,
      unsigned int *spl1_end,
      unsigned int *spl2_start)
  {
    // spl1 left of spl2
    if (diff_mid < 0){
      *spl1_start = static_cast<unsigned int>(-diff_mid)*ksp;
      *spl1_end = (ksp * (spl_order + 1)) - 2;
      *spl2_start = 0;
    }
    // spl1 right of spl2 or total overlap
    else{
      *spl1_start = 0;
      *spl1_end = (ksp * (spl_order + 1)) - 2 - (diff_mid * ksp);
      *spl2_start = diff_mid * ksp;
    }
  } // calculate_overlap

  __device__ bool is_valid_index(
      const int index,
      const unsigned int n_vals)
  {
    if (index < 0) return false;
    else if (static_cast<unsigned int>(index) >= n_vals) return false;
    else return true;
  } // is_valid_index
} // namespace MMORF

//////////////////////////////////////////////////////////////////////////////////////////////
// Jacobian function definitions
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details  Simple function for the rapid calculation of the Jacobian determinant of
  ///           multiple 3x3 Jacobian matrices of the form:
  ///
  ///           [j11 j12 j13
  ///            j21 j22 j23
  ///            j31 j32 j33]
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// blockIdx.x = number of the current chunk within the results vector
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each
  ///              chunk within the results vector)
  /// threadIdx.x = number of the current thread within the chunk
  /// The location in the results vector is then blockIdx.x * blockDim.x + threadIdx.x
  ///
  /// \param jac_det_sz number of matrices being operated on
  /// \param j11 pointer to vector of j11 elements with length jac_det_sz
  /// \param j12 pointer to vector of j12 elements with length jac_det_sz
  /// \param j13 pointer to vector of j13 elements with length jac_det_sz
  /// \param j21 pointer to vector of j21 elements with length jac_det_sz
  /// \param j22 pointer to vector of j22 elements with length jac_det_sz
  /// \param j23 pointer to vector of j23 elements with length jac_det_sz
  /// \param j31 pointer to vector of j31 elements with length jac_det_sz
  /// \param j32 pointer to vector of j32 elements with length jac_det_sz
  /// \param j33 pointer to vector of j33 elements with length jac_det_sz
  ///
  /// \param jac_det pointer to vector for storing resulting jac_det values with length
  ///                jac_det_sz
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jac_det)
  {
    // Only continue if this thread represents a valid point in the result vector
    const auto this_jac_det = blockIdx.x*blockDim.x + threadIdx.x;
    if (this_jac_det < jac_det_sz){
      jac_det[this_jac_det] =
            (j11[this_jac_det] + 1.0f)*(
              (j22[this_jac_det] + 1.0f)*(j33[this_jac_det] + 1.0f) - j23[this_jac_det]*j32[this_jac_det])
          - j12[this_jac_det]*(
              j21[this_jac_det]*(j33[this_jac_det] + 1.0f) - j23[this_jac_det]*j31[this_jac_det])
          + j13[this_jac_det]*(
              j21[this_jac_det]*j32[this_jac_det] - (j22[this_jac_det] + 1.0f)*j31[this_jac_det]);
    }
  }// kernel_jac_det

} // Jacobian function definitions

//////////////////////////////////////////////////////////////////////////////////////////////
// HessianKernels Definitions
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating the Jte for use in a nonlinear
  ///          optimisation algorithm. In this case it calculates the product of the Jacobian
  ///          matrix of partial derivatives of the cost fxn wrt the function parameters for
  ///          all voxels and then multiplies this by the current error-volume and sums if,
  ///          resulting in a vector with the same length as the number of parameters being
  ///          optimised over.
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of chunks the results vector is broken into
  /// blockIdx.y = number of the current chunk within the results vector
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each
  ///              chunk within the results vector)
  ///
  /// Additionally, it is assumed that the error volume and derivative volume of the moving
  /// volume (resampled into the reference object space) have been pre-multiplied.
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x 1D spline kernel in x-direction
  /// \param spl_y 1D spline kernel in y-direction
  /// \param spl_z 1D spline kernel in z-direction
  /// \param spl_ksp_x knot spacing of spline in x-direction
  /// \param spl_ksp_y knot spacing of spline in y-direction
  /// \param spl_ksp_z knot spacing of spline in z-direction
  /// \param param_sz_x size of parameter space in x-direction
  /// \param param_sz_y size of parameter space in y-direction
  /// \param param_sz_z size of parameter space in z-direction
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jte_values)
  {
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];

    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();

    // Only continue if this thread represents a valid point in the result vector
    const auto this_jte = blockIdx.x*blockDim.x + threadIdx.x;
    const auto jte_sz = param_sz_x * param_sz_y * param_sz_z;
    if (this_jte < jte_sz){
      // Calculate some constants
      const unsigned int spl_xind_end = spl_ksp_x * (spl_order + 1) - 2;
      const unsigned int spl_yind_end = spl_ksp_y * (spl_order + 1) - 2;
      const unsigned int spl_zind_end = spl_ksp_z * (spl_order + 1) - 2;
      // Calculate centres of spline
      unsigned int spl_mid_x, spl_mid_y, spl_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          this_jte,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl_mid_x,
          &spl_mid_y,
          &spl_mid_z);
      // Use spl indices to calculate the corresponding indexes into the image volume
      // NOTE: These values may be negative
      int vol_xind_start =
        static_cast<int>(spl_mid_x*spl_ksp_x) // Centre of spline in volume
        - static_cast<int>(spl_order*spl_ksp_x - 1); // Deal with spline "0" being outside volume
      int vol_yind_start =
        static_cast<int>(spl_mid_y*spl_ksp_y) // Centre of spline in volume
        - static_cast<int>(spl_order*spl_ksp_y - 1); // Deal with spline "0" being outside volume
      int vol_zind_start =
        static_cast<int>(spl_mid_z*spl_ksp_z) // Centre of spline in volume
        - static_cast<int>(spl_order*spl_ksp_z - 1); // Deal with spline "0" being outside volume
      // Calculate value in JtJ
      // This is done via a nested FOR loop, iterating through volume with x-direction
      // varying fastest
      /// \todo Figure this bit out again and document it
      int i_start = 0;
      int i_end = spl_xind_end;
      if (vol_xind_start < 0) i_start = -vol_xind_start;
      if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
      int j_start = 0;
      int j_end = spl_yind_end;
      if (vol_yind_start < 0) j_start = -vol_yind_start;
      if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
      int k_start = 0;
      int k_end = spl_zind_end;
      if (vol_zind_start < 0) k_start = -vol_zind_start;
      if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
      float jte_val = 0.0;
      // The big loop
      for (auto k = k_start; k <= k_end; ++k){
        for (auto j = j_start; j <= j_end; ++j){
          for (auto i = i_start; i <= i_end; ++i){
            int vol_zind = vol_zind_start + k;
            int vol_yind = vol_yind_start + j;
            int vol_xind = vol_xind_start + i;
            // All indices are valid, therefore calculate a value
            unsigned int vol_lind = 0;
            unsigned int spl_zind = k;
            unsigned int spl_yind = j;
            unsigned int spl_xind = i;
            MMORF::index_vol_to_lin(
                // Input
                vol_xind,
                vol_yind,
                vol_zind,
                ima_sz_x,
                ima_sz_y,
                ima_sz_z,
                // Output
                &vol_lind);
            // The big calc
            jte_val += tex1Dfetch<float>(ima,vol_lind)
                * shared_spline_x[spl_xind]
                * shared_spline_y[spl_yind]
                * shared_spline_z[spl_zind];
          }
        }
      }
      // Save final value
      jte_values[this_jte] = jte_val;
    }
  } // kernel_make_jtj_symmetrical

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.y  = total number of chunks each diagonal is broken into
  /// blockIdx.x = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.y = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x 1D spline kernel in x-direction
  /// \param spl_y 1D spline kernel in y-direction
  /// \param spl_z 1D spline kernel in z-direction
  /// \param spl_ksp_x knot spacing of spline in x-direction
  /// \param spl_ksp_y knot spacing of spline in y-direction
  /// \param spl_ksp_z knot spacing of spline in z-direction
  /// \param param_sz_x size of parameter space in x-direction
  /// \param param_sz_y size of parameter space in y-direction
  /// \param param_sz_z size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x * param_sz_y * param_sz_z;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];
    // We only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.x];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
      // Find difference in centres to calculate orientation.
      // NB this needs to be a signed operation
      auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
      auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
      auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_x,
          spl_ksp_x,
          spl_order,
          // Output
          &spl1_xind_start,
          &spl1_xind_end,
          &spl2_xind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_y,
          spl_ksp_y,
          spl_order,
          // Output
          &spl1_yind_start,
          &spl1_yind_end,
          &spl2_yind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_z,
          spl_ksp_z,
          spl_order,
          // Output
          &spl1_zind_start,
          &spl1_zind_end,
          &spl2_zind_start);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - diag_first_row;

    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_diag)) return;
    else if (!is_valid_index(this_col, jtj_sz_diag)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z, spl_order + 1)) return;
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + static_cast<int>(k);
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + static_cast<int>(j);
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + static_cast<int>(i);
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x[spl1_xind]
              * shared_spline_y[spl1_yind]
              * shared_spline_z[spl1_zind]
              * shared_spline_x[spl2_xind]
              * shared_spline_y[spl2_yind]
              * shared_spline_z[spl2_zind];
        }
      }
    }
    // Calculate levels of symmetry
    unsigned int symm_1, symm_2, symm_3;
    MMORF::index_lin_to_vol(
        // Input
        blockIdx.x,
        2*spl_order + 1,
        2*spl_order + 1,
        2*spl_order + 1,
        // Output
        &symm_1,
        &symm_2,
        &symm_3);
    int previous_diag_idx = blockIdx.x;
    int previous_row = this_row;
    //int previous_col = this_col;
    for (int k = 0; k <= 1; ++k){
      unsigned int symm_k;
      // Avoid redundant loops
      if (symm_3 == spl_order) ++k;
      if (k == 0) symm_k = symm_3;
      else symm_k = 2*spl_order - symm_3;
      for (int j = 0; j <= 1; ++j){
        unsigned int symm_j;
        // Avoid redundant loops
        if (symm_2 == spl_order) ++j;
        if (j == 0) symm_j = symm_2;
        else symm_j = 2*spl_order - symm_2;
        for (int i = 0; i <= 1; ++i){
          unsigned int symm_i;
          // Avoid redundant loops
          if (symm_1 == spl_order) ++i;
          if (i == 0) symm_i = symm_1;
          else symm_i = 2*spl_order - symm_1;
          unsigned int inner_diag_idx;
          MMORF::index_vol_to_lin(
              // Input
              symm_i,
              symm_j,
              symm_k,
              2*spl_order + 1,
              2*spl_order + 1,
              2*spl_order + 1,
              // Output
              &inner_diag_idx);
          int diag_diff = jtj_offsets[inner_diag_idx] - jtj_offsets[previous_diag_idx];
          int inner_row = previous_row - diag_diff/2;
          previous_diag_idx = inner_diag_idx;
          previous_row = inner_row;
          // Save the value
          jtj_values[inner_diag_idx*jtj_sz_diag + inner_row] = jtj_val;
        }
      }
    }
  } // kernel_make_jtj_symmetrical

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.y  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.x  = total number of chunks each diagonal is broken into
  /// blockIdx.y = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.x = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x 1D spline kernel in x-direction
  /// \param spl_y 1D spline kernel in y-direction
  /// \param spl_z 1D spline kernel in z-direction
  /// \param spl_ksp_x knot spacing of spline in x-direction
  /// \param spl_ksp_y knot spacing of spline in y-direction
  /// \param spl_ksp_z knot spacing of spline in z-direction
  /// \param param_sz_x size of parameter space in x-direction
  /// \param param_sz_y size of parameter space in y-direction
  /// \param param_sz_z size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x * param_sz_y * param_sz_z;
    const auto offset_into_diag = blockIdx.x*blockDim.x + threadIdx.x;
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];
    // We only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.y];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
      // Find difference in centres to calculate orientation.
      // NB this needs to be a signed operation
      auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
      auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
      auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_x,
          spl_ksp_x,
          spl_order,
          // Output
          &spl1_xind_start,
          &spl1_xind_end,
          &spl2_xind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_y,
          spl_ksp_y,
          spl_order,
          // Output
          &spl1_yind_start,
          &spl1_yind_end,
          &spl2_yind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_z,
          spl_ksp_z,
          spl_order,
          // Output
          &spl1_zind_start,
          &spl1_zind_end,
          &spl2_zind_start);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - diag_first_row;

    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_diag)) return;
    else if (!is_valid_index(this_col, jtj_sz_diag)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z, spl_order + 1)) return;
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + static_cast<int>(k);
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + static_cast<int>(j);
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + static_cast<int>(i);
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x[spl1_xind]
              * shared_spline_y[spl1_yind]
              * shared_spline_z[spl1_zind]
              * shared_spline_x[spl2_xind]
              * shared_spline_y[spl2_yind]
              * shared_spline_z[spl2_zind];
        }
      }
    }
    // Calculate levels of symmetry
    unsigned int symm_1, symm_2, symm_3;
    MMORF::index_lin_to_vol(
        // Input
        blockIdx.y,
        2*spl_order + 1,
        2*spl_order + 1,
        2*spl_order + 1,
        // Output
        &symm_1,
        &symm_2,
        &symm_3);
    int previous_diag_idx = blockIdx.y;
    int previous_row = this_row;
    //int previous_col = this_col;
    for (int k = 0; k <= 1; ++k){
      unsigned int symm_k;
      // Avoid redundant loops
      if (symm_3 == spl_order) ++k;
      if (k == 0) symm_k = symm_3;
      else symm_k = 2*spl_order - symm_3;
      for (int j = 0; j <= 1; ++j){
        unsigned int symm_j;
        // Avoid redundant loops
        if (symm_2 == spl_order) ++j;
        if (j == 0) symm_j = symm_2;
        else symm_j = 2*spl_order - symm_2;
        for (int i = 0; i <= 1; ++i){
          unsigned int symm_i;
          // Avoid redundant loops
          if (symm_1 == spl_order) ++i;
          if (i == 0) symm_i = symm_1;
          else symm_i = 2*spl_order - symm_1;
          unsigned int inner_diag_idx;
          MMORF::index_vol_to_lin(
              // Input
              symm_i,
              symm_j,
              symm_k,
              2*spl_order + 1,
              2*spl_order + 1,
              2*spl_order + 1,
              // Output
              &inner_diag_idx);
          int diag_diff = jtj_offsets[inner_diag_idx] - jtj_offsets[previous_diag_idx];
          int inner_row = previous_row - diag_diff/2;
          previous_diag_idx = inner_diag_idx;
          previous_row = inner_row;
          // Save the value
          //jtj_values[inner_row] += fabsf(jtj_val);
          atomicAdd(&jtj_values[inner_row], fabsf(jtj_val));
        }
      }
    }
  } // kernel_make_jtj_symmetrical_diag_hess

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.y  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.x  = total number of chunks each diagonal is broken into
  /// blockIdx.y = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.x = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x 1D spline kernel in x-direction
  /// \param spl_y 1D spline kernel in y-direction
  /// \param spl_z 1D spline kernel in z-direction
  /// \param spl_ksp_x knot spacing of spline in x-direction
  /// \param spl_ksp_y knot spacing of spline in y-direction
  /// \param spl_ksp_z knot spacing of spline in z-direction
  /// \param param_sz_x size of parameter space in x-direction
  /// \param param_sz_y size of parameter space in y-direction
  /// \param param_sz_z size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values_1 linearised values of sparse diagonal representation of jtj
  ///        submatrix 1
  /// \param jtj_values_2 linearised values of sparse diagonal representation of jtj
  ///        submatrix 2
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values_2)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x * param_sz_y * param_sz_z;
    const auto offset_into_diag = blockIdx.x*blockDim.x + threadIdx.x;
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];
    // We only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.y];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
      // Find difference in centres to calculate orientation.
      // NB this needs to be a signed operation
      auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
      auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
      auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_x,
          spl_ksp_x,
          spl_order,
          // Output
          &spl1_xind_start,
          &spl1_xind_end,
          &spl2_xind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_y,
          spl_ksp_y,
          spl_order,
          // Output
          &spl1_yind_start,
          &spl1_yind_end,
          &spl2_yind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_z,
          spl_ksp_z,
          spl_order,
          // Output
          &spl1_zind_start,
          &spl1_zind_end,
          &spl2_zind_start);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - diag_first_row;

    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_diag)) return;
    else if (!is_valid_index(this_col, jtj_sz_diag)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z, spl_order + 1)) return;
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + static_cast<int>(k);
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + static_cast<int>(j);
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + static_cast<int>(i);
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x[spl1_xind]
              * shared_spline_y[spl1_yind]
              * shared_spline_z[spl1_zind]
              * shared_spline_x[spl2_xind]
              * shared_spline_y[spl2_yind]
              * shared_spline_z[spl2_zind];
        }
      }
    }
    // Calculate levels of symmetry
    unsigned int symm_1, symm_2, symm_3;
    MMORF::index_lin_to_vol(
        // Input
        blockIdx.y,
        2*spl_order + 1,
        2*spl_order + 1,
        2*spl_order + 1,
        // Output
        &symm_1,
        &symm_2,
        &symm_3);
    int previous_diag_idx = blockIdx.y;
    int previous_row = this_row;
    //int previous_col = this_col;
    for (int k = 0; k <= 1; ++k){
      unsigned int symm_k;
      // Avoid redundant loops
      if (symm_3 == spl_order) ++k;
      if (k == 0) symm_k = symm_3;
      else symm_k = 2*spl_order - symm_3;
      for (int j = 0; j <= 1; ++j){
        unsigned int symm_j;
        // Avoid redundant loops
        if (symm_2 == spl_order) ++j;
        if (j == 0) symm_j = symm_2;
        else symm_j = 2*spl_order - symm_2;
        for (int i = 0; i <= 1; ++i){
          unsigned int symm_i;
          // Avoid redundant loops
          if (symm_1 == spl_order) ++i;
          if (i == 0) symm_i = symm_1;
          else symm_i = 2*spl_order - symm_1;
          unsigned int inner_diag_idx;
          MMORF::index_vol_to_lin(
              // Input
              symm_i,
              symm_j,
              symm_k,
              2*spl_order + 1,
              2*spl_order + 1,
              2*spl_order + 1,
              // Output
              &inner_diag_idx);
          int diag_diff = jtj_offsets[inner_diag_idx] - jtj_offsets[previous_diag_idx];
          int inner_row = previous_row - diag_diff/2;
          previous_diag_idx = inner_diag_idx;
          previous_row = inner_row;
          // Save the value
          int inner_col = jtj_offsets[inner_diag_idx] + inner_row;
          atomicAdd(&jtj_values_1[inner_row], fabsf(jtj_val));
          atomicAdd(&jtj_values_2[inner_col], fabsf(jtj_val));
        }
      }
    }
  } // kernel_make_jtj_symmetrical_diag_hess

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.y  = total number of chunks each diagonal is broken into
  /// blockIdx.x = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.y = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x_1 1D spline kernel in x-direction
  /// \param spl_y_1 1D spline kernel in y-direction
  /// \param spl_z_1 1D spline kernel in z-direction
  /// \param spl_x_2 1D spline kernel in x-direction
  /// \param spl_y_2 1D spline kernel in y-direction
  /// \param spl_z_2 1D spline kernel in z-direction
  /// \param spl_ksp_x_1 knot spacing of spline in x-direction
  /// \param spl_ksp_y_1 knot spacing of spline in y-direction
  /// \param spl_ksp_z_1 knot spacing of spline in z-direction
  /// \param spl_ksp_x_2 knot spacing of spline in x-direction
  /// \param spl_ksp_y_2 knot spacing of spline in y-direction
  /// \param spl_ksp_z_2 knot spacing of spline in z-direction
  /// \param param_sz_x_1 size of parameter space in x-direction
  /// \param param_sz_y_1 size of parameter space in y-direction
  /// \param param_sz_z_1 size of parameter space in z-direction
  /// \param param_sz_x_2 size of parameter space in x-direction
  /// \param param_sz_y_2 size of parameter space in y-direction
  /// \param param_sz_z_2 size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x_1 * param_sz_y_1 * param_sz_z_1;
    const auto jtj_sz_row = jtj_sz_diag;
    const auto jtj_sz_col = param_sz_x_2 * param_sz_y_2 * param_sz_z_2;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x_1 = all_splines;
    float *shared_spline_y_1 = &shared_spline_x_1[spl_ksp_x_1 * (spl_order + 1) - 1];
    float *shared_spline_z_1 = &shared_spline_y_1[spl_ksp_y_1 * (spl_order + 1) - 1];
    float *shared_spline_x_2 = &shared_spline_z_1[spl_ksp_z_1 * (spl_order + 1) - 1];
    float *shared_spline_y_2 = &shared_spline_x_2[spl_ksp_x_2 * (spl_order + 1) - 1];
    float *shared_spline_z_2 = &shared_spline_y_2[spl_ksp_y_2 * (spl_order + 1) - 1];

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.x];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x_1,
          param_sz_y_1,
          param_sz_z_1,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x_2,
          param_sz_y_2,
          param_sz_z_2,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_1) - 1; ++i){
        shared_spline_x_1[i*blockDim.x + threadIdx.x] = spl_x_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_1) - 1; ++i){
        shared_spline_y_1[i*blockDim.x + threadIdx.x] = spl_y_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_1) - 1; ++i){
        shared_spline_z_1[i*blockDim.x + threadIdx.x] = spl_z_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_2) - 1; ++i){
        shared_spline_x_2[i*blockDim.x + threadIdx.x] = spl_x_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_2) - 1; ++i){
        shared_spline_y_2[i*blockDim.x + threadIdx.x] = spl_y_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_2) - 1; ++i){
        shared_spline_z_2[i*blockDim.x + threadIdx.x] = spl_z_2[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - static_cast<int>(diag_first_row);
    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_row)) return;
    else if (!is_valid_index(this_col, jtj_sz_col)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x_1,
        param_sz_y_1,
        param_sz_z_1,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x_2,
        param_sz_y_2,
        param_sz_z_2,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    // Find difference in centres to calculate orientation.
    // NB this needs to be a signed operation
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z + spl_order, 2*spl_order + 1)) return;
    // We actually only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;
    // If this is a real point, calculate the overlap
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_x,
        spl_ksp_x_1,
        spl_order,
        // Output
        &spl1_xind_start,
        &spl1_xind_end,
        &spl2_xind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_y,
        spl_ksp_y_1,
        spl_order,
        // Output
        &spl1_yind_start,
        &spl1_yind_end,
        &spl2_yind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_z,
        spl_ksp_z_1,
        spl_order,
        // Output
        &spl1_zind_start,
        &spl1_zind_end,
        &spl2_zind_start);
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x_1[spl1_xind]
              * shared_spline_y_1[spl1_yind]
              * shared_spline_z_1[spl1_zind]
              * shared_spline_x_2[spl2_xind]
              * shared_spline_y_2[spl2_yind]
              * shared_spline_z_2[spl2_zind];
        }
      }
    }
    // Save value
    int diag_idx = blockIdx.x;
    jtj_values[diag_idx*jtj_sz_diag + this_row] += jtj_val;
  } // kernel_make_jtj_symmetrical

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.y  = total number of chunks each diagonal is broken into
  /// blockIdx.x = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.y = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x_1 1D spline kernel in x-direction
  /// \param spl_y_1 1D spline kernel in y-direction
  /// \param spl_z_1 1D spline kernel in z-direction
  /// \param spl_x_2 1D spline kernel in x-direction
  /// \param spl_y_2 1D spline kernel in y-direction
  /// \param spl_z_2 1D spline kernel in z-direction
  /// \param spl_ksp_x_1 knot spacing of spline in x-direction
  /// \param spl_ksp_y_1 knot spacing of spline in y-direction
  /// \param spl_ksp_z_1 knot spacing of spline in z-direction
  /// \param spl_ksp_x_2 knot spacing of spline in x-direction
  /// \param spl_ksp_y_2 knot spacing of spline in y-direction
  /// \param spl_ksp_z_2 knot spacing of spline in z-direction
  /// \param param_sz_x_1 size of parameter space in x-direction
  /// \param param_sz_y_1 size of parameter space in y-direction
  /// \param param_sz_z_1 size of parameter space in z-direction
  /// \param param_sz_x_2 size of parameter space in x-direction
  /// \param param_sz_y_2 size of parameter space in y-direction
  /// \param param_sz_z_2 size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x_1 * param_sz_y_1 * param_sz_z_1;
    const auto jtj_sz_row = jtj_sz_diag;
    const auto jtj_sz_col = param_sz_x_2 * param_sz_y_2 * param_sz_z_2;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x_1 = all_splines;
    float *shared_spline_y_1 = &shared_spline_x_1[spl_ksp_x_1 * (spl_order + 1) - 1];
    float *shared_spline_z_1 = &shared_spline_y_1[spl_ksp_y_1 * (spl_order + 1) - 1];
    float *shared_spline_x_2 = &shared_spline_z_1[spl_ksp_z_1 * (spl_order + 1) - 1];
    float *shared_spline_y_2 = &shared_spline_x_2[spl_ksp_x_2 * (spl_order + 1) - 1];
    float *shared_spline_z_2 = &shared_spline_y_2[spl_ksp_y_2 * (spl_order + 1) - 1];

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.x];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x_1,
          param_sz_y_1,
          param_sz_z_1,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x_2,
          param_sz_y_2,
          param_sz_z_2,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_1) - 1; ++i){
        shared_spline_x_1[i*blockDim.x + threadIdx.x] = spl_x_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_1) - 1; ++i){
        shared_spline_y_1[i*blockDim.x + threadIdx.x] = spl_y_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_1) - 1; ++i){
        shared_spline_z_1[i*blockDim.x + threadIdx.x] = spl_z_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_2) - 1; ++i){
        shared_spline_x_2[i*blockDim.x + threadIdx.x] = spl_x_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_2) - 1; ++i){
        shared_spline_y_2[i*blockDim.x + threadIdx.x] = spl_y_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_2) - 1; ++i){
        shared_spline_z_2[i*blockDim.x + threadIdx.x] = spl_z_2[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - static_cast<int>(diag_first_row);
    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_row)) return;
    else if (!is_valid_index(this_col, jtj_sz_col)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x_1,
        param_sz_y_1,
        param_sz_z_1,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x_2,
        param_sz_y_2,
        param_sz_z_2,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    // Find difference in centres to calculate orientation.
    // NB this needs to be a signed operation
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z + spl_order, 2*spl_order + 1)) return;
    // We actually only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;
    // If this is a real point, calculate the overlap
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_x,
        spl_ksp_x_1,
        spl_order,
        // Output
        &spl1_xind_start,
        &spl1_xind_end,
        &spl2_xind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_y,
        spl_ksp_y_1,
        spl_order,
        // Output
        &spl1_yind_start,
        &spl1_yind_end,
        &spl2_yind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_z,
        spl_ksp_z_1,
        spl_order,
        // Output
        &spl1_zind_start,
        &spl1_zind_end,
        &spl2_zind_start);
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x_1[spl1_xind]
              * shared_spline_y_1[spl1_yind]
              * shared_spline_z_1[spl1_zind]
              * shared_spline_x_2[spl2_xind]
              * shared_spline_y_2[spl2_yind]
              * shared_spline_z_2[spl2_zind];
        }
      }
    }
    // Save value
    atomicAdd(&jtj_values[this_row], fabsf(jtj_val));
  } // kernel_make_jtj_symmetrical_diag_hess

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.y  = total number of chunks each diagonal is broken into
  /// blockIdx.x = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.y = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x_1 1D spline kernel in x-direction
  /// \param spl_y_1 1D spline kernel in y-direction
  /// \param spl_z_1 1D spline kernel in z-direction
  /// \param spl_x_2 1D spline kernel in x-direction
  /// \param spl_y_2 1D spline kernel in y-direction
  /// \param spl_z_2 1D spline kernel in z-direction
  /// \param spl_ksp_x_1 knot spacing of spline in x-direction
  /// \param spl_ksp_y_1 knot spacing of spline in y-direction
  /// \param spl_ksp_z_1 knot spacing of spline in z-direction
  /// \param spl_ksp_x_2 knot spacing of spline in x-direction
  /// \param spl_ksp_y_2 knot spacing of spline in y-direction
  /// \param spl_ksp_z_2 knot spacing of spline in z-direction
  /// \param param_sz_x_1 size of parameter space in x-direction
  /// \param param_sz_y_1 size of parameter space in y-direction
  /// \param param_sz_z_1 size of parameter space in z-direction
  /// \param param_sz_x_2 size of parameter space in x-direction
  /// \param param_sz_y_2 size of parameter space in y-direction
  /// \param param_sz_z_2 size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values_1 linearised values of sparse diagonal representation of jtj
  ///        submatrix 1
  /// \param jtj_values_2 linearised values of sparse diagonal representation of jtj
  ///        submatrix 2
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values_2)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x_1 * param_sz_y_1 * param_sz_z_1;
    const auto jtj_sz_row = jtj_sz_diag;
    const auto jtj_sz_col = param_sz_x_2 * param_sz_y_2 * param_sz_z_2;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x_1 = all_splines;
    float *shared_spline_y_1 = &shared_spline_x_1[spl_ksp_x_1 * (spl_order + 1) - 1];
    float *shared_spline_z_1 = &shared_spline_y_1[spl_ksp_y_1 * (spl_order + 1) - 1];
    float *shared_spline_x_2 = &shared_spline_z_1[spl_ksp_z_1 * (spl_order + 1) - 1];
    float *shared_spline_y_2 = &shared_spline_x_2[spl_ksp_x_2 * (spl_order + 1) - 1];
    float *shared_spline_z_2 = &shared_spline_y_2[spl_ksp_y_2 * (spl_order + 1) - 1];

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.x];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x_1,
          param_sz_y_1,
          param_sz_z_1,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x_2,
          param_sz_y_2,
          param_sz_z_2,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_1) - 1; ++i){
        shared_spline_x_1[i*blockDim.x + threadIdx.x] = spl_x_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_1) - 1; ++i){
        shared_spline_y_1[i*blockDim.x + threadIdx.x] = spl_y_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_1) - 1; ++i){
        shared_spline_z_1[i*blockDim.x + threadIdx.x] = spl_z_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_2) - 1; ++i){
        shared_spline_x_2[i*blockDim.x + threadIdx.x] = spl_x_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_2) - 1; ++i){
        shared_spline_y_2[i*blockDim.x + threadIdx.x] = spl_y_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_2) - 1; ++i){
        shared_spline_z_2[i*blockDim.x + threadIdx.x] = spl_z_2[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - static_cast<int>(diag_first_row);
    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_row)) return;
    else if (!is_valid_index(this_col, jtj_sz_col)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x_1,
        param_sz_y_1,
        param_sz_z_1,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x_2,
        param_sz_y_2,
        param_sz_z_2,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    // Find difference in centres to calculate orientation.
    // NB this needs to be a signed operation
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z + spl_order, 2*spl_order + 1)) return;
    // We actually only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;
    // If this is a real point, calculate the overlap
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_x,
        spl_ksp_x_1,
        spl_order,
        // Output
        &spl1_xind_start,
        &spl1_xind_end,
        &spl2_xind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_y,
        spl_ksp_y_1,
        spl_order,
        // Output
        &spl1_yind_start,
        &spl1_yind_end,
        &spl2_yind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_z,
        spl_ksp_z_1,
        spl_order,
        // Output
        &spl1_zind_start,
        &spl1_zind_end,
        &spl2_zind_start);
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          // The big calc
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x_1[spl1_xind]
              * shared_spline_y_1[spl1_yind]
              * shared_spline_z_1[spl1_zind]
              * shared_spline_x_2[spl2_xind]
              * shared_spline_y_2[spl2_yind]
              * shared_spline_z_2[spl2_zind];
        }
      }
    }
    // Save value
    atomicAdd(&jtj_values_1[this_row], fabsf(jtj_val));
    atomicAdd(&jtj_values_2[this_col], fabsf(jtj_val));
  } // kernel_make_jtj_symmetrical_diag_hess

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This is the most important part of the code so far. We launch a single Kernel
  /// which is responsible for calculating the Hessian. The first iteration of this code was
  /// monolithic and really hard to follow, but I was paranoid about performance. This time I
  /// I am trying to do a much better job of splitting things up into useful functions. Also,
  /// I will try and give the compiler at least a passable shot at optimising by declaring
  /// everything inline!
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.y  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.x  = total number of chunks each diagonal is broken into
  /// blockIdx.y = number of the current diagonal in jtj, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.x = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param ima_sz_x size of image in x-direction
  /// \param ima_sz_y size of image in y-direction
  /// \param ima_sz_z size of image in z-direction
  /// \param ima pre-multiplied image volume
  /// \param spl_x_1 1D spline kernel in x-direction
  /// \param spl_y_1 1D spline kernel in y-direction
  /// \param spl_z_1 1D spline kernel in z-direction
  /// \param spl_x_2 1D spline kernel in x-direction
  /// \param spl_y_2 1D spline kernel in y-direction
  /// \param spl_z_2 1D spline kernel in z-direction
  /// \param spl_ksp_x_1 knot spacing of spline in x-direction
  /// \param spl_ksp_y_1 knot spacing of spline in y-direction
  /// \param spl_ksp_z_1 knot spacing of spline in z-direction
  /// \param spl_ksp_x_2 knot spacing of spline in x-direction
  /// \param spl_ksp_y_2 knot spacing of spline in y-direction
  /// \param spl_ksp_z_2 knot spacing of spline in z-direction
  /// \param param_sz_x_1 size of parameter space in x-direction
  /// \param param_sz_y_1 size of parameter space in y-direction
  /// \param param_sz_z_1 size of parameter space in z-direction
  /// \param param_sz_x_2 size of parameter space in x-direction
  /// \param param_sz_y_2 size of parameter space in y-direction
  /// \param param_sz_z_2 size of parameter space in z-direction
  /// \param jtj_offsets offsets of sparse diagonal representation of jtj
  ///
  /// \param jtj_values linearise values of sparse diagonal representation of jtj
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto jtj_sz_diag = param_sz_x_1 * param_sz_y_1 * param_sz_z_1;
    const auto jtj_sz_row = jtj_sz_diag;
    const auto jtj_sz_col = param_sz_x_2 * param_sz_y_2 * param_sz_z_2;
    const auto offset_into_diag = blockIdx.x*blockDim.x + threadIdx.x;
    float *shared_spline_x_1 = all_splines;
    float *shared_spline_y_1 = &shared_spline_x_1[9*(spl_ksp_x_1 * (spl_order + 1) - 1)];
    float *shared_spline_z_1 = &shared_spline_y_1[9*(spl_ksp_y_1 * (spl_order + 1) - 1)];
    float *shared_spline_x_2 = &shared_spline_z_1[9*(spl_ksp_z_1 * (spl_order + 1) - 1)];
    float *shared_spline_y_2 = &shared_spline_x_2[9*(spl_ksp_x_2 * (spl_order + 1) - 1)];
    float *shared_spline_z_2 = &shared_spline_y_2[9*(spl_ksp_y_2 * (spl_order + 1) - 1)];

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = jtj_offsets[blockIdx.y];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x_1,
          param_sz_y_1,
          param_sz_z_1,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x_2,
          param_sz_y_2,
          param_sz_z_2,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_x_1) - 1); ++i){
        shared_spline_x_1[i*blockDim.x + threadIdx.x] = spl_x_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_y_1) - 1); ++i){
        shared_spline_y_1[i*blockDim.x + threadIdx.x] = spl_y_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_z_1) - 1); ++i){
        shared_spline_z_1[i*blockDim.x + threadIdx.x] = spl_z_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_x_2) - 1); ++i){
        shared_spline_x_2[i*blockDim.x + threadIdx.x] = spl_x_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_y_2) - 1); ++i){
        shared_spline_y_2[i*blockDim.x + threadIdx.x] = spl_y_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < 9*(((spl_order + 1) * spl_ksp_z_2) - 1); ++i){
        shared_spline_z_2[i*blockDim.x + threadIdx.x] = spl_z_2[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - static_cast<int>(diag_first_row);
    // Is this a valid point?
    if (!is_valid_index(this_row, jtj_sz_row)) return;
    else if (!is_valid_index(this_col, jtj_sz_col)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x_1,
        param_sz_y_1,
        param_sz_z_1,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x_2,
        param_sz_y_2,
        param_sz_z_2,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    // Find difference in centres to calculate orientation.
    // NB this needs to be a signed operation
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y + spl_order, 2*spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z + spl_order, 2*spl_order + 1)) return;
    // We actually only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;
    // If this is a real point, calculate the overlap
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_x,
        spl_ksp_x_1,
        spl_order,
        // Output
        &spl1_xind_start,
        &spl1_xind_end,
        &spl2_xind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_y,
        spl_ksp_y_1,
        spl_order,
        // Output
        &spl1_yind_start,
        &spl1_yind_end,
        &spl2_yind_start);
    MMORF::calculate_overlap(
        // Input
        spl_diff_mid_z,
        spl_ksp_z_1,
        spl_order,
        // Output
        &spl1_zind_start,
        &spl1_zind_end,
        &spl2_zind_start);
    // Use spl1 indices to calculate the corresponding indexes into the image volume
    // NOTE: These values may be negative
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_x_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_xind_start); // Deal with area of valid overlap
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_y_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_yind_start); // Deal with area of valid overlap
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z_1) // Centre of spline in volume
      - static_cast<int>(spl_order*spl_ksp_z_1 - 1) // Deal with spline "0" being outside volume
      + static_cast<int>(spl1_zind_start); // Deal with area of valid overlap
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    // The big loop
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          // All indices are valid, therefore calculate a value
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          MMORF::index_vol_to_lin(
              // Input
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              // Output
              &vol_lind);
          for (int sub_jtj = 0
              ; sub_jtj < 9
              ; ++sub_jtj)
          {
            // The big calc
            jtj_val += tex1Dfetch<float>(ima[sub_jtj], vol_lind)
                * shared_spline_x_1[sub_jtj*(spl_ksp_x_1 * (spl_order + 1) - 1) + spl1_xind]
                * shared_spline_y_1[sub_jtj*(spl_ksp_y_1 * (spl_order + 1) - 1) + spl1_yind]
                * shared_spline_z_1[sub_jtj*(spl_ksp_z_1 * (spl_order + 1) - 1) + spl1_zind]
                * shared_spline_x_2[sub_jtj*(spl_ksp_x_2 * (spl_order + 1) - 1) + spl2_xind]
                * shared_spline_y_2[sub_jtj*(spl_ksp_y_2 * (spl_order + 1) - 1) + spl2_yind]
                * shared_spline_z_2[sub_jtj*(spl_ksp_z_2 * (spl_order + 1) - 1) + spl2_zind];
          }
        }
      }
    }
    // Save value
    atomicAdd(&jtj_values[this_row], fabsf(jtj_val));
  } // kernel_make_jtj_symmetrical_diag_hess

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This code calculates the matrix central to computing values for the Bending
  ///          Energy cost function. Note that this code is probably sub-optimal, in terms of
  ///          performing some computations which should really be done on the CPU
  ///
  /// An important aspect of this kernel is the logic regarding block and thread ids. They
  /// work as follows:
  ///
  /// gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
  /// gridDim.y  = total number of chunks each diagonal is broken into
  /// blockIdx.x = number of the current diagonal in bkk, starting at 0 for the lowest
  ///              diagonal present (343 total for cubic spline, therefore main diagonal has
  ///              blockIdx.x == 171)
  /// blockIdx.y = number of the current chunk within the current diagonal
  /// blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
  ///              within each diagonal)
  ///
  /// \param spl_x 1D spline kernel in x-direction
  /// \param spl_y 1D spline kernel in y-direction
  /// \param spl_z 1D spline kernel in z-direction
  /// \param spl_ksp_x knot spacing of spline in x-direction
  /// \param spl_ksp_y knot spacing of spline in y-direction
  /// \param spl_ksp_z knot spacing of spline in z-direction
  /// \param param_sz_x size of parameter space in x-direction
  /// \param param_sz_y size of parameter space in y-direction
  /// \param param_sz_z size of parameter space in z-direction
  /// \param bkk_offsets offsets of sparse diagonal representation of bkk
  /// \param double_differentiated_spline same spline differentiated twice
  ///
  /// \param bkk_values linearise values of sparse diagonal representation of bkk
  ////////////////////////////////////////////////////////////////////////////////////////////
  __global__ void kernel_make_bkk(
      // Input
      const float* __restrict__ spl_x,
      const float* __restrict__ spl_y,
      const float* __restrict__ spl_z,
      const unsigned int spl_ksp_x,
      const unsigned int spl_ksp_y,
      const unsigned int spl_ksp_z,
      const unsigned int param_sz_x,
      const unsigned int param_sz_y,
      const unsigned int param_sz_z,
      const int* __restrict__ bkk_offsets,
      const bool twice_differentiated_spline,
      // Output
      float* __restrict__ bkk_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; // Might replace this with a parameter later
    const auto bkk_sz_diag = param_sz_x * param_sz_y * param_sz_z;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];
    // We only need to calculate the valid spline indices once per diagonal
    // This could potentially be done outside of the kernel in fact, but would require a
    // a fairly major reworking of the logic involved, but still potentially worth it.
    __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

    // Calculate the overlapping regions of each spline. This only needs to be calculated
    // once per block, as all threads within a block are characterised by the same type of
    // overlap (except when the splines reach the end of the image in any dimension)
    // NB!!! Calculating the overlap once per diagonal only works if the diagonal starts
    // with a valid overlap!!! I.E. this will not work when symmetry is does not hold!!!
    if (threadIdx.x == 0){
      // Identify the first valid (row,col) pair for this particular block
      auto this_offset = bkk_offsets[blockIdx.x];
      MMORF::calculate_diagonal_range(
          // Input
          this_offset,
          bkk_sz_diag,
          bkk_sz_diag,
          // Output
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      // Calculate the position of the centre points of the two splines involved in
      // calculating the value at this point in JtJ. Note we assume that the pirst two points
      // in this diagonal are representitive of the overall diagonal
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      MMORF::index_lin_to_vol(
          // Input
          diag_first_row,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      MMORF::index_lin_to_vol(
          // Input
          diag_first_col,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          // Output
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
      // Find difference in centres to calculate orientation.
      // NB this needs to be a signed operation
      auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
      auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
      auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_x,
          spl_ksp_x,
          spl_order,
          // Output
          &spl1_xind_start,
          &spl1_xind_end,
          &spl2_xind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_y,
          spl_ksp_y,
          spl_order,
          // Output
          &spl1_yind_start,
          &spl1_yind_end,
          &spl2_yind_start);
      MMORF::calculate_overlap(
          // Input
          spl_diff_mid_z,
          spl_ksp_z,
          spl_order,
          // Output
          &spl1_zind_start,
          &spl1_zind_end,
          &spl2_zind_start);
    }
    // Wait for thread 0 here
    __syncthreads();
    // Populate shared 1D splines
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    // Wait for all threads here
    __syncthreads();
    // Which point in the hessian is this particular thread calculating?
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - diag_first_row;

    // Is this a valid point?
    if (!is_valid_index(this_row, bkk_sz_diag)) return;
    else if (!is_valid_index(this_col, bkk_sz_diag)) return;
    // Calculate offset between splines
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    MMORF::index_lin_to_vol(
        // Input
        this_row,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    MMORF::index_lin_to_vol(
        // Input
        this_col,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        // Output
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    // !!!NB!!!NB!!!NB!!!
    // Here we take care of sparsity. We only accept those points where spl1 is "right"
    // of spl2 in all directions. I.E. spl_diff_mid_? must be > 0.
    // Additionally, if the difference is greater than the order of the spline then we are at
    // a "wrap" point, and there is no spline overlap.
    // !!!NB!!!NB!!!NB!!!
    if (!MMORF::is_valid_index(spl_diff_mid_x, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_y, spl_order + 1)) return;
    else if (!MMORF::is_valid_index(spl_diff_mid_z, spl_order + 1)) return;
    // Calculate value in JtJ
    // This is done via a nested FOR loop, iterating through volume with x-direction
    // varying fastest
    /*
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    */
    float bkk_val = 0.0;
    // The big loop
    for (auto k = 0
        ; k <= spl1_zind_end
        ; ++k)
    {
      for (auto j = 0
          ; j <= spl1_yind_end
          ; ++j)
      {
        for (auto i = 0
            ; i <= spl1_xind_end
            ; ++i)
        {
          // All indices are valid, therefore calculate a value
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          // The big calc
          bkk_val +=
              shared_spline_x[spl1_xind]
              * shared_spline_y[spl1_yind]
              * shared_spline_z[spl1_zind]
              * shared_spline_x[spl2_xind]
              * shared_spline_y[spl2_yind]
              * shared_spline_z[spl2_zind];
        }
      }
    }
    // Calculate levels of symmetry
    unsigned int symm_1, symm_2, symm_3;
    MMORF::index_lin_to_vol(
        // Input
        blockIdx.x,
        2*spl_order + 1,
        2*spl_order + 1,
        2*spl_order + 1,
        // Output
        &symm_1,
        &symm_2,
        &symm_3);
    int previous_diag_idx = blockIdx.x;
    int previous_row = this_row;
    //int previous_col = this_col;
    for (int k = 0; k <= 1; ++k){
      unsigned int symm_k;
      // Avoid redundant loops
      if (symm_3 == spl_order) ++k;
      if (k == 0) symm_k = symm_3;
      else symm_k = 2*spl_order - symm_3;
      for (int j = 0; j <= 1; ++j){
        unsigned int symm_j;
        // Avoid redundant loops
        if (symm_2 == spl_order) ++j;
        if (j == 0) symm_j = symm_2;
        else symm_j = 2*spl_order - symm_2;
        for (int i = 0; i <= 1; ++i){
          unsigned int symm_i;
          // Avoid redundant loops
          if (symm_1 == spl_order) ++i;
          if (i == 0) symm_i = symm_1;
          else symm_i = 2*spl_order - symm_1;
          unsigned int inner_diag_idx;
          MMORF::index_vol_to_lin(
              // Input
              symm_i,
              symm_j,
              symm_k,
              2*spl_order + 1,
              2*spl_order + 1,
              2*spl_order + 1,
              // Output
              &inner_diag_idx);
          int diag_diff = bkk_offsets[inner_diag_idx] - bkk_offsets[previous_diag_idx];
          int inner_row = previous_row - diag_diff/2;
          previous_diag_idx = inner_diag_idx;
          previous_row = inner_row;
          // Save the value
          if (twice_differentiated_spline){
            bkk_values[inner_diag_idx*bkk_sz_diag + inner_row] += bkk_val;
          }
          else{
            bkk_values[inner_diag_idx*bkk_sz_diag + inner_row] += 2*bkk_val;
          }
        }
      }
    }
  } // kernel_make_bkk_symmetrical
} // CostFxnKernels definitions

//////////////////////////////////////////////////////////////////////////////////////////////
// Log Jacobian Singular Values Kernel Definitions
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating the cost associated with a
  ///          regularisation strategy based on the sum of the square of the log of the
  ///          local Jacobian in 3D. The calculation is very
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param cost well now, this should be obvious by this point
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ cost)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      /* direct_jacobian_cost_3d */
      /*     cost = direct_jacobian_cost_3d(j11,j12,j13,j21,j22,j23,j31,j32,j33) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     15-nov-2018 09:34:20 */
      auto t3 = j11[jacobian_id] * j22[jacobian_id] * j33[jacobian_id];
      auto t4 = j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id];
      auto t5 = j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id];
      auto t6 = j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id];
      auto t7 = j13[jacobian_id] * j21[jacobian_id] * j32[jacobian_id];
      auto t8 = j13[jacobian_id] * j22[jacobian_id] * j31[jacobian_id];
      auto t2 = __logf(((((t3 - t4) - t5) + t6) + t7) - t8);

      auto cost_tmp = t2 * t2 * ((((((t3 - t4) - t5) + t6) + t7) - t8) + 1.0f);
      if (cost_tmp <= 0.0f){
        cost_tmp = 1.0e-20f;
      }
      cost[jacobian_id] = cost_tmp;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating gradient of the cost associated
  ///          with a regularisation strategy based on the sum of the square of the log of the
  ///          local Jacobian in 3D. The calculation is massively
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param d_cost_d_j11 derivitive of the cost at each location with respect to entry
  ///                     j[1][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j12 derivitive of the cost at each location with respect to entry
  ///                     j[1][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j13 derivitive of the cost at each location with respect to entry
  ///                     j[1][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j21 derivitive of the cost at each location with respect to entry
  ///                     j[2][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j22 derivitive of the cost at each location with respect to entry
  ///                     j[2][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j23 derivitive of the cost at each location with respect to entry
  ///                     j[2][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j31 derivitive of the cost at each location with respect to entry
  ///                     j[3][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j32 derivitive of the cost at each location with respect to entry
  ///                     j[3][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j33 derivitive of the cost at each location with respect to entry
  ///                     j[3][3] in the local 3x3 Jacobian matrix
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ d_cost_d_j33)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      /* direct_jacobian_grad_3d */
      /*     grad = direct_jacobian_grad_3d(j11,j12,j13,j21,j22,j23,j31,j32,j33) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     15-nov-2018 09:34:20 */
      auto t3 = j11[jacobian_id] * j22[jacobian_id] * j33[jacobian_id];
      auto t4 = j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id];
      auto t5 = j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id];
      auto t6 = j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id];
      auto t7 = j13[jacobian_id] * j21[jacobian_id] * j32[jacobian_id];
      auto t8 = j13[jacobian_id] * j22[jacobian_id] * j31[jacobian_id];
      auto t9 = ((((t3 - t4) - t5) + t6) + t7) - t8;
      auto t2 = __logf(t9);
      auto t11 = j22[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j32[jacobian_id];
      auto t12 = t2 * t2;
      auto t14 = j21[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j31[jacobian_id];
      auto t15 = 1.0f / t9;
      t9 = (((((t3 - t4) - t5) + t6) + t7) - t8) + 1.0f;
      t3 = j21[jacobian_id] * j32[jacobian_id] - j22[jacobian_id] * j31[jacobian_id];
      t4 = j12[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j32[jacobian_id];
      t5 = j11[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j31[jacobian_id];
      t6 = j11[jacobian_id] * j32[jacobian_id] - j12[jacobian_id] * j31[jacobian_id];
      t7 = j12[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j22[jacobian_id];
      t8 = j11[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j21[jacobian_id];
      auto t30 = j11[jacobian_id] * j22[jacobian_id] - j12[jacobian_id] * j21[jacobian_id];
      d_cost_d_j11[jacobian_id] = t11 * t12 + t2 * t11 * t15 * t9 * 2.0f;
      d_cost_d_j12[jacobian_id] = -t12 * t14 - t2 * t14 * t15 * t9 * 2.0f;
      d_cost_d_j13[jacobian_id] = t12 * t3 + t2 * t15 * t9 * t3 * 2.0f;
      d_cost_d_j21[jacobian_id] = -t12 * t4 - t2 * t15 * t9 * t4 * 2.0f;
      d_cost_d_j22[jacobian_id] = t12 * t5 + t2 * t15 * t9 * t5 * 2.0f;
      d_cost_d_j23[jacobian_id] = -t12 * t6 - t2 * t15 * t9 * t6 * 2.0f;
      d_cost_d_j31[jacobian_id] = t12 * t7 + t2 * t15 * t9 * t7 * 2.0f;
      d_cost_d_j32[jacobian_id] = -t12 * t8 - t2 * t15 * t9 * t8 * 2.0f;
      d_cost_d_j33[jacobian_id] = t12 * t30 + t2 * t15 * t9 * t30 * 2.0f;
    }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating the cost associated with a
  ///          regularisation strategy based on the sum of the square of the log of the
  ///          singular values of the local Jacobian in 3D. The calculation is very
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param cost well now, this should be obvious by this point
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ cost)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      // And now, the horrible bit...
      float t2;
      float t3;
      float t4;
      float t5;
      float t6;
      float t7;
      float t9;
      float t13;
      float t14;
      float t15;
      float t16;
      float t17;
      float t18;

      /* simple_jacobian_cost_3d */
      /*     cost = simple_jacobian_cost_3d(j11,j12[jacobian_id],j13,j21,j22,j23,j31,j32,j33) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     23-oct-2018 12:52:38 */
      t2 = j11[jacobian_id] * j22[jacobian_id] - j12[jacobian_id] * j21[jacobian_id];
      t3 = j11[jacobian_id] * j22[jacobian_id] * j33[jacobian_id];
      t4 = j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id];
      t5 = j13[jacobian_id] * j21[jacobian_id] * j32[jacobian_id];
      t6 = j11[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j21[jacobian_id];
      t7 = ((((t3 + t4) + t5) - j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id]) - j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id]) - j13[jacobian_id] * j22[jacobian_id] *
        j31[jacobian_id];
      t7 = 1.0f / (t7 * t7);
      t9 = j12[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j22[jacobian_id];
      t13 = j11[jacobian_id] * j32[jacobian_id] - j12[jacobian_id] * j31[jacobian_id];
      t14 = j11[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j31[jacobian_id];
      t15 = j12[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j32[jacobian_id];
      t16 = j21[jacobian_id] * j32[jacobian_id] - j22[jacobian_id] * j31[jacobian_id];
      t17 = j21[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j31[jacobian_id];
      t18 = j22[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j32[jacobian_id];

      float cost_tmp =
        ((((((t3 + t4) + t5) - j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id]) - j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id]) - j13[jacobian_id] * j22[jacobian_id] *
        j31[jacobian_id]) + 1.0f) * ((((((((((((((((((t2 * t2 * t7 + t6 * t6 * t7) + t7 *
        (t9 * t9)) + t7 * (t13 * t13)) + t7 * (t14 * t14)) + t7 * (t15 * t15)) + t7 *
        (t16 * t16)) + t7 * (t17 * t17)) + t7 * (t18 * t18)) + j11[jacobian_id] * j11[jacobian_id]) + j12[jacobian_id] *
        j12[jacobian_id]) + j13[jacobian_id] * j13[jacobian_id]) + j21[jacobian_id] * j21[jacobian_id]) + j22[jacobian_id] * j22[jacobian_id]) + j23[jacobian_id] * j23[jacobian_id]) + j31[jacobian_id] * j31[jacobian_id]) + j32[jacobian_id] *
        j32[jacobian_id]) + j33[jacobian_id] * j33[jacobian_id]) - 6.0f) * 0.25f;

      if (cost_tmp <= 0.0f){
        cost_tmp = 1.0e-20f;
      }
      cost[jacobian_id] = cost_tmp;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating gradient of the cost associated
  ///          with a regularisation strategy based on the sum of the square of the log of the
  ///          singular values of the local Jacobian in 3D. The calculation is massively
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param d_cost_d_j11 derivitive of the cost at each location with respect to entry
  ///                     j[1][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j12 derivitive of the cost at each location with respect to entry
  ///                     j[1][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j13 derivitive of the cost at each location with respect to entry
  ///                     j[1][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j21 derivitive of the cost at each location with respect to entry
  ///                     j[2][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j22 derivitive of the cost at each location with respect to entry
  ///                     j[2][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j23 derivitive of the cost at each location with respect to entry
  ///                     j[2][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j31 derivitive of the cost at each location with respect to entry
  ///                     j[3][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j32 derivitive of the cost at each location with respect to entry
  ///                     j[3][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j33 derivitive of the cost at each location with respect to entry
  ///                     j[3][3] in the local 3x3 Jacobian matrix
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ d_cost_d_j33)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      float t2;
      float t3;
      float t4;
      float t5;
      float t6;
      float t8;
      float t9;
      float t10;
      float t7;
      float t11;
      float t13;
      float t16;
      float t18;
      float t19;
      float t21;
      float t23;
      float t24;
      float t25;
      float t26;
      float t28;
      float t30;
      float t33;
      float t35;
      float t37;
      float t40;
      float t43;
      float t46;

      /* simple_jacobian_grad_3d */
      /*     grad = simple_jacobian_grad_3d(j11,j12,j13,j21,j22,j23,j31,j32,j33) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     23-oct-2018 12:54:20 */
      t2 = j22[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j32[jacobian_id];
      t3 = t2 * t2;
      t4 = j11[jacobian_id] * j22[jacobian_id] * j33[jacobian_id];
      t5 = j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id];
      t6 = j13[jacobian_id] * j21[jacobian_id] * j32[jacobian_id];
      t8 = j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id];
      t9 = j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id];
      t10 = j13[jacobian_id] * j22[jacobian_id] * j31[jacobian_id];
      t7 = ((((t4 + t5) + t6) - t8) - t9) - t10;
      t11 = 1.0f / (t7 * t7);
      t13 = j11[jacobian_id] * j22[jacobian_id] - j12[jacobian_id] * j21[jacobian_id];
      t16 = 1.0f / (t7 * t7 * t7);
      //t16 = 1.0f / powf(t7, 3.0f);
      t18 = j11[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j21[jacobian_id];
      t19 = j12[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j22[jacobian_id];
      t21 = j11[jacobian_id] * j32[jacobian_id] - j12[jacobian_id] * j31[jacobian_id];
      t23 = j11[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j31[jacobian_id];
      t24 = j12[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j32[jacobian_id];
      t25 = j21[jacobian_id] * j32[jacobian_id] - j22[jacobian_id] * j31[jacobian_id];
      t26 = j21[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j31[jacobian_id];
      t28 = t13 * t13;
      t30 = t18 * t18;
      t33 = t19 * t19;
      t35 = t21 * t21;
      t37 = t23 * t23;
      t40 = t24 * t24;
      t43 = t25 * t25;
      t46 = t26 * t26;
      t7 = (((((t4 + t5) + t6) - t8) - t9) - t10) + 1.0f;
      t4 = (((((((((((((((((t11 * t28 + t11 * t30) + t11 * t33) + t11 * t35) + t11 *
                         t37) + t11 * t40) + t11 * t43) + t11 * t46) + t3 * t11) +
                    j11[jacobian_id] * j11[jacobian_id]) + j12[jacobian_id] * j12[jacobian_id]) + j13[jacobian_id] * j13[jacobian_id]) + j21[jacobian_id] * j21[jacobian_id]) + j22[jacobian_id] * j22[jacobian_id]) +
               j23[jacobian_id] * j23[jacobian_id]) + j31[jacobian_id] * j31[jacobian_id]) + j32[jacobian_id] * j32[jacobian_id]) + j33[jacobian_id] * j33[jacobian_id]) - 6.0f;
      d_cost_d_j11[jacobian_id] = t2 * t4 * 0.25f - t7 * (((((((((((((j11[jacobian_id] * -2.0f - j22[jacobian_id] * t11 * t13 *
        2.0f) - j23[jacobian_id] * t11 * t18 * 2.0f) - j32[jacobian_id] * t11 * t21 * 2.0f) - j33[jacobian_id] * t11 * t23 *
        2.0f) + t2 * t3 * t16 * 2.0f) + t2 * t16 * t28 * 2.0f) + t2 * t16 * t30 *
        2.0f) + t2 * t16 * t33 * 2.0f) + t2 * t16 * t35 * 2.0f) + t2 * t16 * t37 *
        2.0f) + t2 * t16 * t40 * 2.0f) + t2 * t16 * t43 * 2.0f) + t2 * t16 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j12[jacobian_id] = t26 * t4 * -0.25f + t7 * (((((((((((((j12[jacobian_id] * 2.0f - j21[jacobian_id] * t11 * t13 *
        2.0f) + j23[jacobian_id] * t11 * t19 * 2.0f) - j31[jacobian_id] * t11 * t21 * 2.0f) + j33[jacobian_id] * t11 * t24 *
        2.0f) + t3 * t16 * t26 * 2.0f) + t16 * t26 * t28 * 2.0f) + t16 * t26 * t30 *
        2.0f) + t16 * t26 * t33 * 2.0f) + t16 * t26 * t35 * 2.0f) + t16 * t26 * t37 *
        2.0f) + t16 * t26 * t40 * 2.0f) + t16 * t26 * t43 * 2.0f) + t16 * t26 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j13[jacobian_id] = t25 * t4 * 0.25f - t7 * (((((((((((((j13[jacobian_id] * -2.0f + j21[jacobian_id] * t11 * t18 *
        2.0f) + j22[jacobian_id] * t11 * t19 * 2.0f) + j31[jacobian_id] * t11 * t23 * 2.0f) + j32[jacobian_id] * t11 * t24 *
        2.0f) + t3 * t16 * t25 * 2.0f) + t16 * t25 * t28 * 2.0f) + t16 * t25 * t30 *
        2.0f) + t16 * t25 * t33 * 2.0f) + t16 * t25 * t35 * 2.0f) + t16 * t25 * t37 *
        2.0f) + t16 * t25 * t40 * 2.0f) + t16 * t25 * t43 * 2.0f) + t16 * t25 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j21[jacobian_id] = t24 * t4 * -0.25f + t7 * (((((((((((((j21[jacobian_id] * 2.0f - j12[jacobian_id] * t11 * t13 *
        2.0f) - j13[jacobian_id] * t11 * t18 * 2.0f) + j32[jacobian_id] * t11 * t25 * 2.0f) + j33[jacobian_id] * t11 * t26 *
        2.0f) + t3 * t16 * t24 * 2.0f) + t16 * t24 * t28 * 2.0f) + t16 * t24 * t30 *
        2.0f) + t16 * t24 * t33 * 2.0f) + t16 * t24 * t35 * 2.0f) + t16 * t24 * t37 *
        2.0f) + t16 * t24 * t40 * 2.0f) + t16 * t24 * t43 * 2.0f) + t16 * t24 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j22[jacobian_id] = t23 * t4 * 0.25f - t7 * (((((((((((((j22[jacobian_id] * -2.0f - j11[jacobian_id] * t11 * t13 *
        2.0f) + j13[jacobian_id] * t11 * t19 * 2.0f) - j33[jacobian_id] * t2 * t11 * 2.0f) + j31[jacobian_id] * t11 * t25 *
        2.0f) + t3 * t16 * t23 * 2.0f) + t16 * t23 * t28 * 2.0f) + t16 * t23 * t30 *
        2.0f) + t16 * t23 * t33 * 2.0f) + t16 * t23 * t35 * 2.0f) + t16 * t23 * t37 *
        2.0f) + t16 * t23 * t40 * 2.0f) + t16 * t23 * t43 * 2.0f) + t16 * t23 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j23[jacobian_id] = t21 * t4 * -0.25f + t7 * (((((((((((((j23[jacobian_id] * 2.0f + j11[jacobian_id] * t11 * t18 *
        2.0f) + j12[jacobian_id] * t11 * t19 * 2.0f) - j32[jacobian_id] * t2 * t11 * 2.0f) - j31[jacobian_id] * t11 * t26 *
        2.0f) + t3 * t16 * t21 * 2.0f) + t16 * t21 * t28 * 2.0f) + t16 * t21 * t30 *
        2.0f) + t16 * t21 * t33 * 2.0f) + t16 * t21 * t35 * 2.0f) + t16 * t21 * t37 *
        2.0f) + t16 * t21 * t40 * 2.0f) + t16 * t21 * t43 * 2.0f) + t16 * t21 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j31[jacobian_id] = t19 * t4 * 0.25f - t7 * (((((((((((((j31[jacobian_id] * -2.0f + j12[jacobian_id] * t11 * t21 *
        2.0f) + j13[jacobian_id] * t11 * t23 * 2.0f) + j22[jacobian_id] * t11 * t25 * 2.0f) + j23[jacobian_id] * t11 * t26 *
        2.0f) + t3 * t16 * t19 * 2.0f) + t16 * t19 * t28 * 2.0f) + t16 * t19 * t30 *
        2.0f) + t16 * t19 * t33 * 2.0f) + t16 * t19 * t35 * 2.0f) + t16 * t19 * t37 *
        2.0f) + t16 * t19 * t40 * 2.0f) + t16 * t19 * t43 * 2.0f) + t16 * t19 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j32[jacobian_id] = t18 * t4 * -0.25f + t7 * (((((((((((((j32[jacobian_id] * 2.0f - j23[jacobian_id] * t2 * t11 *
        2.0f) + j11[jacobian_id] * t11 * t21 * 2.0f) - j13[jacobian_id] * t11 * t24 * 2.0f) + j21[jacobian_id] * t11 * t25 *
        2.0f) + t3 * t16 * t18 * 2.0f) + t16 * t18 * t28 * 2.0f) + t16 * t18 * t30 *
        2.0f) + t16 * t18 * t33 * 2.0f) + t16 * t18 * t35 * 2.0f) + t16 * t18 * t37 *
        2.0f) + t16 * t18 * t40 * 2.0f) + t16 * t18 * t43 * 2.0f) + t16 * t18 * t46 *
        2.0f) * 0.25f;
      d_cost_d_j33[jacobian_id] = t13 * t4 * 0.25f - t7 * (((((((((((((j33[jacobian_id] * -2.0f - j22[jacobian_id] * t2 * t11 *
        2.0f) - j11[jacobian_id] * t11 * t23 * 2.0f) - j12[jacobian_id] * t11 * t24 * 2.0f) - j21[jacobian_id] * t11 * t26 *
        2.0f) + t3 * t13 * t16 * 2.0f) + t13 * t16 * t28 * 2.0f) + t13 * t16 * t30 *
        2.0f) + t13 * t16 * t33 * 2.0f) + t13 * t16 * t35 * 2.0f) + t13 * t16 * t37 *
        2.0f) + t13 * t16 * t40 * 2.0f) + t13 * t16 * t43 * 2.0f) + t13 * t16 * t46 *
        2.0f) * 0.25f;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating the cost associated with a
  ///          regularisation strategy based on the sum of the square of the log of the
  ///          singular values of the local Jacobian in 3D. The calculation is very
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner. This function does not rely on the
  ///          approximation used in kernel_log_jacobian_singular_values_cost_3d, and assumes
  ///          that the eigenvalues and eigenvectors of JTJ have been precomputed.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param cost well now, this should be obvious by this point
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ cost)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      // And now, the horrible bit...
      float t2;
      float t3;
      float t4;

      /* jacobian_cost_3d_eigenvector */
      /*     cost = jacobian_cost_3d_eigenvector(j11,j12,j13,j21,j22,j23,j31,j32,j33,u1,u2,u3) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     20-nov-2018 15:42:09 */
      t2 = __logf(u1[jacobian_id]);
      t3 = __logf(u2[jacobian_id]);
      t4 = __logf(u3[jacobian_id]);
      auto cost_tmp = ((t2 * t2 * 0.25f + t3 * t3 * 0.25f) + t4 * t4 * 0.25f) * ((((((j11[jacobian_id] *
        j22[jacobian_id] * j33[jacobian_id] - j11[jacobian_id] * j23[jacobian_id] * j32[jacobian_id]) - j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id]) + j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id]) + j13[jacobian_id] *
        j21[jacobian_id] * j32[jacobian_id]) - j13[jacobian_id] * j22[jacobian_id] * j31[jacobian_id]) + 1.0f);

      if (cost_tmp <= 0.0f){
        cost_tmp = 1.0e-20f;
      }
      cost[jacobian_id] = cost_tmp;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \details This function is responsible for calculating gradient of the cost associated
  ///          with a regularisation strategy based on the sum of the square of the log of the
  ///          singular values of the local Jacobian in 3D. The calculation is massively
  ///          complicated, but as it is an analytical solution it is faster than any other
  ///          way of determining it. Apart from this, the logic of the calculation itself is
  ///          very straightforward, with each thread accessing the necessary parameters in
  ///          what should be a perfectly coalesced manner, and writing the results in a
  ///          similarly perfectly coalesced manner.
  ///
  /// \param n_samples number of points being sampled
  /// \param j11 entry j[1][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j12 entry j[1][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j13 entry j[1][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j21 entry j[2][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j22 entry j[2][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j23 entry j[2][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j31 entry j[3][1] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j32 entry j[3][2] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  /// \param j33 entry j[3][3] (one-indexed (yes slightly confusing but I did this in Matlab
  ///            first)) in the local 3x3 Jacobian matrix
  ///
  /// \param d_cost_d_j11 derivitive of the cost at each location with respect to entry
  ///                     j[1][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j12 derivitive of the cost at each location with respect to entry
  ///                     j[1][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j13 derivitive of the cost at each location with respect to entry
  ///                     j[1][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j21 derivitive of the cost at each location with respect to entry
  ///                     j[2][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j22 derivitive of the cost at each location with respect to entry
  ///                     j[2][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j23 derivitive of the cost at each location with respect to entry
  ///                     j[2][3] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j31 derivitive of the cost at each location with respect to entry
  ///                     j[3][1] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j32 derivitive of the cost at each location with respect to entry
  ///                     j[3][2] in the local 3x3 Jacobian matrix
  /// \param d_cost_d_j33 derivitive of the cost at each location with respect to entry
  ///                     j[3][3] in the local 3x3 Jacobian matrix
  ////////////////////////////////////////////////////////////////////////////////////////////
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
      float* __restrict__ d_cost_d_j33)
  {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      float t5;
      float t9;
      float t10;
      float t11;
      float t12;
      float t13;
      float t14;
      float t21;
      float t25;
      float t29;
      float t33;
      float t40;
      float t47;

      /* jacobian_grad_3d_eigenvector */
      /*     grad = jacobian_grad_3d_eigenvector(j11,j12,j13,j21,j22,j23,j31,j32,j33,u1,u2,u3,v11,v12,v13,v21,v22,v23,v31,v32,v33) */
      /*     this function was generated by the symbolic math toolbox version 8.1. */
      /*     20-nov-2018 15:42:10 */
      t5 = (((((j11[jacobian_id] * j22[jacobian_id] * j33[jacobian_id] + j12[jacobian_id] * j23[jacobian_id] * j31[jacobian_id]) + j13[jacobian_id] * j21[jacobian_id] * j32[jacobian_id]) - j11[jacobian_id] * j23[jacobian_id] *
              j32[jacobian_id]) - j12[jacobian_id] * j21[jacobian_id] * j33[jacobian_id]) - j13[jacobian_id] * j22[jacobian_id] * j31[jacobian_id]) + 1.0f;
      t9 = __logf(u1[jacobian_id]);
      t10 = __logf(u2[jacobian_id]);
      t11 = __logf(u3[jacobian_id]);
      t12 = 1.0f / u1[jacobian_id];
      t13 = 1.0f / u2[jacobian_id];
      t14 = 1.0f / u3[jacobian_id];
      t21 = (t9 * t9 * 0.25f + t10 * t10 * 0.25f) + t11 * t11 * 0.25f;
      t25 = (v11[jacobian_id] * v12[jacobian_id] * t5 * t9 * t12 * 0.5f + v21[jacobian_id] * v22[jacobian_id] * t5 * t10 * t13 * 0.5f) +
        v31[jacobian_id] * v32[jacobian_id] * t5 * t11 * t14 * 0.5f;
      t29 = (v11[jacobian_id] * v13[jacobian_id] * t5 * t9 * t12 * 0.5f + v21[jacobian_id] * v23[jacobian_id] * t5 * t10 * t13 * 0.5f) +
        v31[jacobian_id] * v33[jacobian_id] * t5 * t11 * t14 * 0.5f;
      t33 = (v12[jacobian_id] * v13[jacobian_id] * t5 * t9 * t12 * 0.5f + v22[jacobian_id] * v23[jacobian_id] * t5 * t10 * t13 * 0.5f) +
        v32[jacobian_id] * v33[jacobian_id] * t5 * t11 * t14 * 0.5f;
      t40 = (t5 * t9 * t12 * (v11[jacobian_id] * v11[jacobian_id]) * 0.5f + t5 * t10 * t13 * (v21[jacobian_id] * v21[jacobian_id]) *
             0.5f) + t5 * t11 * t14 * (v31[jacobian_id] * v31[jacobian_id]) * 0.5f;
      t47 = (t5 * t9 * t12 * (v12[jacobian_id] * v12[jacobian_id]) * 0.5f + t5 * t10 * t13 * (v22[jacobian_id] * v22[jacobian_id]) *
             0.5f) + t5 * t11 * t14 * (v32[jacobian_id] * v32[jacobian_id]) * 0.5f;
      t5 = (t5 * t9 * t12 * (v13[jacobian_id] * v13[jacobian_id]) * 0.5f + t5 * t10 * t13 * (v23[jacobian_id] * v23[jacobian_id]) * 0.5f)
        + t5 * t11 * t14 * (v33[jacobian_id] * v33[jacobian_id]) * 0.5f;
      d_cost_d_j11[jacobian_id] = ((t21 * (j22[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j32[jacobian_id]) + j12[jacobian_id] * t25 * 2.0f) + j13[jacobian_id] * t29 *
                 2.0f) + j11[jacobian_id] * t40 * 2.0f;
      d_cost_d_j12[jacobian_id] = ((-t21 * (j21[jacobian_id] * j33[jacobian_id] - j23[jacobian_id] * j31[jacobian_id]) + j11[jacobian_id] * t25 * 2.0f) + j13[jacobian_id] * t33 *
                 2.0f) + j12[jacobian_id] * t47 * 2.0f;
      d_cost_d_j13[jacobian_id] = ((t21 * (j21[jacobian_id] * j32[jacobian_id] - j22[jacobian_id] * j31[jacobian_id]) + j11[jacobian_id] * t29 * 2.0f) + j12[jacobian_id] * t33 *
                 2.0f) + j13[jacobian_id] * t5 * 2.0f;
      d_cost_d_j21[jacobian_id] = ((-t21 * (j12[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j32[jacobian_id]) + j22[jacobian_id] * t25 * 2.0f) + j23[jacobian_id] * t29 *
                 2.0f) + j21[jacobian_id] * t40 * 2.0f;
      d_cost_d_j22[jacobian_id] = ((t21 * (j11[jacobian_id] * j33[jacobian_id] - j13[jacobian_id] * j31[jacobian_id]) + j21[jacobian_id] * t25 * 2.0f) + j23[jacobian_id] * t33 *
                 2.0f) + j22[jacobian_id] * t47 * 2.0f;
      d_cost_d_j23[jacobian_id] = ((-t21 * (j11[jacobian_id] * j32[jacobian_id] - j12[jacobian_id] * j31[jacobian_id]) + j21[jacobian_id] * t29 * 2.0f) + j22[jacobian_id] * t33 *
                 2.0f) + j23[jacobian_id] * t5 * 2.0f;
      d_cost_d_j31[jacobian_id] = ((t21 * (j12[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j22[jacobian_id]) + j32[jacobian_id] * t25 * 2.0f) + j33[jacobian_id] * t29 *
                 2.0f) + j31[jacobian_id] * t40 * 2.0f;
      d_cost_d_j32[jacobian_id] = ((-t21 * (j11[jacobian_id] * j23[jacobian_id] - j13[jacobian_id] * j21[jacobian_id]) + j31[jacobian_id] * t25 * 2.0f) + j33[jacobian_id] * t33 *
                 2.0f) + j32[jacobian_id] * t47 * 2.0f;
      d_cost_d_j33[jacobian_id] = ((t21 * (j11[jacobian_id] * j22[jacobian_id] - j12[jacobian_id] * j21[jacobian_id]) + j31[jacobian_id] * t29 * 2.0f) + j32[jacobian_id] * t33 *
                 2.0f) + j33[jacobian_id] * t5 * 2.0f;
    }
  }
} // namespace MMORF

//////////////////////////////////////////////////////////////////////////////////////////////
// Diffusion Tensor Registration Kernel Definitions
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MMORF
{
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
      float* __restrict__ d_g11_d_j11,
      float* __restrict__ d_g11_d_j12,
      float* __restrict__ d_g11_d_j13,
      float* __restrict__ d_g11_d_j21,
      float* __restrict__ d_g11_d_j22,
      float* __restrict__ d_g11_d_j23,
      float* __restrict__ d_g11_d_j31,
      float* __restrict__ d_g11_d_j32,
      float* __restrict__ d_g11_d_j33,
      float* __restrict__ d_g12_d_j11,
      float* __restrict__ d_g12_d_j12,
      float* __restrict__ d_g12_d_j13,
      float* __restrict__ d_g12_d_j21,
      float* __restrict__ d_g12_d_j22,
      float* __restrict__ d_g12_d_j23,
      float* __restrict__ d_g12_d_j31,
      float* __restrict__ d_g12_d_j32,
      float* __restrict__ d_g12_d_j33,
      float* __restrict__ d_g13_d_j11,
      float* __restrict__ d_g13_d_j12,
      float* __restrict__ d_g13_d_j13,
      float* __restrict__ d_g13_d_j21,
      float* __restrict__ d_g13_d_j22,
      float* __restrict__ d_g13_d_j23,
      float* __restrict__ d_g13_d_j31,
      float* __restrict__ d_g13_d_j32,
      float* __restrict__ d_g13_d_j33,
      float* __restrict__ d_g22_d_j11,
      float* __restrict__ d_g22_d_j12,
      float* __restrict__ d_g22_d_j13,
      float* __restrict__ d_g22_d_j21,
      float* __restrict__ d_g22_d_j22,
      float* __restrict__ d_g22_d_j23,
      float* __restrict__ d_g22_d_j31,
      float* __restrict__ d_g22_d_j32,
      float* __restrict__ d_g22_d_j33,
      float* __restrict__ d_g23_d_j11,
      float* __restrict__ d_g23_d_j12,
      float* __restrict__ d_g23_d_j13,
      float* __restrict__ d_g23_d_j21,
      float* __restrict__ d_g23_d_j22,
      float* __restrict__ d_g23_d_j23,
      float* __restrict__ d_g23_d_j31,
      float* __restrict__ d_g23_d_j32,
      float* __restrict__ d_g23_d_j33,
      float* __restrict__ d_g33_d_j11,
      float* __restrict__ d_g33_d_j12,
      float* __restrict__ d_g33_d_j13,
      float* __restrict__ d_g33_d_j21,
      float* __restrict__ d_g33_d_j22,
      float* __restrict__ d_g33_d_j23,
      float* __restrict__ d_g33_d_j31,
      float* __restrict__ d_g33_d_j32,
      float* __restrict__ d_g33_d_j33)
      {
    auto jacobian_id = blockIdx.x*blockDim.x + threadIdx.x;
    if (jacobian_id < n_samples){
      float t2;
      float t3;
      float t4;
      float t5;
      float t6;
      float t7;
      float t57;
      float t87;
      float t16;
      float t25;
      float t26;
      float t30;
      float t34;
      float t38;
      float t39;
      float t43;
      float t48;
      float t69;
      float t74;
      float t75;
      float t97;
      float t107;
      float t111;
      float t113;
      float t117_tmp;
      float b_t117_tmp;
      float c_t117_tmp;
      float t117;
      float t119;
      float t121;
      float t426;
      float t125_tmp;
      float b_t125_tmp;
      float t125;
      float t127;
      float t129;
      float t131;
      float t137;
      float t143;
      float t148;
      float t150;
      float t152;
      float t153;
      float t155;
      float t157;
      float t158;
      float t159;
      float t253;
      float t160_tmp;
      float t162;
      float t163;
      float t254;
      float t164_tmp;
      float t165;
      float t259;
      float t166_tmp;
      float t168;
      float t169;
      float t260;
      float t170_tmp;
      float t171;
      float t265;
      float t172_tmp;
      float t174;
      float t175;
      float t266;
      float t176_tmp;
      float t182;
      float t184;
      float t190;
      float t195;
      float t199;
      float t204;
      float t206;
      float t210;
      float t215;
      float t216;
      float t222;
      float t224;
      float t228;
      float t227_tmp;
      float t230;
      float t232;
      float t233;
      float t237;
      float t236_tmp;
      float t239;
      float t241;
      float t242;
      float t246;
      float t245_tmp;
      float t248;
      float t250;
      float t251;
      float t255;
      float t257;
      float t261;
      float t263;
      float t267;
      float t269;
      float t275;
      float t280;
      float t285;
      float t294_tmp;
      float t296_tmp;
      float t298_tmp;
      float t302;
      float t304;
      float t309;
      float t311;
      float t315;
      float t320;
      float t321;
      float t324;
      float t326;
      float t328;
      float t330;
      float t333;
      float t335;
      float t338;
      float t340;
      float t345;
      float t349;
      float t353;
      float t356;
      float t358;
      float t363;
      float t365;
      float t370;
      float t372;
      float t376;
      float t378;
      float t383;
      float t385;
      float t390;
      float t409;

      /* DG_DJ */
      /*     OUT1 = DG_DJ(g11[jacobian_id],g12[jacobian_id],g13[jacobian_id],g22[jacobian_id],g23[jacobian_id],g33[jacobian_id],r11[jacobian_id],r12[jacobian_id],r13[jacobian_id],r21[jacobian_id],r22[jacobian_id],r23[jacobian_id],r31[jacobian_id],r32[jacobian_id],r33[jacobian_id],S11,s12[jacobian_id],s13[jacobian_id],s22[jacobian_id],s23[jacobian_id],s33[jacobian_id]) */
      /*     This function was generated by the Symbolic Math Toolbox version 8.2. */
      /*     27-Aug-2019 15:03:20 */
      t2 = s11[jacobian_id] * s11[jacobian_id];
      t3 = s12[jacobian_id] * s12[jacobian_id];
      t4 = s23[jacobian_id] * s23[jacobian_id];
      t5 = s13[jacobian_id] * s13[jacobian_id];
      t6 = s33[jacobian_id] * s33[jacobian_id];
      t7 = s22[jacobian_id] * s22[jacobian_id];
      t57 = s11[jacobian_id] * s22[jacobian_id];
      t87 = s12[jacobian_id] * s13[jacobian_id];
      t16 = 1.0f / (((((((((((((s11[jacobian_id] * t3 + s11[jacobian_id] * t5) + s22[jacobian_id] * t3) + s22[jacobian_id] * t4) + s33[jacobian_id] *
        t5) + s33[jacobian_id] * t4) + t87 * s23[jacobian_id] * 2.0f) - s11[jacobian_id] * t7) - s22[jacobian_id] * t2) - s11[jacobian_id] * t6) -
                       s33[jacobian_id] * t2) - s22[jacobian_id] * t6) - s33[jacobian_id] * t7) - t57 * s33[jacobian_id] * 2.0f);
      t25 = s11[jacobian_id] * s33[jacobian_id];
      t26 = s22[jacobian_id] * s33[jacobian_id];
      t30 = (s11[jacobian_id] * s12[jacobian_id] + s12[jacobian_id] * s22[jacobian_id]) + s13[jacobian_id] * s23[jacobian_id];
      t34 = (s11[jacobian_id] * s13[jacobian_id] + s12[jacobian_id] * s23[jacobian_id]) + s13[jacobian_id] * s33[jacobian_id];
      t38 = (t87 + s22[jacobian_id] * s23[jacobian_id]) + s23[jacobian_id] * s33[jacobian_id];
      t39 = (((t2 - t4) + t57) + t25) + t26;
      t87 = r21[jacobian_id] * t16;
      t2 = r11[jacobian_id] * t16;
      t4 = r31[jacobian_id] * t16;
      t43 = (t2 * t39 + t87 * t30) + t4 * t34;
      t7 = (((-t5 + t7) + t57) + t25) + t26;
      t48 = (t87 * t7 + t2 * t30) + t4 * t38;
      t5 = (((-t3 + t6) + t57) + t25) + t26;
      t26 = (t4 * t5 + t2 * t34) + t87 * t38;
      t87 = r22[jacobian_id] * t16;
      t2 = r12[jacobian_id] * t16;
      t4 = r32[jacobian_id] * t16;
      t57 = (t2 * t39 + t87 * t30) + t4 * t34;
      t25 = (t87 * t7 + t2 * t30) + t4 * t38;
      t3 = (t4 * t5 + t2 * t34) + t87 * t38;
      t69 = (r12[jacobian_id] * t57 + r22[jacobian_id] * t25) + r32[jacobian_id] * t3;
      t74 = (r13[jacobian_id] * t57 + r23[jacobian_id] * t25) + r33[jacobian_id] * t3;
      t75 = r13[jacobian_id] * t69 - r12[jacobian_id] * t74;
      t2 = r23[jacobian_id] * t16;
      t4 = r13[jacobian_id] * t16;
      t87 = r33[jacobian_id] * t16;
      t6 = (t4 * t39 + t2 * t30) + t87 * t34;
      t7 = (t2 * t7 + t4 * t30) + t87 * t38;
      t87 = (t87 * t5 + t4 * t34) + t2 * t38;
      t39 = (r12[jacobian_id] * t43 + r22[jacobian_id] * t48) + r32[jacobian_id] * t26;
      t2 = (r13[jacobian_id] * t43 + r23[jacobian_id] * t48) + r33[jacobian_id] * t26;
      t97 = r13[jacobian_id] * t39 - r12[jacobian_id] * t2;
      t16 = (r12[jacobian_id] * t6 + r22[jacobian_id] * t7) + r32[jacobian_id] * t87;
      t4 = (r13[jacobian_id] * t6 + r23[jacobian_id] * t7) + r33[jacobian_id] * t87;
      t107 = r13[jacobian_id] * t16 - r12[jacobian_id] * t4;
      t30 = g13[jacobian_id] * r11[jacobian_id];
      t38 = g23[jacobian_id] * r12[jacobian_id];
      t34 = g33[jacobian_id] * r13[jacobian_id];
      t111 = (t30 * 2.0f + t38 * 2.0f) + t34 * 2.0f;
      t113 = r23[jacobian_id] * t69 - r22[jacobian_id] * t74;
      t117_tmp = g11[jacobian_id] * r11[jacobian_id];
      b_t117_tmp = g12[jacobian_id] * r12[jacobian_id];
      c_t117_tmp = g13[jacobian_id] * r13[jacobian_id];
      t117 = (t117_tmp * 2.0f + b_t117_tmp * 2.0f) + c_t117_tmp * 2.0f;
      t119 = r23[jacobian_id] * t39 - r22[jacobian_id] * t2;
      t121 = r23[jacobian_id] * t16 - r22[jacobian_id] * t4;
      t426 = g12[jacobian_id] * r11[jacobian_id];
      t125_tmp = g22[jacobian_id] * r12[jacobian_id];
      b_t125_tmp = g23[jacobian_id] * r13[jacobian_id];
      t125 = (t426 * 2.0f + t125_tmp * 2.0f) + b_t125_tmp * 2.0f;
      t127 = r33[jacobian_id] * t69 - r32[jacobian_id] * t74;
      t129 = r33[jacobian_id] * t39 - r32[jacobian_id] * t2;
      t131 = r33[jacobian_id] * t16 - r32[jacobian_id] * t4;
      t3 = (r11[jacobian_id] * t57 + r21[jacobian_id] * t25) + r31[jacobian_id] * t3;
      t137 = r11[jacobian_id] * t74 - r13[jacobian_id] * t3;
      t5 = (r11[jacobian_id] * t43 + r21[jacobian_id] * t48) + r31[jacobian_id] * t26;
      t143 = r11[jacobian_id] * t2 - r13[jacobian_id] * t5;
      t7 = (r11[jacobian_id] * t6 + r21[jacobian_id] * t7) + r31[jacobian_id] * t87;
      t148 = r11[jacobian_id] * t4;
      t150 = r21[jacobian_id] * t74 - r23[jacobian_id] * t3;
      t152 = r21[jacobian_id] * t2 - r23[jacobian_id] * t5;
      t153 = r21[jacobian_id] * t4;
      t155 = r31[jacobian_id] * t74 - r33[jacobian_id] * t3;
      t157 = r31[jacobian_id] * t2 - r33[jacobian_id] * t5;
      t158 = r31[jacobian_id] * t4;
      t159 = r11[jacobian_id] * t69;
      t253 = r12[jacobian_id] * t3;
      t160_tmp = t159 - t253;
      t162 = r11[jacobian_id] * t39 - r12[jacobian_id] * t5;
      t163 = r11[jacobian_id] * t16;
      t254 = r12[jacobian_id] * t7;
      t164_tmp = t163 - t254;
      t165 = r21[jacobian_id] * t69;
      t259 = r22[jacobian_id] * t3;
      t166_tmp = t165 - t259;
      t168 = r21[jacobian_id] * t39 - r22[jacobian_id] * t5;
      t169 = r21[jacobian_id] * t16;
      t260 = r22[jacobian_id] * t7;
      t170_tmp = t169 - t260;
      t171 = r31[jacobian_id] * t69;
      t265 = r32[jacobian_id] * t3;
      t172_tmp = t171 - t265;
      t174 = r31[jacobian_id] * t39 - r32[jacobian_id] * t5;
      t175 = r31[jacobian_id] * t16;
      t266 = r32[jacobian_id] * t7;
      t176_tmp = t175 - t266;
      t182 = r13[jacobian_id] * t75 - r12[jacobian_id] * t107;
      t184 = r13[jacobian_id] * t97 - r11[jacobian_id] * t107;
      t190 = (t30 + t38) + t34;
      t195 = (t117_tmp + b_t117_tmp) + c_t117_tmp;
      t199 = (t426 + t125_tmp) + b_t125_tmp;
      t2 = g13[jacobian_id] * r21[jacobian_id];
      t4 = g23[jacobian_id] * r22[jacobian_id];
      t5 = g33[jacobian_id] * r23[jacobian_id];
      t204 = (t2 + t4) + t5;
      t206 = r13[jacobian_id] * t113 - r12[jacobian_id] * t121;
      t3 = g11[jacobian_id] * r21[jacobian_id];
      t6 = g12[jacobian_id] * r22[jacobian_id];
      t25 = g13[jacobian_id] * r23[jacobian_id];
      t210 = (t3 + t6) + t25;
      t26 = g12[jacobian_id] * r21[jacobian_id];
      t39 = g22[jacobian_id] * r22[jacobian_id];
      t16 = g23[jacobian_id] * r23[jacobian_id];
      t215 = (t26 + t39) + t16;
      t216 = r11[jacobian_id] * t121 - r13[jacobian_id] * t119;
      t222 = r13[jacobian_id] * t127 - r12[jacobian_id] * t131;
      t224 = r11[jacobian_id] * t131 - r13[jacobian_id] * t129;
      t228 = r13[jacobian_id] * t7;
      t227_tmp = t148 - t228;
      t230 = r11[jacobian_id] * t137 - r12[jacobian_id] * t143;
      t232 = r13[jacobian_id] * t137 - r12[jacobian_id] * t227_tmp;
      t233 = r13[jacobian_id] * t143;
      t237 = r23[jacobian_id] * t7;
      t236_tmp = t153 - t237;
      t239 = r11[jacobian_id] * t150 - r12[jacobian_id] * t152;
      t241 = r13[jacobian_id] * t150 - r12[jacobian_id] * t236_tmp;
      t242 = r13[jacobian_id] * t152;
      t246 = r33[jacobian_id] * t7;
      t245_tmp = t158 - t246;
      t248 = r11[jacobian_id] * t155 - r12[jacobian_id] * t157;
      t250 = r13[jacobian_id] * t155 - r12[jacobian_id] * t245_tmp;
      t251 = r13[jacobian_id] * t157;
      t255 = r13[jacobian_id] * t160_tmp;
      t257 = r11[jacobian_id] * t164_tmp - r13[jacobian_id] * t162;
      t261 = r13[jacobian_id] * t166_tmp;
      t263 = r11[jacobian_id] * t170_tmp - r13[jacobian_id] * t168;
      t267 = r13[jacobian_id] * t172_tmp;
      t269 = r11[jacobian_id] * t176_tmp - r13[jacobian_id] * t174;
      t7 = g13[jacobian_id] * r31[jacobian_id];
      t30 = g23[jacobian_id] * r32[jacobian_id];
      t38 = g33[jacobian_id] * r33[jacobian_id];
      t275 = (t7 + t30) + t38;
      t34 = g11[jacobian_id] * r31[jacobian_id];
      t87 = g12[jacobian_id] * r32[jacobian_id];
      t57 = g13[jacobian_id] * r33[jacobian_id];
      t280 = (t34 + t87) + t57;
      t74 = g12[jacobian_id] * r31[jacobian_id];
      t48 = g22[jacobian_id] * r32[jacobian_id];
      t43 = g23[jacobian_id] * r33[jacobian_id];
      t285 = (t74 + t48) + t43;
      t294_tmp = r12[jacobian_id] * t162;
      t296_tmp = r12[jacobian_id] * t168;
      t298_tmp = r12[jacobian_id] * t174;
      t302 = r23[jacobian_id] * t75 - r22[jacobian_id] * t107;
      t304 = r23[jacobian_id] * t97 - r21[jacobian_id] * t107;
      t309 = (t2 * 2.0f + t4 * 2.0f) + t5 * 2.0f;
      t311 = r23[jacobian_id] * t113 - r22[jacobian_id] * t121;
      t315 = (t3 * 2.0f + t6 * 2.0f) + t25 * 2.0f;
      t320 = (t26 * 2.0f + t39 * 2.0f) + t16 * 2.0f;
      t321 = r21[jacobian_id] * t121 - r23[jacobian_id] * t119;
      t324 = r23[jacobian_id] * t127 - r22[jacobian_id] * t131;
      t326 = r21[jacobian_id] * t131 - r23[jacobian_id] * t129;
      t328 = r21[jacobian_id] * t137 - r22[jacobian_id] * t143;
      t330 = r23[jacobian_id] * t137 - r22[jacobian_id] * t227_tmp;
      t333 = r21[jacobian_id] * t150 - r22[jacobian_id] * t152;
      t335 = r23[jacobian_id] * t150 - r22[jacobian_id] * t236_tmp;
      t338 = r21[jacobian_id] * t155 - r22[jacobian_id] * t157;
      t340 = r23[jacobian_id] * t155 - r22[jacobian_id] * t245_tmp;
      t345 = r21[jacobian_id] * t164_tmp - r23[jacobian_id] * t162;
      t349 = r21[jacobian_id] * t170_tmp - r23[jacobian_id] * t168;
      t353 = r21[jacobian_id] * t176_tmp - r23[jacobian_id] * t174;
      t356 = r33[jacobian_id] * t75 - r32[jacobian_id] * t107;
      t358 = r33[jacobian_id] * t97 - r31[jacobian_id] * t107;
      t363 = r33[jacobian_id] * t113 - r32[jacobian_id] * t121;
      t365 = r31[jacobian_id] * t121 - r33[jacobian_id] * t119;
      t370 = r33[jacobian_id] * t127 - r32[jacobian_id] * t131;
      t372 = r31[jacobian_id] * t131 - r33[jacobian_id] * t129;
      t376 = r31[jacobian_id] * t137 - r32[jacobian_id] * t143;
      t378 = r33[jacobian_id] * t137 - r32[jacobian_id] * t227_tmp;
      t383 = r31[jacobian_id] * t150 - r32[jacobian_id] * t152;
      t385 = r33[jacobian_id] * t150 - r32[jacobian_id] * t236_tmp;
      t390 = r31[jacobian_id] * t155 - r32[jacobian_id] * t157;
      t121 = r33[jacobian_id] * t155 - r32[jacobian_id] * t245_tmp;
      t137 = r31[jacobian_id] * t164_tmp - r33[jacobian_id] * t162;
      t131 = r31[jacobian_id] * t170_tmp - r33[jacobian_id] * t168;
      t409 = r31[jacobian_id] * t176_tmp - r33[jacobian_id] * t174;
      t155 = (t7 * 2.0f + t30 * 2.0f) + t38 * 2.0f;
      t107 = (t34 * 2.0f + t87 * 2.0f) + t57 * 2.0f;
      t426 = (t74 * 2.0f + t48 * 2.0f) + t43 * 2.0f;
      t125_tmp = r11[jacobian_id] * t75 - r12[jacobian_id] * t97;
      d_g11_d_j11[jacobian_id] = (-t117 * t182 + t125 * t184) + t111 * t125_tmp;
      b_t125_tmp = r11[jacobian_id] * t113 - r12[jacobian_id] * t119;
      d_g11_d_j12[jacobian_id] = (-t117 * t206 - t125 * t216) + t111 * b_t125_tmp;
      t150 = r11[jacobian_id] * t127 - r12[jacobian_id] * t129;
      d_g11_d_j13[jacobian_id] = (-t117 * t222 - t125 * t224) + t111 * t150;
      d_g11_d_j21[jacobian_id] = (t111 * t230 - t117 * t232) + t125 * (t233 - r11[jacobian_id] * t227_tmp);
      d_g11_d_j22[jacobian_id] = (t111 * t239 - t117 * t241) + t125 * (t242 - r11[jacobian_id] * t236_tmp);
      d_g11_d_j23[jacobian_id] = (t111 * t248 - t117 * t250) + t125 * (t251 - r11[jacobian_id] * t245_tmp);
      d_g11_d_j31[jacobian_id] = (t125 * t257 - t111 * (r11[jacobian_id] * t160_tmp - t294_tmp)) + t117 * (t255 -
        r12[jacobian_id] * t164_tmp);
      d_g11_d_j32[jacobian_id] = (t125 * t263 - t111 * (r11[jacobian_id] * t166_tmp - t296_tmp)) + t117 * (t261 -
        r12[jacobian_id] * t170_tmp);
      d_g11_d_j33[jacobian_id] = (t125 * t269 - t111 * (r11[jacobian_id] * t172_tmp - t298_tmp)) + t117 * (t267 -
        r12[jacobian_id] * t176_tmp);
      t48 = r21[jacobian_id] * t75 - r22[jacobian_id] * t97;
      d_g12_d_j11[jacobian_id] = ((((-t182 * t210 + t184 * t215) - t195 * t302) + t199 * t304) + t204
                 * t125_tmp) + t190 * t48;
      t43 = r21[jacobian_id] * t113 - r22[jacobian_id] * t119;
      d_g12_d_j12[jacobian_id] = ((((-t206 * t210 - t215 * t216) - t195 * t311) - t199 * t321) +
                  t204 * b_t125_tmp) + t190 * t43;
      t69 = r21[jacobian_id] * t127 - r22[jacobian_id] * t129;
      d_g12_d_j13[jacobian_id] = ((((-t210 * t222 - t215 * t224) - t195 * t324) - t199 * t326) +
                  t204 * t150) + t190 * t69;
      t117_tmp = r23[jacobian_id] * t143 - r21[jacobian_id] * t227_tmp;
      d_g12_d_j21[jacobian_id] = ((((t204 * t230 - t210 * t232) + t190 * t328) - t195 * t330) + t215
                  * (t233 - r11[jacobian_id] * (t148 - t228))) + t199 * t117_tmp;
      b_t117_tmp = r23[jacobian_id] * t152 - r21[jacobian_id] * t236_tmp;
      d_g12_d_j22[jacobian_id] = ((((t204 * t239 - t210 * t241) + t190 * t333) - t195 * t335) + t215
                  * (t242 - r11[jacobian_id] * (t153 - t237))) + t199 * b_t117_tmp;
      c_t117_tmp = r23[jacobian_id] * t157 - r21[jacobian_id] * t245_tmp;
      d_g12_d_j23[jacobian_id] = ((((t204 * t248 - t210 * t250) + t190 * t338) - t195 * t340) + t215
                  * (t251 - r11[jacobian_id] * (t158 - t246))) + t199 * c_t117_tmp;
      t87 = t294_tmp - r11[jacobian_id] * (t159 - t253);
      t57 = r22[jacobian_id] * t162 - r21[jacobian_id] * t160_tmp;
      t74 = r23[jacobian_id] * t160_tmp - r22[jacobian_id] * t164_tmp;
      d_g12_d_j31[jacobian_id] = ((((t215 * t257 + t199 * t345) + t210 * (t255 - r12[jacobian_id] * (t163 - t254)))
                   + t204 * t87) + t190 * t57) + t195 * t74;
      t30 = t296_tmp - r11[jacobian_id] * (t165 - t259);
      t38 = r22[jacobian_id] * t168 - r21[jacobian_id] * t166_tmp;
      t34 = r23[jacobian_id] * t166_tmp - r22[jacobian_id] * t170_tmp;
      d_g12_d_j32[jacobian_id] = ((((t215 * t263 + t199 * t349) + t210 * (t261 - r12[jacobian_id] * (t169 - t260)))
                   + t204 * t30) + t190 * t38) + t195 * t34;
      t26 = t298_tmp - r11[jacobian_id] * (t171 - t265);
      t39 = r22[jacobian_id] * t174 - r21[jacobian_id] * t172_tmp;
      t16 = r23[jacobian_id] * t172_tmp - r22[jacobian_id] * t176_tmp;
      d_g12_d_j33[jacobian_id] = ((((t215 * t269 + t199 * t353) + t210 * (t267 - r12[jacobian_id] * (t175 - t266)))
                   + t204 * t26) + t190 * t39) + t195 * t16;
      t25 = r31[jacobian_id] * t75 - r32[jacobian_id] * t97;
      d_g13_d_j11[jacobian_id] = ((((-t182 * t280 + t184 * t285) - t195 * t356) + t199 * t358) +
                  t275 * t125_tmp) + t190 * t25;
      t125_tmp = r31[jacobian_id] * t113 - r32[jacobian_id] * t119;
      d_g13_d_j12[jacobian_id] = ((((-t206 * t280 - t216 * t285) - t195 * t363) - t199 * t365) +
                  t275 * b_t125_tmp) + t190 * t125_tmp;
      b_t125_tmp = r31[jacobian_id] * t127 - r32[jacobian_id] * t129;
      d_g13_d_j13[jacobian_id] = ((((-t222 * t280 - t224 * t285) - t195 * t370) - t199 * t372) +
                  t275 * t150) + t190 * b_t125_tmp;
      t150 = r33[jacobian_id] * t143 - r31[jacobian_id] * t227_tmp;
      d_g13_d_j21[jacobian_id] = ((((t230 * t275 - t232 * t280) + t190 * t376) - t195 * t378) + t285
                  * (t233 - r11[jacobian_id] * (t148 - t228))) + t199 * t150;
      t6 = r33[jacobian_id] * t152 - r31[jacobian_id] * t236_tmp;
      d_g13_d_j22[jacobian_id] = ((((t239 * t275 - t241 * t280) + t190 * t383) - t195 * t385) + t285
                  * (t242 - r11[jacobian_id] * (t153 - t237))) + t199 * t6;
      t3 = r33[jacobian_id] * t157 - r31[jacobian_id] * t245_tmp;
      d_g13_d_j23[jacobian_id] = ((((t248 * t275 - t250 * t280) + t190 * t390) - t195 * t121) + t285
                  * (t251 - r11[jacobian_id] * (t158 - t246))) + t199 * t3;
      t7 = r32[jacobian_id] * t162 - r31[jacobian_id] * t160_tmp;
      t5 = r33[jacobian_id] * t160_tmp - r32[jacobian_id] * t164_tmp;
      d_g13_d_j31[jacobian_id] = ((((t257 * t285 + t199 * t137) + t280 * (t255 - r12[jacobian_id] * (t163 - t254)))
                   + t275 * t87) + t190 * t7) + t195 * t5;
      t87 = r32[jacobian_id] * t168 - r31[jacobian_id] * t166_tmp;
      t4 = r33[jacobian_id] * t166_tmp - r32[jacobian_id] * t170_tmp;
      d_g13_d_j32[jacobian_id] = ((((t263 * t285 + t199 * t131) + t280 * (t261 - r12[jacobian_id] * (t169 - t260)))
                   + t275 * t30) + t190 * t87) + t195 * t4;
      t30 = r32[jacobian_id] * t174 - r31[jacobian_id] * t172_tmp;
      t2 = r33[jacobian_id] * t172_tmp - r32[jacobian_id] * t176_tmp;
      d_g13_d_j33[jacobian_id] = ((((t269 * t285 + t199 * t409) + t280 * (t267 - r12[jacobian_id] * (t175 - t266)))
                   + t275 * t26) + t190 * t30) + t195 * t2;
      d_g22_d_j11[jacobian_id] = (-t302 * t315 + t304 * t320) + t309 * t48;
      d_g22_d_j12[jacobian_id] = (-t311 * t315 - t320 * t321) + t309 * t43;
      d_g22_d_j13[jacobian_id] = (-t315 * t324 - t320 * t326) + t309 * t69;
      d_g22_d_j21[jacobian_id] = (t309 * t328 - t315 * t330) + t320 * t117_tmp;
      d_g22_d_j22[jacobian_id] = (t309 * t333 - t315 * t335) + t320 * b_t117_tmp;
      d_g22_d_j23[jacobian_id] = (t309 * t338 - t315 * t340) + t320 * c_t117_tmp;
      d_g22_d_j31[jacobian_id] = (t320 * t345 + t309 * t57) + t315 * t74;
      d_g22_d_j32[jacobian_id] = (t320 * t349 + t309 * t38) + t315 * t34;
      d_g22_d_j33[jacobian_id] = (t320 * t353 + t309 * t39) + t315 * t16;
      d_g23_d_j11[jacobian_id] = ((((-t210 * t356 + t215 * t358) - t280 * t302) + t285 * t304) +
                  t204 * t25) + t275 * t48;
      d_g23_d_j12[jacobian_id] = ((((-t210 * t363 - t215 * t365) - t280 * t311) - t285 * t321) +
                  t204 * t125_tmp) + t275 * t43;
      d_g23_d_j13[jacobian_id] = ((((-t210 * t370 - t215 * t372) - t280 * t324) - t285 * t326) +
                  t204 * b_t125_tmp) + t275 * t69;
      d_g23_d_j21[jacobian_id] = ((((t204 * t376 - t210 * t378) + t275 * t328) - t280 * t330) + t215
                  * t150) + t285 * t117_tmp;
      d_g23_d_j22[jacobian_id] = ((((t204 * t383 - t210 * t385) + t275 * t333) - t280 * t335) + t215
                  * t6) + t285 * b_t117_tmp;
      d_g23_d_j23[jacobian_id] = ((((t204 * t390 - t210 * t121) + t275 * t338) - t280 * t340) + t215
                  * t3) + t285 * c_t117_tmp;
      d_g23_d_j31[jacobian_id] = ((((t215 * t137 + t285 * t345) + t204 * t7) + t275 * t57) + t210 *
                  t5) + t280 * t74;
      d_g23_d_j32[jacobian_id] = ((((t215 * t131 + t285 * t349) + t204 * t87) + t275 * t38) + t210 *
                  t4) + t280 * t34;
      d_g23_d_j33[jacobian_id] = ((((t215 * t409 + t285 * t353) + t204 * t30) + t275 * t39) + t210 *
                  t2) + t280 * t16;
      d_g33_d_j11[jacobian_id] = (-t356 * t107 + t358 * t426) + t155 * t25;
      d_g33_d_j12[jacobian_id] = (-t363 * t107 - t365 * t426) + t155 * t125_tmp;
      d_g33_d_j13[jacobian_id] = (-t370 * t107 - t372 * t426) + t155 * b_t125_tmp;
      d_g33_d_j21[jacobian_id] = (t376 * t155 - t378 * t107) + t426 * t150;
      d_g33_d_j22[jacobian_id] = (t383 * t155 - t385 * t107) + t426 * t6;
      d_g33_d_j23[jacobian_id] = (t390 * t155 - t121 * t107) + t426 * t3;
      d_g33_d_j31[jacobian_id] = (t137 * t426 + t155 * t7) + t107 * t5;
      d_g33_d_j32[jacobian_id] = (t131 * t426 + t155 * t87) + t107 * t4;
      d_g33_d_j33[jacobian_id] = (t409 * t426 + t155 * t30) + t107 * t2;
    }
  }
} // namespace MMORF
