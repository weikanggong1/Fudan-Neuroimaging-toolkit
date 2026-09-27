
// Author:  Frederik Lange
// Date:    01/09/2017
//
// These are the Kernels required to calculate the hessian for a single gradient image
#include <vector>

#include <cuda.h>

#include "EddyHessianHelperFunctions.cuh"
#include "EddyHessianKernels.cuh"

// HessianHelperFunctions Definitions
namespace MMORF
{
    // Convert from volume coordinates to linear coordinates
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

    // Convert from linear coordinates to volume coordinates
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

    // Convert from linear coordinates back to linear coordinates of different spaces
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

    // Calculate the start and end row & column for each diagonal in the Hessian
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

    __host__ __device__ void identify_diagonal_fast(// Input
                                unsigned int diag_ind, // Index of current diagonal
                                // No. of overlapping splines in each direction
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                // No. of diagonals to actually retrieve in each direction
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                // Output
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column)
    {
        // Reference variables
        unsigned int hess_side_length = spl_x*spl_y*spl_z;

        // Ensure the main diagonal value is valid
        unsigned int main_diag_lind = rep_x*rep_y*rep_z-1;
        unsigned int hess_main_diag_lind = 0;
        unsigned int hess_lind = 0;
        index_lin_to_lin(main_diag_lind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_main_diag_lind);
        // Calculate linear index into hessian
        index_lin_to_lin(diag_ind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_lind);
        // Deal with below main diagonal
        if (diag_ind <= main_diag_lind)
        {
            *first_row = hess_main_diag_lind - hess_lind;
            *last_row = hess_side_length - 1;
            *first_column = 0;
            *last_column = hess_side_length - *first_row - 1;
        }

        // Deal with above main diagonal
        else if (diag_ind > main_diag_lind)
        {
            *first_column = hess_lind - hess_main_diag_lind;
            *last_column = hess_side_length - 1;
            *first_row = 0;
            *last_row = hess_side_length - *first_column - 1;
        }
    }

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
// HessianKernels Definitions
//////////////////////////////////////////////////////////////////////////////////////////////
namespace MMORF
{
  /** details This is the most important part of the code so far. We launch a single Kernel
   * which is responsible for calculating the Hessian. The first iteration of this code was
   * monolithic and really hard to follow, but I was paranoid about performance. This time I
   * I am trying to do a much better job of splitting things up into useful functions. Also,
   * I will try and give the compiler at least a passable shot at optimising by declaring
   * everything inline!
   *
   * An important aspect of this kernel is the logic regarding block and thread ids. They
   * work as follows:
   *
   * gridDim.x  = total number of diagonals being calculated (343 if no symmetry used)
   * gridDim.y  = total number of chunks each diagonal is broken into
   * blockIdx.x = number of the current diagonal in jtj, starting at 0 for the lowest
   *              diagonal present (343 total for cubic spline, therefore main diagonal has
   *              blockIdx.x == 171)
   * blockIdx.y = number of the current chunk within the current diagonal
   * blockDim.x = total number of threads launched for each block (i.e. the size of each chunk
   *              within each diagonal)
   *
   * \param ima_sz_x size of image in x-direction
   * \param ima_sz_y size of image in y-direction
   * \param ima_sz_z size of image in z-direction
   * \param ima pre-multiplied image volume
   * \param spl_x 1D spline kernel in x-direction
   * \param spl_y 1D spline kernel in y-direction
   * \param spl_z 1D spline kernel in z-direction
   * \param spl_ksp_x knot spacing of spline in x-direction
   * \param spl_ksp_y knot spacing of spline in y-direction
   * \param spl_ksp_z knot spacing of spline in z-direction
   * \param param_sz_x size of parameter space in x-direction
   * \param param_sz_y size of parameter space in y-direction
   * \param param_sz_z size of parameter space in z-direction
   * \param jtj_offsets offsets of sparse diagonal representation of jtj
   *
   * \param jtj_values linearise values of sparse diagonal representation of jtj
   */
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

    //if (this_row == 10 && this_col == 9){
    //  std::cout << "Spline 1 x_ind start/end: " << spl1_

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
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + static_cast<int>(k);
      //if (!MMORF::is_valid_index(vol_zind, ima_sz_z)) continue;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + static_cast<int>(j);
        //if (!MMORF::is_valid_index(vol_yind, ima_sz_y)) continue;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + static_cast<int>(i);
          //if (!MMORF::is_valid_index(vol_xind, ima_sz_x)) continue;
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
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      /** \todo Replace this if with a hard index check outside the loop */
      //if (!MMORF::is_valid_index(vol_zind, ima_sz_z)) continue;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        /** \todo Replace this if with a hard index check outside the loop */
        //if (!MMORF::is_valid_index(vol_yind, ima_sz_y)) continue;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          /** \todo Replace this if with a hard index check outside the loop */
          //if (!MMORF::is_valid_index(vol_xind, ima_sz_x)) continue;
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
    jtj_values[diag_idx*jtj_sz_diag + this_row] = jtj_val;
  } // kernel_make_jtj_symmetrical

    // Here is the kernel. There will be 343*dl copies of this kernel launched by a single call.
    // Each of these copies are responsible for calculating a single value in the Hessian
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
                    float* __restrict__ hess_val)
    {
        __shared__ unsigned int first_col, last_col;
        __shared__ unsigned int first_row, last_row;
        extern __shared__ float all_splines[];
        float *shared_spline_x = all_splines;
        float *shared_spline_y = &all_splines[kspx*4 - 1];
        float *shared_spline_z = &shared_spline_y[kspy*4 - 1];
        // We only need to calculate the valid spline indices once per diagonal!
        __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
        __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
        __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

        // Calculate first column and row of each diagonal
        // Note: Main diagonal is block with ID = 171
        // Essentially the way we are doing this is that that we decompose the Block ID
        // into a "base-7" representation (for a cubic spline there are 7 overlaps in
        // each direction). We then use this to calculate the first row and column that
        // this particular Block ID represents
        if (threadIdx.x == 0)
        {
            // Determine which diagonal this is
            MMORF::identify_diagonal(blockIdx.x,7,7,7,spl_szx,spl_szy,spl_szz,
                        &first_row,&last_row,&first_col,&last_col);

            // Calculate valid spline indices
            unsigned int spl1_midx, spl1_midy, spl1_midz, spl2_midx, spl2_midy, spl2_midz;
            // Calculate spline centres
            MMORF::index_lin_to_vol(first_row,spl_szx,spl_szy,spl_szz,&spl1_midx,&spl1_midy,&spl1_midz);
            MMORF::index_lin_to_vol(first_col,spl_szx,spl_szy,spl_szz,&spl2_midx,&spl2_midy,&spl2_midz);
            unsigned int spl_diffx = 0;
            unsigned int spl_diffy = 0;
            unsigned int spl_diffz = 0;

            // Calculate start and end indices into splines
            // Begin with x-direction
            if (spl1_midx == spl2_midx) // Splines have same x centre
            {
                spl1_xind_start = spl2_xind_start = 0;
                spl1_xind_end = kspx*4 - 2;
            }
            else if (spl1_midx < spl2_midx)
            { // Spline 1 on "left"
                spl_diffx = spl2_midx - spl1_midx;
        //      if (spl_diffx > 3) printf("1) spl_diffx = %u\n",spl_diffx);
                spl1_xind_start = spl_diffx*kspx;
                spl2_xind_start = 0;
                spl1_xind_end = kspx*4 - 2;
                //spl2_midxIndEnd = kspx*4 - 2 - spl_diffx*kspx;
            }
            else if (spl1_midx > spl2_midx)
            { // Splin2 on "left"
                spl_diffx = spl1_midx - spl2_midx;
        //      if (spl_diffx > 3) printf("2) spl_diffx = %u\n",spl_diffx);
                spl2_xind_start = spl_diffx*kspx;
                spl1_xind_start = 0;
                //spl2_midxIndEnd = kspx*4 - 2;
                spl1_xind_end = kspx*4 - 2 - spl_diffx*kspx;
            }

            // Now  y-direction
            if (spl1_midy == spl2_midy)
            { // Splines have same y centre
                spl1_yind_start = spl2_yind_start = 0;
                spl1_yind_end = kspy*4 - 2;
            }
            else if (spl1_midy < spl2_midy)
            { // Spline 1 on "left"
                spl_diffy = spl2_midy - spl1_midy;
        //      if (spl_diffy > 3) printf("3) spl_diffy = %u\n",spl_diffy);
                spl1_yind_start = spl_diffy*kspy;
                spl2_yind_start = 0;
                spl1_yind_end = kspy*4 - 2;
                //spl2_midyIndEnd = kspy*4 - 2 - spl_diffy*kspy;
            }
            else if (spl1_midy > spl2_midy)
            { // Splin2 on "left"
                spl_diffy = spl1_midy - spl2_midy;
        //      if (spl_diffy > 3) printf("4) spl_diffy = %u\n",spl_diffy);
                spl2_yind_start = spl_diffy*kspy;
                spl1_yind_start = 0;
                //spl2_midyIndEnd = kspy*4 - 2;
                spl1_yind_end = kspy*4 - 2 - spl_diffy*kspy;
            }

            // Now  z-direction
            if (spl1_midz == spl2_midz)
            { // Splines have same z centre
                spl1_zind_start = spl2_zind_start = 0;
                spl1_zind_end = kspz*4 - 2;
            }
            else if (spl1_midz < spl2_midz)
            { // Spline 1 on "left"
                spl_diffz = spl2_midz - spl1_midz;
        //      if (spl_diffz > 3) printf("5) spl_diffz = %u blockIdx.x = %d\n",spl_diffz,blockIdx.x);
                spl1_zind_start = spl_diffz*kspz;
                spl2_zind_start = 0;
                spl1_zind_end = kspz*4 - 2;
                //spl2_midzIndEnd = kspz*4 - 2 - spl_diffz*kspz;
            }
            else if (spl1_midz > spl2_midz)
            { // Splin2 on "left"
                spl_diffz = spl1_midz - spl2_midz;
        //      if (spl_diffz > 3) printf("6) spl_diffz = %u blockIdx.x = %d\n",spl_diffz,blockIdx.x);
                spl2_zind_start = spl_diffz*kspz;
                spl1_zind_start = 0;
                //spl2_midzIndEnd = kspz*4 - 2;
                spl1_zind_end = kspz*4 - 2 - spl_diffz*kspz;
            }

        }
        __syncthreads();

        // Populate shared spline
        for (int i = 0; i*blockDim.x + threadIdx.x < (4*kspx -1); ++i)
        {
            shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
            shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
            shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
        }
        __syncthreads();

        // Calculate where on the diagonal this thread begins
        // Because we cannot launch a thread for each point in the Hessian, a thread must
        // deal with a subset of the points on a diagonal. Which points are based on the
        // Thread ID and the chunk size.
        unsigned int this_chunk_start = blockIdx.y*blockDim.x + threadIdx.x;
        // Valid rows and columns for this thread
        unsigned int this_row = first_row + this_chunk_start;
        unsigned int this_col = first_col + this_chunk_start;

        // If the beginning of the chunk is after the last row or column, then this thread
        // has no work to do.
        //if ((this_row > last_row) || (this_col > last_col)
        //        || (this_col > this_row) || (this_col%(spl_szy*spl_szx) > this_row%(spl_szy*spl_szx))
        //            || (this_col%(spl_szx) > this_row%(spl_szx))) return;
        if ((this_row > last_row) || (this_col > last_col)) return;

        unsigned int this_spl1_midx, this_spl1_midy, this_spl1_midz, this_spl2_midx, this_spl2_midy, this_spl2_midz;
        // Calculate spline centres
        MMORF::index_lin_to_vol(this_row,spl_szx,spl_szy,spl_szz,&this_spl1_midx,&this_spl1_midy,&this_spl1_midz);
        MMORF::index_lin_to_vol(this_col,spl_szx,spl_szy,spl_szz,&this_spl2_midx,&this_spl2_midy,&this_spl2_midz);
        // If the two splines are at a wrap, skip the rest of the loop
        // NOTE: Due to using unsigned ints, this also takes care of the sparsity!
        if ((this_spl1_midx - this_spl2_midx)  > 3 || (this_spl1_midy - this_spl2_midy) > 3 || (this_spl1_midz - this_spl2_midz) > 3) return;

        // Calculate valid voxel locations (Using spline 1 indices)
        // Start with x indices note: they may go negative!
        int vol_xind_start;
        int vol_yind_start;
        int vol_zind_start;

        vol_xind_start = static_cast<int>(this_spl1_midx)*static_cast<int>(kspx) + static_cast<int>(spl1_xind_start) - (3*static_cast<int>(kspx) - 1);
        vol_yind_start = static_cast<int>(this_spl1_midy)*static_cast<int>(kspy) + static_cast<int>(spl1_yind_start) - (3*static_cast<int>(kspy) - 1);
        vol_zind_start = static_cast<int>(this_spl1_midz)*static_cast<int>(kspz) + static_cast<int>(spl1_zind_start) - (3*static_cast<int>(kspz) - 1);

        //volxIndEnd = static_cast<int>(spl1_midx)*static_cast<int>(kspx) + static_cast<int>(spl1_xind_end) - (3*static_cast<int>(kspx) - 1);
        //volyIndEnd = static_cast<int>(spl1_midy)*static_cast<int>(kspy) + static_cast<int>(spl1_yind_end) - (3*static_cast<int>(kspy) - 1);
        //volzIndEnd = static_cast<int>(spl1_midz)*static_cast<int>(kspz) + static_cast<int>(spl1_zind_end) - (3*static_cast<int>(kspz) - 1);

        // Knowing where in the Hessian this thread goes, we can now start populating it.
        // Now, looping over all valid voxels, calculate the actual Hessian Value
        float this_hess_val = 0.0;
        for (unsigned int k = 0; spl1_zind_start + k <= spl1_zind_end; ++k)
        {
            int this_vol_zind = vol_zind_start + static_cast<int>(k); // Maybe use a continue to skip the rest of this loop if negative
            // Ensure voxel index is valid (Note: alternative check for this below)
            if (this_vol_zind >= ima_szz) continue;
            for (unsigned int j = 0; spl1_yind_start + j <= spl1_yind_end; ++j)
            {
                int this_vol_yind = vol_yind_start + static_cast<int>(j);
                // Ensure voxel index is valid (Note: alternative check for this below)
                if (this_vol_yind >= ima_szy) continue;
                for (unsigned int i = 0; spl1_xind_start + i <= spl1_xind_end; ++i)
                {
                    int this_vol_xind = vol_xind_start + static_cast<int>(i);
                    // Ensure voxel index is valid (Note: alternative check for this below)
                    if (this_vol_xind >= ima_szx) continue;
                    else
                    {
                        unsigned int vol_lind = 0;
                        unsigned int this_spl1_zInd = spl1_zind_start + k;
                        unsigned int this_spl2_zInd = spl2_zind_start + k;
                        unsigned int this_spl1_yInd = spl1_yind_start + j;
                        unsigned int this_spl2_yInd = spl2_yind_start + j;
                        unsigned int this_spl1_xInd = spl1_xind_start + i;
                        unsigned int this_spl2_xInd = spl2_xind_start + i;
                        //float temp_hess_val = 0.0;
                        // If the voxel is outside of the image, skip rest of the loop
                        MMORF::index_vol_to_lin(this_vol_xind,this_vol_yind,this_vol_zind,ima_szx,ima_szy,ima_szz,&vol_lind);// continue;
                        //temp_hess_val += tex1Dfetch<float>(ima1,vol_lind);
                        // get linear indices for valid spline and voxel indices
                        //MMORF::index_vol_to_lin(this_spl1_xInd,this_spl1_yInd,this_spl1_zInd,kspx*4-1,kspy*4-1,kspz*4-1,&spl1_lind);// continue;
                        //temp_hess_val *= shared_spline_x[this_spl1_xInd];
                        //temp_hess_val *= shared_spline_x[this_spl1_yInd];
                        //temp_hess_val *= shared_spline_x[this_spl1_zInd];
                        //temp_hess_val *= shared_spline_x[this_spl2_xInd];
                        //temp_hess_val *= shared_spline_x[this_spl2_yInd];
                        //temp_hess_val *= shared_spline_x[this_spl2_zInd];
                        //MMORF::index_vol_to_lin(this_spl2_xInd,this_spl2_yInd,this_spl2_zInd,kspx*4-1,kspy*4-1,kspz*4-1,&spl2_lind);// continue;
                        //float temp_vol = vol_lind;
                        // Add value to total for this point on the Hessian
                        //this_hess_val += temp_hess_val;
                        this_hess_val += tex1Dfetch<float>(ima1,vol_lind)*
                                         shared_spline_x[this_spl1_xInd]*
                                         shared_spline_y[this_spl1_yInd]*
                                         shared_spline_z[this_spl1_zInd]*
                                         shared_spline_x[this_spl2_xInd]*
                                         shared_spline_y[this_spl2_yInd]*
                                         shared_spline_z[this_spl2_zInd];
                        //this_hess_val += ima[vol_lind]*ima[vol_lind]*spl[spl1_lind]*spl[spl2_lind];
                    }
                }
            }
        }

        // Save current value of Hessian
        unsigned int diagonal_length = spl_szx*spl_szy*spl_szz;
        unsigned int sparse_hess_lind = blockIdx.x*diagonal_length + blockDim.x*blockIdx.y + threadIdx.x;
        // By this stage, we should (hopefully) only ever be in a unique area of the Hessian
        ///////////////////////////////////////////////////
        // NOTE: NB! This next bit is utter black magic! //
        ///////////////////////////////////////////////////
        unsigned int offset_1 = (6 - 2*blockIdx.x%7);
        unsigned int offset_2 = (48 - 2*blockIdx.x%49);
        unsigned int offset_3 = (offset_2 - offset_1);
        hess_val[sparse_hess_lind] = this_hess_val;
        if ((blockIdx.x + offset_1) < 172)
        {
            hess_val[sparse_hess_lind + offset_1*diagonal_length + offset_1/2] = this_hess_val;
        }
        if ((blockIdx.x + offset_2) < 172)
        {
            hess_val[sparse_hess_lind + offset_2*diagonal_length + (offset_3/14)*spl_szx + offset_1/2] = this_hess_val;
        }
        if ((blockIdx.x + offset_3) < 172)
        {
            hess_val[sparse_hess_lind + offset_3*diagonal_length + (offset_3/14)*spl_szx] = this_hess_val;
        }
    }

    __global__ void sparse_matrix_vector_multiply(
                                                    // Input
                                                    const float *mat_vals,
                                                    const float *vec,
                                                    const unsigned int dimension,
                                                    const unsigned int mat_id,
                                                    const unsigned int vec_id,
                                                    const unsigned int result_id,
                                                    // Input/Output
                                                    float *result)
    {
        // This is the id within a diagonal
        const unsigned int diag_id = blockIdx.x*blockDim.x + threadIdx.x;
        // Calculate value
        if ((vec_id + diag_id) < dimension && (result_id + diag_id) < dimension)
        {
            result[result_id + diag_id] += mat_vals[mat_id + diag_id]*vec[vec_id + diag_id];
        }
    }

    __global__ void sparse_matrix_vector_multiply(
                                                    // Input
                                                    const float* __restrict__ mat_vals,
                                                    const unsigned int* __restrict__ mat_rows,
                                                    const float* __restrict__ vec,
                                                    const unsigned int dimension,
                                                    // Output
                                                    float* __restrict__ result)
    {
        __shared__ unsigned int offsets [343];
        // Populate offsets
        if (threadIdx.x < 172)
        {
            offsets[threadIdx.x] = mat_rows[threadIdx.x];
        }
        if (threadIdx.x >= 172 && threadIdx.x < 343)
        {
            offsets[threadIdx.x] = mat_rows[342-threadIdx.x];
        }
        __syncthreads();

        const unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        if (id < dimension)
        {
            float result_accumulator = 0.0;
            // Loop across all diagonals
            for (unsigned int i = 0; i < 343; i++)
            {
                // Calculate index into current diagonal and rhs vector
                //
                // mat_id value is dependat on which side of the main diagonal we are.
                // Below (and on) the main diagonal "i" corresponds exactly to the desired
                // diagonal. Above the main diagonal, we must begin indexing back through the
                // lower diagonals in reverse order. This is due to fact that oly the lower 
                // triangular matrix is stored explicitly.
                //
                // Note: the "id" variable corresponds to a row in the matrix/result vector
                //
                // The logic behind the indexing is as follows:
                // - We imagine all diagonals as starting at "row zero", but possibly
                //   at a negative column index if it is below the main diagonal.
                //   Additionally, a diagonal may extend beyond the last column, if it
                //   originates above the main diagonal. Note that this assumes all diagonals
                //   to be equal in length to the main diagonal.
                // - Obviously, any index into diagonals outside of the actual matrix bounds
                //   are not allowed. We enforce this by checking the corresponding rhs vector
                //   index to see whether it is out of bounds or not
                unsigned int vec_id = (i<172) ? id - offsets[i] : id + offsets[i];
                if (vec_id >=0 && vec_id < dimension)
                {
                    unsigned int mat_id = (i<172) ? i*dimension + id - offsets[i] :
                                                    (342-i)*dimension + id;
                    result_accumulator += mat_vals[mat_id]*vec[vec_id];
                }
            result[id] = result_accumulator;
            }
        }
    }
}
