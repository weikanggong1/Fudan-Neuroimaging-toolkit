#ifndef EDDY_HESSIAN_HELPER_FUNCTIONS_CUH
#define EDDY_HESSIAN_HELPER_FUNCTIONS_CUH

#include <vector>

//#include <cstdio>
#include <cuda.h>

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
                                unsigned int *lind);

    // Convert from linear coordinates to volume coordinates
    __host__ __device__ void index_lin_to_vol(
                                // Input
                                // Index into linear array
                                unsigned int lind,
                                // Vol dims
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                 // Output
                                unsigned int *xind, unsigned int *yind, unsigned int *zind);

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
                                unsigned int *lind_out);

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
                                unsigned int *first_column, unsigned int *last_column);

    __host__ __device__ void identify_diagonal_fast(// Input
                                unsigned int diag_ind, // Index of current diagonal
                                // No. of overlapping splines in each direction
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                // No. of diagonals to actually retrieve in each direction
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                // Output
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column);

    __host__ std::vector<unsigned int> get_spl_coef_dim(const std::vector<unsigned int>& ksp,
                                                        const std::vector<unsigned int>& isz);
}

#endif // EDDY_HESSIAN_HELPER_FUNCTIONS_CUH
