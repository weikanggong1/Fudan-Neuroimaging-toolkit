//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the L2 norm of a masked tensor volume
/// \details Requires a B-spline parametrised warp field, Note that the Hessian calculation
///          uses a Majorise-Minimise diagonalisation strategy.
///          This is now extended to include a greedy approximation to symmetrisation of the
///          problem by multiplying the cost by (1 + |J|).
/// \author Frederik Lange
/// \date March 2021
/// \copyright Copyright (C) 2021 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess.cuh"
#include "MmorfMemory.h"
#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"
#include "TextureHandleLinear.cuh"
#include "CostFxnHelpers.cuh"
#include "CostFxnKernels.cuh"
#include "IntensityMapperPolynomial.cuh"
#include "helper_cuda.h"

#include "basisfield/fsl_splines.h"

#include <armadillo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <memory>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <functional>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, and defining warp field
      /// characteristics
      /// \param vol_ref Reference (stationary) volume
      /// \param vol_mov Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param mask_ref 3D volume used to mask the reference volume
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param knot_spacing Warp field B-spline knot spacing (mm) in reference space
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        warp field will be sampled every 2mm
      Impl(
          std::shared_ptr<MMORF::VolumeTensor>     vol_ref,
          std::shared_ptr<MMORF::VolumeTensor>     vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::Volume>           mask_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency
          )
        : vol_ref_(vol_ref)
        , vol_mov_(vol_mov)
        , affine_ref_(affine_ref)
        , affine_mov_(affine_mov)
        , mask_ref_(mask_ref)
        , mask_mov_(mask_mov)
        , warp_field_(warp_field)
        , sample_dimensions_(warp_field_->get_robust_sample_dimensions(sampling_frequency))
        , spline_1D_(3,sampling_frequency) // Cubic spline with the correct no. samples
      {
        /// \todo Replace assert with exception
        assert(sampling_frequency > 0);
        sample_positions_warp_ = warp_field_->get_robust_sample_positions(sampling_frequency);
        sample_positions_ref_ = affine_transform_(sample_positions_warp_, affine_ref_);
        // Calculate sample resolution
        auto warp_dims = warp_field_->get_dimensions();
        auto warp_extents = warp_field_->get_extents();
        for (auto i = 0; i < warp_dims.size(); ++i){
          auto dim_res =
            warp_field_->get_knot_spacing() / static_cast<float>(sampling_frequency);
          sample_resolution_.push_back(dim_res);
        }
      }

      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const
      {
        return warp_field_->get_parameters();
      }

      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        warp_field_->set_parameters(parameters);
      }

      /// Get cost under current parameterisation
      /// \details There are several non-intuitive factors influencing the cost. The first is
      ///          that the warp space is not necessarily the same as the reference image
      ///          space. Therefore we need reorient the reference samples according to
      ///          affine_ref_. This is similarly true for the moving image and affine_mov_.
      ///          Finally, the warp induces local rotations on the reference image which need
      ///          to be accounted for on a voxel by voxel basis. Only once all of these
      ///          reorientations have been performed correctly can the cost be properly
      ///          calculated.
      float cost() const
      {
        // Calculate rotational effect of warp
        auto rotations = calculate_rotation_(sample_positions_warp_);
        // Sample reference volume and reorient
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        affine_reorient_samples_(samples_ref, affine_ref_.i());
        nonlin_reorient_samples_(samples_ref, rotations);
        // Sample moving volume and reorient
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        affine_reorient_samples_(samples_mov, affine_mov_.i());
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate cost from samples
        auto samples_error = std::vector<float>(samples_ref[0].size());
        auto cost = 0.0f;
        for (auto i = 0; i < 9; ++i){
          std::transform(
              samples_mov[i].begin(),
              samples_mov[i].end(),
              samples_ref[i].begin(),
              samples_error.begin(),
              std::minus<float>()
              );
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_error.begin(),
              samples_error.begin(),
              std::multiplies<float>()
              );
          std::transform(
              samples_error.begin(),
              samples_error.end(),
              samples_jac_det.begin(),
              samples_error.begin(),
              [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
              );
          if (mask_ref_){
            auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
            std::transform(
                samples_error.begin(),
                samples_error.end(),
                samples_mask_ref.begin(),
                samples_error.begin(),
                std::multiplies<float>()
                );
          }
          if (mask_mov_){
            auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
            std::transform(
                samples_error.begin(),
                samples_error.end(),
                samples_mask_mov.begin(),
                samples_error.begin(),
                std::multiplies<float>()
                );
          }
          cost = std::accumulate(
              samples_error.begin(),
              samples_error.end(),
              cost
              );
        }
          // Avergage cost
        cost /= samples_ref[0].size();
        return cost;
      }

      /// Get Jte under current parameterisation
      arma::fvec grad() const
      {
        // Calculate rotational effect of warp
        auto rotations = calculate_rotation_(sample_positions_warp_);
        // Calculate jte_sz
        auto warp_sz = warp_field_->get_parameter_size();
        auto jte_sz = warp_sz.first * warp_sz.second;
        // Make a device vector for storing the result
        auto jte_dev = thrust::device_vector<float>(jte_sz, 0);
        // Create and sample error volume
        // Sample reference and moving volumes
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        affine_reorient_samples_(samples_ref, affine_ref_.i());
        // Keep an un-nonlinear-rotated copy of the reference for rotation gradient calc
        auto samples_ref_unrotated = samples_ref;
        nonlin_reorient_samples_(samples_ref, rotations);
        // Sample moving volume and reorient
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        auto samples_mov = vol_mov_->sample(sample_positions_warped);
        affine_reorient_samples_(samples_mov, affine_mov_.i());
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate current sample error
        auto samples_error = std::vector<std::vector<float> >(
            9, std::vector<float>(samples_ref[0].size()));
        for (auto i = 0; i < 9; ++i){
          std::transform(
              samples_mov[i].begin(),
              samples_mov[i].end(),
              samples_ref[i].begin(),
              samples_error[i].begin(),
              std::minus<float>()
              );
          if (mask_ref_){
            auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
            std::transform(
                samples_error[i].begin(),
                samples_error[i].end(),
                samples_mask_ref.begin(),
                samples_error[i].begin(),
                std::multiplies<float>()
                );
          }
          if (mask_mov_){
            auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
            std::transform(
                samples_error[i].begin(),
                samples_error[i].end(),
                samples_mask_mov.begin(),
                samples_error[i].begin(),
                std::multiplies<float>()
                );
          }
        }
        // Convert spline to vec
        auto spline_vec = spline_as_vec_(spline_1D_);
        /////////////////////////////////////////////////////////////////////////////////////
        /// Gradient due to displacement
        /////////////////////////////////////////////////////////////////////////////////////
        // Calculate derivative samples
        auto samples_mov_deriv_vec = calculate_derivatives_mov_(
            sample_positions_warped,
            affine_mov_);
        // Loop over all warp directions
        for (auto warp_dim = 0; warp_dim < 3; ++warp_dim){
          // Pre-multiply the error and derivative samples
          auto samples_pre_mult = std::vector<float>(samples_error[0].size(), 0.0f);
          // Loop over all elements of the tensor
          for (auto tensor_elem = 0; tensor_elem < 9; ++tensor_elem){
            auto tmp_pre_mult = std::vector<float>(samples_error[0].size());
            std::transform(
                samples_error[tensor_elem].begin(),
                samples_error[tensor_elem].end(),
                samples_mov_deriv_vec[warp_dim][tensor_elem].begin(),
                tmp_pre_mult.begin(),
                std::multiplies<float>()
                );
            std::transform(
                samples_pre_mult.begin(),
                samples_pre_mult.end(),
                tmp_pre_mult.begin(),
                samples_pre_mult.begin(),
                std::plus<float>()
                );
          }
          std::transform(
              samples_pre_mult.begin(),
              samples_pre_mult.end(),
              samples_jac_det.begin(),
              samples_pre_mult.begin(),
              [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
              );
          // Get correct index into jte_dev (x, then y, then z)
          auto jte_dev_ptr = thrust::raw_pointer_cast(
              &jte_dev[warp_dim * warp_sz.second]);
          calculate_sub_jte_(
              // Input
              samples_pre_mult,
              sample_dimensions_,
              spline_vec,
              spline_vec,
              spline_vec,
              warp_field_->get_dimensions(),
              std::vector<int>(3, spline_1D_.KnotSpacing()),
              // Output
              jte_dev_ptr);
        }
        // Copy jte back to host
        auto jte_host = thrust::host_vector<float>(jte_dev);
        /////////////////////////////////////////////////////////////////////////////////////
        /// Gradient due to rotation
        /////////////////////////////////////////////////////////////////////////////////////
        // Calculate dG/dJ
        // First calculate "S"
        auto s_mats = calculate_s_(sample_positions_warp_);
        auto grad_per_jacobian_element = calculate_grad_per_jacobian_element_(
            samples_ref_unrotated,
            rotations,
            s_mats);
        auto grad_per_spline = calculate_grad_per_spline_(
            grad_per_jacobian_element,
            samples_error,
            sample_positions_warped);
        // Return result
        auto jte_return = arma::fvec(jte_host.data(), jte_host.size());
        /// \todo Check the sign (i.e., do we add or subtract this?)
        jte_return = jte_return + grad_per_spline;
        jte_return = 2.0f * jte_return/samples_ref[0].size();
        return jte_return;
      }

      /// Get JtJ under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const
      {
        // Calculate rotational effect of warp
        auto rotations = calculate_rotation_(sample_positions_warp_);
        // Get size of warp field
        auto warp_sz = warp_field_->get_parameter_size();
        // Apply warp field to sample positions
        auto sample_positions_warped = warp_field_->apply_warp(sample_positions_warp_);
        sample_positions_warped = affine_transform_(sample_positions_warped, affine_mov_);
        // Sample derivative of moving volume in all directions and store in vector
        auto samples_mov_deriv_vec = calculate_derivatives_mov_(
            sample_positions_warped,
            affine_mov_);
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        /////////////////////////////////////////////////////////////////////////////////////
        /// Hessian due to displacement
        /////////////////////////////////////////////////////////////////////////////////////
        // Create sparse tiled matrix to store Hessian
        auto sparse_jtj = create_empty_jtj_(warp_field_->get_dimensions());
        // Convert spline to vec
        auto spline_vec = spline_as_vec_(spline_1D_);
        // Loop over all warp directions
        for (auto row = 0; row < warp_sz.first; ++row){
          // Loop over all warp directions again
          for (auto col = 0; col < warp_sz.first; ++col){
            // Check if this is a unique sub-jtj - we will keep the main diagonal and below
            if (col > row) continue;
            // Pre-multiply the derivative samples
            auto samples_pre_mult = std::vector<float>(
                samples_mov_deriv_vec[0][0].size(), 0.0f);
            // Loop over all elements of the tensor
            for (auto tensor_elem = 0; tensor_elem < 9; ++tensor_elem){
              auto tmp_pre_mult = std::vector<float>(samples_pre_mult.size());
              std::transform(
                  samples_mov_deriv_vec[row][tensor_elem].begin(),
                  samples_mov_deriv_vec[row][tensor_elem].end(),
                  samples_mov_deriv_vec[col][tensor_elem].begin(),
                  tmp_pre_mult.begin(),
                  std::multiplies<float>()
                  );
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  tmp_pre_mult.begin(),
                  samples_pre_mult.begin(),
                  std::plus<float>()
                  );
            }
            // Pre-multiply by the Jacobian modulation
            std::transform(
                samples_pre_mult.begin(),
                samples_pre_mult.end(),
                samples_jac_det.begin(),
                samples_pre_mult.begin(),
                [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
                );
            if (mask_ref_){
              auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_ref.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            if (mask_mov_){
              auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_mov.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            // Call kernel
            calculate_sub_jtj_symmetrical_(
              // Input
              samples_pre_mult,
              sample_dimensions_,
              spline_vec,
              spline_vec,
              spline_vec,
              warp_field_->get_dimensions(),
              std::vector<int>(3, spline_1D_.KnotSpacing()),
              row,
              col,
              sparse_jtj);
            // Save above diagonal sub-jtj
          }
        }
        /////////////////////////////////////////////////////////////////////////////////////
        /// Hessian due to rotation
        /////////////////////////////////////////////////////////////////////////////////////
        // Calculate grad_per_jacobian_element
        auto samples_ref = vol_ref_->sample(sample_positions_ref_);
        affine_reorient_samples_(samples_ref, affine_ref_.i());
        auto s_mats = calculate_s_(sample_positions_warp_);
        auto grad_per_jacobian_element = calculate_grad_per_jacobian_element_(
            samples_ref,
            rotations,
            s_mats);
        // Calculate hess_per_spline
        auto hess_per_spline = calculate_hess_per_spline_(
          grad_per_jacobian_element,
          sample_positions_warped);
        sparse_jtj += hess_per_spline;
        sparse_jtj *= (2.0f / static_cast<float>(samples_mov_deriv_vec[0][0].size()));
        // Return completed matrix
        return sparse_jtj;
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Affine transform a given set of positions
      std::vector<std::vector<float> > affine_transform_(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        /// \todo Exception testing
        assert(positions.size() == sample_dimensions_.size());
        auto positions_out = std::vector<std::vector<float> >(positions.size());
        // Apply affine transform
        // NB!!! Because armadillo is column major, we use a column vector for each vector of
        // positions for a particular dimension, and therefore the transpose of the affine_
        // matrix
        arma::fmat augmented_positions(positions[0].size(), positions.size() + 1);
        { // Scope limiting
          auto vec_dim = 0;
          for (const auto& vec : positions){
            augmented_positions.col(vec_dim) = arma::fvec(vec);
            ++vec_dim;
          }
          augmented_positions.col(vec_dim) = arma::fvec(
              positions[0].size(),
              arma::fill::ones);
          augmented_positions = augmented_positions * (affine_mat.t());
        } // Scope limiting
        { // Scope limiting
          auto vec_dim = 0;
          for (auto& vec : positions_out){
            vec = arma::conv_to<std::vector<float> >::from(augmented_positions.col(vec_dim));
            ++vec_dim;
          }
        } // Scope limiting
        return positions_out;
      }

      /// Calculate the rotational effect of the warp field in the warp space
      std::vector<arma::fmat> calculate_rotation_(
          const std::vector<std::vector<float> >& sample_positions) const
      {
        auto rotation_matrices = std::vector<arma::fmat>(sample_positions[0].size());
        // Begin by getting the jacobian matrices
        auto jacobian_elements = warp_field_->get_jacobian_elements(sample_positions);
        auto jacobian_matrices_arma = vector_to_arma_(jacobian_elements);
        // Calculate Finite Strain rotation matrices as follows:
        //
        //        If J = UR where U is a scaling matrix and R is a rotation, then
        //
        //        R = ((JJ^T)^-1/2)J
#pragma omp parallel for
        for (auto i = 0; i < sample_positions[0].size(); ++i){
          if (arma::sqrtmat_sympd(
                rotation_matrices[i],
                jacobian_matrices_arma[i]*(jacobian_matrices_arma[i].t()))){
            rotation_matrices[i] = rotation_matrices[i].i() * jacobian_matrices_arma[i];
          }
          else{
            rotation_matrices[i].eye(3,3);
            std::cout << "WARNING: sqrtmat_sympd failed at location " << i << std::endl;
          }
        }
        return rotation_matrices;
      }

      /// Calculate the following for each position:
      ///
      ///         S = (JJ^T)^1/2
      std::vector<arma::fmat> calculate_s_(
          const std::vector<std::vector<float> >& sample_positions) const
      {
        auto s_matrices = std::vector<arma::fmat>(sample_positions[0].size());
        // Begin by getting the jacobian matrices
        auto jacobian_elements = warp_field_->get_jacobian_elements(sample_positions);
        auto jacobian_matrices_arma = vector_to_arma_(jacobian_elements);
#pragma omp parallel for
        for (auto i = 0; i < sample_positions[0].size(); ++i){
          if (!arma::sqrtmat_sympd(
                s_matrices[i],
                jacobian_matrices_arma[i]*(jacobian_matrices_arma[i].t()))){
            s_matrices[i].eye(3,3);
            std::cout << "WARNING: sqrtmat_sympd failed at location " << i << std::endl;
          }
        }
        return s_matrices;
      }

      /// Convert a 3x3 row-major vectorised matrix into a vector of armadillo matrices
      std::vector<arma::fmat> vector_to_arma_(
          const std::vector<std::vector<float> >& vector_rm) const
      {
        auto matrices = std::vector<arma::fmat>(vector_rm[0].size());
#pragma omp parallel for
        for (auto i = 0; i < vector_rm[0].size(); ++i){
          matrices[i] =
              arma::fmat{
                {vector_rm[0][i], vector_rm[1][i], vector_rm[2][i]},
                {vector_rm[3][i], vector_rm[4][i], vector_rm[5][i]},
                {vector_rm[6][i], vector_rm[7][i], vector_rm[8][i]}};
        }
        return matrices;
      }

      /// Convert a 3x3 armadillo matrix to row-major vectorised format
      std::vector<std::vector<float> > arma_to_vector_(
          const std::vector<arma::fmat>& matrices) const
      {
        // Note that vector is row-major flattened, but armadillo matrices are column major.
        auto vectors = std::vector<std::vector<float> >(
            9, std::vector<float>(matrices.size()));
        for (auto i = 0; i < matrices.size(); ++i){
          vectors[0][i] = matrices[i][0];
          vectors[1][i] = matrices[i][3];
          vectors[2][i] = matrices[i][6];
          vectors[3][i] = matrices[i][1];
          vectors[4][i] = matrices[i][4];
          vectors[5][i] = matrices[i][7];
          vectors[6][i] = matrices[i][2];
          vectors[7][i] = matrices[i][5];
          vectors[8][i] = matrices[i][8];
        }
        return vectors;
      }

      /// Rotate tensors by calculating the rotational effect of the affine matrix as follows:
      ///
      ///     R  = ((JJ^T)^-1/2)J
      ///     D' = R*D*R^T
      void affine_reorient_samples_(
          std::vector<std::vector<float> >& tensors,
          const arma::fmat&                 affine_mat) const
      {
        arma::fmat affine_submat = affine_mat.submat(0,0,2,2);
        arma::fmat r_mat = arma::sqrtmat_sympd(
           (affine_submat * (affine_submat.t())).i()) * affine_submat;
        auto d_mats = vector_to_arma_(tensors);
#pragma omp parallel for
        for (auto i = 0; i < d_mats.size(); ++i){
          d_mats[i] = r_mat*d_mats[i]*(r_mat.t());
        }
        tensors = arma_to_vector_(d_mats);
      }

      /// Rotate tensors as follows
      ///
      ///     D' = R*D*R^T
      void nonlin_reorient_samples_(
          std::vector<std::vector<float> >& tensors,
          const std::vector<arma::fmat>&    rotations) const
      {
        /// \todo replace assert with exception
        assert(tensors[0].size() == rotations.size());
        auto rotated_tensors_arma = std::vector<arma::fmat>(rotations.size());
        auto tensor_matrices_arma = vector_to_arma_(tensors);
#pragma omp parallel for
        for (auto i = 0; i < rotations.size(); ++i){
          rotated_tensors_arma[i] =
              rotations[i]*tensor_matrices_arma[i]*(rotations[i].t());
        }
        tensors = arma_to_vector_(rotated_tensors_arma);
      }

      /// Calculate the derivitive of the moving volume wrt to changes in the x, y and z
      /// displacements in the shared reference space
      /// \detail The dimensions of the result are: [3,9,N], i.e. number of directions, number
      ///         of elements in the tensor, number of positions.
      std::vector<std::vector<std::vector<float> > > calculate_derivatives_mov_(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        /// \todo replace assert with exception
        assert(positions.size() == sample_dimensions_.size());
        auto sample_sz = positions[0].size();
        // Create an armadillo matrix to store the samples as this will make life easier when
        // doing the affine transform
        /// \todo change dims from [3][9][n] to [9][3][n]
        auto derivatives_vec = vol_mov_->sample_gradient(positions);
        auto derivatives = std::vector<std::vector<std::vector<float> > >(
            3,
            std::vector<std::vector<float> >(
              9,
              std::vector<float>(
                sample_sz)));
#pragma omp parallel for
        for (auto i = 0; i < 3; ++i){
          for (auto j = 0; j < 9; ++j){
            for (auto k = 0; k < sample_sz; ++k){
              derivatives[i][j][k] = derivatives_vec[j][i][k];
            }
          }
        }
        // Transform derivative through affine
#pragma omp parallel for
        for (auto i = 0; i < 9; ++ i){
          auto derivatives_arma = arma::fmat(positions[0].size(), 3);
          derivatives_arma.col(0) = arma::fvec(derivatives[0][i]);
          derivatives_arma.col(1) = arma::fvec(derivatives[1][i]);
          derivatives_arma.col(2) = arma::fvec(derivatives[2][i]);
          derivatives_arma = derivatives_arma * affine_mat.submat(0,0,2,2);
          derivatives[0][i] = arma::conv_to<std::vector<float> >::from(
              derivatives_arma.col(0));
          derivatives[1][i] = arma::conv_to<std::vector<float> >::from(
              derivatives_arma.col(1));
          derivatives[2][i] = arma::conv_to<std::vector<float> >::from(
              derivatives_arma.col(2));
        }
        // Rotate tensors through affine
#pragma omp parallel for
        for (auto i = 0; i < 3; ++i){
          affine_reorient_samples_(derivatives[i], affine_mat.i());
        }
        return derivatives;
      }

      // Convert 1D spline to vector of floats
      std::vector<float> spline_as_vec_(
          const BASISFIELD::Spline1D<float>& spline,
          float spline_res = 1.0f,
          int diff_order = 0) const
      {
        auto spline_vals = std::vector<float>(spline.KernelSize());
        for (auto i = 0; i < spline.KernelSize(); ++i)
        {
          spline_vals.at(i) =
            spline(static_cast<unsigned int>(i+1))
            / std::pow(spline_res, diff_order);
        }
        return spline_vals;
      }

      // Calculate a sub-vector of Jte
      void calculate_sub_jte_(
          // Input
          const std::vector<float>& prod_ima,
          const std::vector<int>&   ima_sz,
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>&   coef_sz,
          const std::vector<int>&   ksp,
          // Output
          float*                    sub_jte_raw) const
      {
        // Make everything a device vector using thrust::
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_make_jte,
            0,
            0);
        unsigned int blocks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_1d(blocks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jte<<<blocks_1d,threads,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            // Output
            sub_jte_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
      }

      /// Calculate the gradient of the cost per jacobian element
      std::vector<std::vector<std::vector<float> > > calculate_grad_per_jacobian_element_(
          const std::vector<std::vector<float> >& g_samples,
          const std::vector<arma::fmat>&          r_mats,
          const std::vector<arma::fmat>&          s_mats) const
      {
        // Flatten r_mats and s_mats
        auto r_samples = arma_to_vector_(r_mats);
        auto s_samples = arma_to_vector_(s_mats);

        // Create device vectors using thrust::
        auto n_samples = g_samples[0].size();

        auto g11_dev = thrust::device_vector<float>(g_samples[0]);
        auto g12_dev = thrust::device_vector<float>(g_samples[1]);
        auto g13_dev = thrust::device_vector<float>(g_samples[2]);
        auto g22_dev = thrust::device_vector<float>(g_samples[4]);
        auto g23_dev = thrust::device_vector<float>(g_samples[5]);
        auto g33_dev = thrust::device_vector<float>(g_samples[8]);

        auto r11_dev = thrust::device_vector<float>(r_samples[0]);
        auto r12_dev = thrust::device_vector<float>(r_samples[1]);
        auto r13_dev = thrust::device_vector<float>(r_samples[2]);
        auto r21_dev = thrust::device_vector<float>(r_samples[3]);
        auto r22_dev = thrust::device_vector<float>(r_samples[4]);
        auto r23_dev = thrust::device_vector<float>(r_samples[5]);
        auto r31_dev = thrust::device_vector<float>(r_samples[6]);
        auto r32_dev = thrust::device_vector<float>(r_samples[7]);
        auto r33_dev = thrust::device_vector<float>(r_samples[8]);

        auto s11_dev = thrust::device_vector<float>(s_samples[0]);
        auto s12_dev = thrust::device_vector<float>(s_samples[1]);
        auto s13_dev = thrust::device_vector<float>(s_samples[2]);
        auto s22_dev = thrust::device_vector<float>(s_samples[4]);
        auto s23_dev = thrust::device_vector<float>(s_samples[5]);
        auto s33_dev = thrust::device_vector<float>(s_samples[8]);

        auto d_g11_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g11_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g12_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g12_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g13_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g13_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g21_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g21_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g22_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g22_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g23_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g23_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g31_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g31_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g32_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g32_d_j33_dev = thrust::device_vector<float>(n_samples);

        auto d_g33_d_j11_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j12_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j13_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j21_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j22_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j23_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j31_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j32_dev = thrust::device_vector<float>(n_samples);
        auto d_g33_d_j33_dev = thrust::device_vector<float>(n_samples);

        // Get raw pointers to device data
        auto g11_raw = thrust::raw_pointer_cast(g11_dev.data());
        auto g12_raw = thrust::raw_pointer_cast(g12_dev.data());
        auto g13_raw = thrust::raw_pointer_cast(g13_dev.data());
        auto g22_raw = thrust::raw_pointer_cast(g22_dev.data());
        auto g23_raw = thrust::raw_pointer_cast(g23_dev.data());
        auto g33_raw = thrust::raw_pointer_cast(g33_dev.data());

        auto r11_raw = thrust::raw_pointer_cast(r11_dev.data());
        auto r12_raw = thrust::raw_pointer_cast(r12_dev.data());
        auto r13_raw = thrust::raw_pointer_cast(r13_dev.data());
        auto r21_raw = thrust::raw_pointer_cast(r21_dev.data());
        auto r22_raw = thrust::raw_pointer_cast(r22_dev.data());
        auto r23_raw = thrust::raw_pointer_cast(r23_dev.data());
        auto r31_raw = thrust::raw_pointer_cast(r31_dev.data());
        auto r32_raw = thrust::raw_pointer_cast(r32_dev.data());
        auto r33_raw = thrust::raw_pointer_cast(r33_dev.data());

        auto s11_raw = thrust::raw_pointer_cast(s11_dev.data());
        auto s12_raw = thrust::raw_pointer_cast(s12_dev.data());
        auto s13_raw = thrust::raw_pointer_cast(s13_dev.data());
        auto s22_raw = thrust::raw_pointer_cast(s22_dev.data());
        auto s23_raw = thrust::raw_pointer_cast(s23_dev.data());
        auto s33_raw = thrust::raw_pointer_cast(s33_dev.data());

        auto d_g11_d_j11_raw = thrust::raw_pointer_cast(d_g11_d_j11_dev.data());
        auto d_g11_d_j12_raw = thrust::raw_pointer_cast(d_g11_d_j12_dev.data());
        auto d_g11_d_j13_raw = thrust::raw_pointer_cast(d_g11_d_j13_dev.data());
        auto d_g11_d_j21_raw = thrust::raw_pointer_cast(d_g11_d_j21_dev.data());
        auto d_g11_d_j22_raw = thrust::raw_pointer_cast(d_g11_d_j22_dev.data());
        auto d_g11_d_j23_raw = thrust::raw_pointer_cast(d_g11_d_j23_dev.data());
        auto d_g11_d_j31_raw = thrust::raw_pointer_cast(d_g11_d_j31_dev.data());
        auto d_g11_d_j32_raw = thrust::raw_pointer_cast(d_g11_d_j32_dev.data());
        auto d_g11_d_j33_raw = thrust::raw_pointer_cast(d_g11_d_j33_dev.data());

        auto d_g12_d_j11_raw = thrust::raw_pointer_cast(d_g12_d_j11_dev.data());
        auto d_g12_d_j12_raw = thrust::raw_pointer_cast(d_g12_d_j12_dev.data());
        auto d_g12_d_j13_raw = thrust::raw_pointer_cast(d_g12_d_j13_dev.data());
        auto d_g12_d_j21_raw = thrust::raw_pointer_cast(d_g12_d_j21_dev.data());
        auto d_g12_d_j22_raw = thrust::raw_pointer_cast(d_g12_d_j22_dev.data());
        auto d_g12_d_j23_raw = thrust::raw_pointer_cast(d_g12_d_j23_dev.data());
        auto d_g12_d_j31_raw = thrust::raw_pointer_cast(d_g12_d_j31_dev.data());
        auto d_g12_d_j32_raw = thrust::raw_pointer_cast(d_g12_d_j32_dev.data());
        auto d_g12_d_j33_raw = thrust::raw_pointer_cast(d_g12_d_j33_dev.data());

        auto d_g13_d_j11_raw = thrust::raw_pointer_cast(d_g13_d_j11_dev.data());
        auto d_g13_d_j12_raw = thrust::raw_pointer_cast(d_g13_d_j12_dev.data());
        auto d_g13_d_j13_raw = thrust::raw_pointer_cast(d_g13_d_j13_dev.data());
        auto d_g13_d_j21_raw = thrust::raw_pointer_cast(d_g13_d_j21_dev.data());
        auto d_g13_d_j22_raw = thrust::raw_pointer_cast(d_g13_d_j22_dev.data());
        auto d_g13_d_j23_raw = thrust::raw_pointer_cast(d_g13_d_j23_dev.data());
        auto d_g13_d_j31_raw = thrust::raw_pointer_cast(d_g13_d_j31_dev.data());
        auto d_g13_d_j32_raw = thrust::raw_pointer_cast(d_g13_d_j32_dev.data());
        auto d_g13_d_j33_raw = thrust::raw_pointer_cast(d_g13_d_j33_dev.data());

        auto d_g21_d_j11_raw = thrust::raw_pointer_cast(d_g21_d_j11_dev.data());
        auto d_g21_d_j12_raw = thrust::raw_pointer_cast(d_g21_d_j12_dev.data());
        auto d_g21_d_j13_raw = thrust::raw_pointer_cast(d_g21_d_j13_dev.data());
        auto d_g21_d_j21_raw = thrust::raw_pointer_cast(d_g21_d_j21_dev.data());
        auto d_g21_d_j22_raw = thrust::raw_pointer_cast(d_g21_d_j22_dev.data());
        auto d_g21_d_j23_raw = thrust::raw_pointer_cast(d_g21_d_j23_dev.data());
        auto d_g21_d_j31_raw = thrust::raw_pointer_cast(d_g21_d_j31_dev.data());
        auto d_g21_d_j32_raw = thrust::raw_pointer_cast(d_g21_d_j32_dev.data());
        auto d_g21_d_j33_raw = thrust::raw_pointer_cast(d_g21_d_j33_dev.data());

        auto d_g22_d_j11_raw = thrust::raw_pointer_cast(d_g22_d_j11_dev.data());
        auto d_g22_d_j12_raw = thrust::raw_pointer_cast(d_g22_d_j12_dev.data());
        auto d_g22_d_j13_raw = thrust::raw_pointer_cast(d_g22_d_j13_dev.data());
        auto d_g22_d_j21_raw = thrust::raw_pointer_cast(d_g22_d_j21_dev.data());
        auto d_g22_d_j22_raw = thrust::raw_pointer_cast(d_g22_d_j22_dev.data());
        auto d_g22_d_j23_raw = thrust::raw_pointer_cast(d_g22_d_j23_dev.data());
        auto d_g22_d_j31_raw = thrust::raw_pointer_cast(d_g22_d_j31_dev.data());
        auto d_g22_d_j32_raw = thrust::raw_pointer_cast(d_g22_d_j32_dev.data());
        auto d_g22_d_j33_raw = thrust::raw_pointer_cast(d_g22_d_j33_dev.data());

        auto d_g23_d_j11_raw = thrust::raw_pointer_cast(d_g23_d_j11_dev.data());
        auto d_g23_d_j12_raw = thrust::raw_pointer_cast(d_g23_d_j12_dev.data());
        auto d_g23_d_j13_raw = thrust::raw_pointer_cast(d_g23_d_j13_dev.data());
        auto d_g23_d_j21_raw = thrust::raw_pointer_cast(d_g23_d_j21_dev.data());
        auto d_g23_d_j22_raw = thrust::raw_pointer_cast(d_g23_d_j22_dev.data());
        auto d_g23_d_j23_raw = thrust::raw_pointer_cast(d_g23_d_j23_dev.data());
        auto d_g23_d_j31_raw = thrust::raw_pointer_cast(d_g23_d_j31_dev.data());
        auto d_g23_d_j32_raw = thrust::raw_pointer_cast(d_g23_d_j32_dev.data());
        auto d_g23_d_j33_raw = thrust::raw_pointer_cast(d_g23_d_j33_dev.data());

        auto d_g31_d_j11_raw = thrust::raw_pointer_cast(d_g31_d_j11_dev.data());
        auto d_g31_d_j12_raw = thrust::raw_pointer_cast(d_g31_d_j12_dev.data());
        auto d_g31_d_j13_raw = thrust::raw_pointer_cast(d_g31_d_j13_dev.data());
        auto d_g31_d_j21_raw = thrust::raw_pointer_cast(d_g31_d_j21_dev.data());
        auto d_g31_d_j22_raw = thrust::raw_pointer_cast(d_g31_d_j22_dev.data());
        auto d_g31_d_j23_raw = thrust::raw_pointer_cast(d_g31_d_j23_dev.data());
        auto d_g31_d_j31_raw = thrust::raw_pointer_cast(d_g31_d_j31_dev.data());
        auto d_g31_d_j32_raw = thrust::raw_pointer_cast(d_g31_d_j32_dev.data());
        auto d_g31_d_j33_raw = thrust::raw_pointer_cast(d_g31_d_j33_dev.data());

        auto d_g32_d_j11_raw = thrust::raw_pointer_cast(d_g32_d_j11_dev.data());
        auto d_g32_d_j12_raw = thrust::raw_pointer_cast(d_g32_d_j12_dev.data());
        auto d_g32_d_j13_raw = thrust::raw_pointer_cast(d_g32_d_j13_dev.data());
        auto d_g32_d_j21_raw = thrust::raw_pointer_cast(d_g32_d_j21_dev.data());
        auto d_g32_d_j22_raw = thrust::raw_pointer_cast(d_g32_d_j22_dev.data());
        auto d_g32_d_j23_raw = thrust::raw_pointer_cast(d_g32_d_j23_dev.data());
        auto d_g32_d_j31_raw = thrust::raw_pointer_cast(d_g32_d_j31_dev.data());
        auto d_g32_d_j32_raw = thrust::raw_pointer_cast(d_g32_d_j32_dev.data());
        auto d_g32_d_j33_raw = thrust::raw_pointer_cast(d_g32_d_j33_dev.data());

        auto d_g33_d_j11_raw = thrust::raw_pointer_cast(d_g33_d_j11_dev.data());
        auto d_g33_d_j12_raw = thrust::raw_pointer_cast(d_g33_d_j12_dev.data());
        auto d_g33_d_j13_raw = thrust::raw_pointer_cast(d_g33_d_j13_dev.data());
        auto d_g33_d_j21_raw = thrust::raw_pointer_cast(d_g33_d_j21_dev.data());
        auto d_g33_d_j22_raw = thrust::raw_pointer_cast(d_g33_d_j22_dev.data());
        auto d_g33_d_j23_raw = thrust::raw_pointer_cast(d_g33_d_j23_dev.data());
        auto d_g33_d_j31_raw = thrust::raw_pointer_cast(d_g33_d_j31_dev.data());
        auto d_g33_d_j32_raw = thrust::raw_pointer_cast(d_g33_d_j32_dev.data());
        auto d_g33_d_j33_raw = thrust::raw_pointer_cast(d_g33_d_j33_dev.data());

        // Calculate kernel parameters
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_rotation_gxx_grad_jxx_3d,
            0,
            0);
        unsigned int grid_size = (n_samples + block_size - 1)/block_size;
        // Call CUDA Kernel
        MMORF::kernel_rotation_gxx_grad_jxx_3d<<<grid_size, block_size>>>(
            // Input
            n_samples,
            g11_raw,
            g12_raw,
            g13_raw,
            g22_raw,
            g23_raw,
            g33_raw,
            r11_raw,
            r12_raw,
            r13_raw,
            r21_raw,
            r22_raw,
            r23_raw,
            r31_raw,
            r32_raw,
            r33_raw,
            s11_raw,
            s12_raw,
            s13_raw,
            s22_raw,
            s23_raw,
            s33_raw,
            // Output
            d_g11_d_j11_raw,
            d_g11_d_j12_raw,
            d_g11_d_j13_raw,
            d_g11_d_j21_raw,
            d_g11_d_j22_raw,
            d_g11_d_j23_raw,
            d_g11_d_j31_raw,
            d_g11_d_j32_raw,
            d_g11_d_j33_raw,
            d_g12_d_j11_raw,
            d_g12_d_j12_raw,
            d_g12_d_j13_raw,
            d_g12_d_j21_raw,
            d_g12_d_j22_raw,
            d_g12_d_j23_raw,
            d_g12_d_j31_raw,
            d_g12_d_j32_raw,
            d_g12_d_j33_raw,
            d_g13_d_j11_raw,
            d_g13_d_j12_raw,
            d_g13_d_j13_raw,
            d_g13_d_j21_raw,
            d_g13_d_j22_raw,
            d_g13_d_j23_raw,
            d_g13_d_j31_raw,
            d_g13_d_j32_raw,
            d_g13_d_j33_raw,
            d_g22_d_j11_raw,
            d_g22_d_j12_raw,
            d_g22_d_j13_raw,
            d_g22_d_j21_raw,
            d_g22_d_j22_raw,
            d_g22_d_j23_raw,
            d_g22_d_j31_raw,
            d_g22_d_j32_raw,
            d_g22_d_j33_raw,
            d_g23_d_j11_raw,
            d_g23_d_j12_raw,
            d_g23_d_j13_raw,
            d_g23_d_j21_raw,
            d_g23_d_j22_raw,
            d_g23_d_j23_raw,
            d_g23_d_j31_raw,
            d_g23_d_j32_raw,
            d_g23_d_j33_raw,
            d_g33_d_j11_raw,
            d_g33_d_j12_raw,
            d_g33_d_j13_raw,
            d_g33_d_j21_raw,
            d_g33_d_j22_raw,
            d_g33_d_j23_raw,
            d_g33_d_j31_raw,
            d_g33_d_j32_raw,
            d_g33_d_j33_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy calculated grad data back to host
        auto d_g11_d_j11_host = thrust::host_vector<float>(d_g11_d_j11_dev);
        auto d_g11_d_j12_host = thrust::host_vector<float>(d_g11_d_j12_dev);
        auto d_g11_d_j13_host = thrust::host_vector<float>(d_g11_d_j13_dev);
        auto d_g11_d_j21_host = thrust::host_vector<float>(d_g11_d_j21_dev);
        auto d_g11_d_j22_host = thrust::host_vector<float>(d_g11_d_j22_dev);
        auto d_g11_d_j23_host = thrust::host_vector<float>(d_g11_d_j23_dev);
        auto d_g11_d_j31_host = thrust::host_vector<float>(d_g11_d_j31_dev);
        auto d_g11_d_j32_host = thrust::host_vector<float>(d_g11_d_j32_dev);
        auto d_g11_d_j33_host = thrust::host_vector<float>(d_g11_d_j33_dev);

        auto d_g12_d_j11_host = thrust::host_vector<float>(d_g12_d_j11_dev);
        auto d_g12_d_j12_host = thrust::host_vector<float>(d_g12_d_j12_dev);
        auto d_g12_d_j13_host = thrust::host_vector<float>(d_g12_d_j13_dev);
        auto d_g12_d_j21_host = thrust::host_vector<float>(d_g12_d_j21_dev);
        auto d_g12_d_j22_host = thrust::host_vector<float>(d_g12_d_j22_dev);
        auto d_g12_d_j23_host = thrust::host_vector<float>(d_g12_d_j23_dev);
        auto d_g12_d_j31_host = thrust::host_vector<float>(d_g12_d_j31_dev);
        auto d_g12_d_j32_host = thrust::host_vector<float>(d_g12_d_j32_dev);
        auto d_g12_d_j33_host = thrust::host_vector<float>(d_g12_d_j33_dev);

        auto d_g13_d_j11_host = thrust::host_vector<float>(d_g13_d_j11_dev);
        auto d_g13_d_j12_host = thrust::host_vector<float>(d_g13_d_j12_dev);
        auto d_g13_d_j13_host = thrust::host_vector<float>(d_g13_d_j13_dev);
        auto d_g13_d_j21_host = thrust::host_vector<float>(d_g13_d_j21_dev);
        auto d_g13_d_j22_host = thrust::host_vector<float>(d_g13_d_j22_dev);
        auto d_g13_d_j23_host = thrust::host_vector<float>(d_g13_d_j23_dev);
        auto d_g13_d_j31_host = thrust::host_vector<float>(d_g13_d_j31_dev);
        auto d_g13_d_j32_host = thrust::host_vector<float>(d_g13_d_j32_dev);
        auto d_g13_d_j33_host = thrust::host_vector<float>(d_g13_d_j33_dev);

        auto d_g22_d_j11_host = thrust::host_vector<float>(d_g22_d_j11_dev);
        auto d_g22_d_j12_host = thrust::host_vector<float>(d_g22_d_j12_dev);
        auto d_g22_d_j13_host = thrust::host_vector<float>(d_g22_d_j13_dev);
        auto d_g22_d_j21_host = thrust::host_vector<float>(d_g22_d_j21_dev);
        auto d_g22_d_j22_host = thrust::host_vector<float>(d_g22_d_j22_dev);
        auto d_g22_d_j23_host = thrust::host_vector<float>(d_g22_d_j23_dev);
        auto d_g22_d_j31_host = thrust::host_vector<float>(d_g22_d_j31_dev);
        auto d_g22_d_j32_host = thrust::host_vector<float>(d_g22_d_j32_dev);
        auto d_g22_d_j33_host = thrust::host_vector<float>(d_g22_d_j33_dev);

        auto d_g23_d_j11_host = thrust::host_vector<float>(d_g23_d_j11_dev);
        auto d_g23_d_j12_host = thrust::host_vector<float>(d_g23_d_j12_dev);
        auto d_g23_d_j13_host = thrust::host_vector<float>(d_g23_d_j13_dev);
        auto d_g23_d_j21_host = thrust::host_vector<float>(d_g23_d_j21_dev);
        auto d_g23_d_j22_host = thrust::host_vector<float>(d_g23_d_j22_dev);
        auto d_g23_d_j23_host = thrust::host_vector<float>(d_g23_d_j23_dev);
        auto d_g23_d_j31_host = thrust::host_vector<float>(d_g23_d_j31_dev);
        auto d_g23_d_j32_host = thrust::host_vector<float>(d_g23_d_j32_dev);
        auto d_g23_d_j33_host = thrust::host_vector<float>(d_g23_d_j33_dev);

        auto d_g33_d_j11_host = thrust::host_vector<float>(d_g33_d_j11_dev);
        auto d_g33_d_j12_host = thrust::host_vector<float>(d_g33_d_j12_dev);
        auto d_g33_d_j13_host = thrust::host_vector<float>(d_g33_d_j13_dev);
        auto d_g33_d_j21_host = thrust::host_vector<float>(d_g33_d_j21_dev);
        auto d_g33_d_j22_host = thrust::host_vector<float>(d_g33_d_j22_dev);
        auto d_g33_d_j23_host = thrust::host_vector<float>(d_g33_d_j23_dev);
        auto d_g33_d_j31_host = thrust::host_vector<float>(d_g33_d_j31_dev);
        auto d_g33_d_j32_host = thrust::host_vector<float>(d_g33_d_j32_dev);
        auto d_g33_d_j33_host = thrust::host_vector<float>(d_g33_d_j33_dev);
        // Return cost data
        auto dG_dJ = std::vector<std::vector<std::vector<float> > >(
            6, std::vector<std::vector<float> >(
              9, std::vector<float>(
                n_samples)));

        thrust::copy(
            d_g11_d_j11_host.begin(),
            d_g11_d_j11_host.end(),
            dG_dJ[0][0].begin());
        thrust::copy(
            d_g11_d_j12_host.begin(),
            d_g11_d_j12_host.end(),
            dG_dJ[0][1].begin());
        thrust::copy(
            d_g11_d_j13_host.begin(),
            d_g11_d_j13_host.end(),
            dG_dJ[0][2].begin());
        thrust::copy(
            d_g11_d_j21_host.begin(),
            d_g11_d_j21_host.end(),
            dG_dJ[0][3].begin());
        thrust::copy(
            d_g11_d_j22_host.begin(),
            d_g11_d_j22_host.end(),
            dG_dJ[0][4].begin());
        thrust::copy(
            d_g11_d_j23_host.begin(),
            d_g11_d_j23_host.end(),
            dG_dJ[0][5].begin());
        thrust::copy(
            d_g11_d_j31_host.begin(),
            d_g11_d_j31_host.end(),
            dG_dJ[0][6].begin());
        thrust::copy(
            d_g11_d_j32_host.begin(),
            d_g11_d_j32_host.end(),
            dG_dJ[0][7].begin());
        thrust::copy(
            d_g11_d_j33_host.begin(),
            d_g11_d_j33_host.end(),
            dG_dJ[0][8].begin());

        thrust::copy(
            d_g12_d_j11_host.begin(),
            d_g12_d_j11_host.end(),
            dG_dJ[1][0].begin());
        thrust::copy(
            d_g12_d_j12_host.begin(),
            d_g12_d_j12_host.end(),
            dG_dJ[1][1].begin());
        thrust::copy(
            d_g12_d_j13_host.begin(),
            d_g12_d_j13_host.end(),
            dG_dJ[1][2].begin());
        thrust::copy(
            d_g12_d_j21_host.begin(),
            d_g12_d_j21_host.end(),
            dG_dJ[1][3].begin());
        thrust::copy(
            d_g12_d_j22_host.begin(),
            d_g12_d_j22_host.end(),
            dG_dJ[1][4].begin());
        thrust::copy(
            d_g12_d_j23_host.begin(),
            d_g12_d_j23_host.end(),
            dG_dJ[1][5].begin());
        thrust::copy(
            d_g12_d_j31_host.begin(),
            d_g12_d_j31_host.end(),
            dG_dJ[1][6].begin());
        thrust::copy(
            d_g12_d_j32_host.begin(),
            d_g12_d_j32_host.end(),
            dG_dJ[1][7].begin());
        thrust::copy(
            d_g12_d_j33_host.begin(),
            d_g12_d_j33_host.end(),
            dG_dJ[1][8].begin());

        thrust::copy(
            d_g13_d_j11_host.begin(),
            d_g13_d_j11_host.end(),
            dG_dJ[2][0].begin());
        thrust::copy(
            d_g13_d_j12_host.begin(),
            d_g13_d_j12_host.end(),
            dG_dJ[2][1].begin());
        thrust::copy(
            d_g13_d_j13_host.begin(),
            d_g13_d_j13_host.end(),
            dG_dJ[2][2].begin());
        thrust::copy(
            d_g13_d_j21_host.begin(),
            d_g13_d_j21_host.end(),
            dG_dJ[2][3].begin());
        thrust::copy(
            d_g13_d_j22_host.begin(),
            d_g13_d_j22_host.end(),
            dG_dJ[2][4].begin());
        thrust::copy(
            d_g13_d_j23_host.begin(),
            d_g13_d_j23_host.end(),
            dG_dJ[2][5].begin());
        thrust::copy(
            d_g13_d_j31_host.begin(),
            d_g13_d_j31_host.end(),
            dG_dJ[2][6].begin());
        thrust::copy(
            d_g13_d_j32_host.begin(),
            d_g13_d_j32_host.end(),
            dG_dJ[2][7].begin());
        thrust::copy(
            d_g13_d_j33_host.begin(),
            d_g13_d_j33_host.end(),
            dG_dJ[2][8].begin());

        thrust::copy(
            d_g22_d_j11_host.begin(),
            d_g22_d_j11_host.end(),
            dG_dJ[3][0].begin());
        thrust::copy(
            d_g22_d_j12_host.begin(),
            d_g22_d_j12_host.end(),
            dG_dJ[3][1].begin());
        thrust::copy(
            d_g22_d_j13_host.begin(),
            d_g22_d_j13_host.end(),
            dG_dJ[3][2].begin());
        thrust::copy(
            d_g22_d_j21_host.begin(),
            d_g22_d_j21_host.end(),
            dG_dJ[3][3].begin());
        thrust::copy(
            d_g22_d_j22_host.begin(),
            d_g22_d_j22_host.end(),
            dG_dJ[3][4].begin());
        thrust::copy(
            d_g22_d_j23_host.begin(),
            d_g22_d_j23_host.end(),
            dG_dJ[3][5].begin());
        thrust::copy(
            d_g22_d_j31_host.begin(),
            d_g22_d_j31_host.end(),
            dG_dJ[3][6].begin());
        thrust::copy(
            d_g22_d_j32_host.begin(),
            d_g22_d_j32_host.end(),
            dG_dJ[3][7].begin());
        thrust::copy(
            d_g22_d_j33_host.begin(),
            d_g22_d_j33_host.end(),
            dG_dJ[3][8].begin());

        thrust::copy(
            d_g23_d_j11_host.begin(),
            d_g23_d_j11_host.end(),
            dG_dJ[4][0].begin());
        thrust::copy(
            d_g23_d_j12_host.begin(),
            d_g23_d_j12_host.end(),
            dG_dJ[4][1].begin());
        thrust::copy(
            d_g23_d_j13_host.begin(),
            d_g23_d_j13_host.end(),
            dG_dJ[4][2].begin());
        thrust::copy(
            d_g23_d_j21_host.begin(),
            d_g23_d_j21_host.end(),
            dG_dJ[4][3].begin());
        thrust::copy(
            d_g23_d_j22_host.begin(),
            d_g23_d_j22_host.end(),
            dG_dJ[4][4].begin());
        thrust::copy(
            d_g23_d_j23_host.begin(),
            d_g23_d_j23_host.end(),
            dG_dJ[4][5].begin());
        thrust::copy(
            d_g23_d_j31_host.begin(),
            d_g23_d_j31_host.end(),
            dG_dJ[4][6].begin());
        thrust::copy(
            d_g23_d_j32_host.begin(),
            d_g23_d_j32_host.end(),
            dG_dJ[4][7].begin());
        thrust::copy(
            d_g23_d_j33_host.begin(),
            d_g23_d_j33_host.end(),
            dG_dJ[4][8].begin());

        thrust::copy(
            d_g33_d_j11_host.begin(),
            d_g33_d_j11_host.end(),
            dG_dJ[5][0].begin());
        thrust::copy(
            d_g33_d_j12_host.begin(),
            d_g33_d_j12_host.end(),
            dG_dJ[5][1].begin());
        thrust::copy(
            d_g33_d_j13_host.begin(),
            d_g33_d_j13_host.end(),
            dG_dJ[5][2].begin());
        thrust::copy(
            d_g33_d_j21_host.begin(),
            d_g33_d_j21_host.end(),
            dG_dJ[5][3].begin());
        thrust::copy(
            d_g33_d_j22_host.begin(),
            d_g33_d_j22_host.end(),
            dG_dJ[5][4].begin());
        thrust::copy(
            d_g33_d_j23_host.begin(),
            d_g33_d_j23_host.end(),
            dG_dJ[5][5].begin());
        thrust::copy(
            d_g33_d_j31_host.begin(),
            d_g33_d_j31_host.end(),
            dG_dJ[5][6].begin());
        thrust::copy(
            d_g33_d_j32_host.begin(),
            d_g33_d_j32_host.end(),
            dG_dJ[5][7].begin());
        thrust::copy(
            d_g33_d_j33_host.begin(),
            d_g33_d_j33_host.end(),
            dG_dJ[5][8].begin());

        return dG_dJ;
      }

      /// Calculate the gradient of the cost per spline coefficient
      arma::fvec calculate_grad_per_spline_(
          const std::vector<std::vector<std::vector<float> > >&
            grad_per_jacobian_element,
          const std::vector<std::vector<float> >&
            samples_error,
          const std::vector<std::vector<float> >&
            sample_positions_warped) const
      {
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate jte_sz
        auto warp_sz = warp_field_->get_parameter_size();
        auto jte_sz = warp_sz.first * warp_sz.second;
        // Make a device vector for storing the result
        auto jte_dev = thrust::device_vector<float>(jte_sz, 0.0f);
        // Loop over all positions in the local Jacobian
        for (auto row = 0; row < warp_sz.first; ++row){
          for (auto col = 0; col < warp_sz.first; ++col){
            auto element_id = (row * warp_sz.first) + col;
            auto samples_pre_mult = std::vector<float>(
                samples_error[0].size(), 0.0f);
            for (auto dt_elem = 0; dt_elem < 6; ++dt_elem){
              auto tmp_pre_mult = std::vector<float>(samples_pre_mult.size());
              // Pre-multiply the error and gradient terms
              auto tensor_elem = dt_elem; // First row of DT
              if (dt_elem > 4) tensor_elem += 3; // Third row of DT
              else if (dt_elem > 2) tensor_elem += 1; // Second row of DT
              std::transform(
                  samples_error[tensor_elem].begin(),
                  samples_error[tensor_elem].end(),
                  grad_per_jacobian_element[dt_elem][element_id].begin(),
                  tmp_pre_mult.begin(),
                  std::multiplies<float>()
                  );
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  tmp_pre_mult.begin(),
                  samples_pre_mult.begin(),
                  std::plus<float>()
                  );
              // Account for symmetry
              if (dt_elem == 1 || dt_elem == 2 || dt_elem == 4){
                std::transform(
                    samples_pre_mult.begin(),
                    samples_pre_mult.end(),
                    tmp_pre_mult.begin(),
                    samples_pre_mult.begin(),
                    std::plus<float>()
                    );
              }
            }
            // Pre-multiply by the Jacobian modulation
            std::transform(
                samples_pre_mult.begin(),
                samples_pre_mult.end(),
                samples_jac_det.begin(),
                samples_pre_mult.begin(),
                [](float a, float b) -> float { return 0.5f*(1.0f + b)*a; }
                );
            if (mask_ref_){
              auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_ref.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            if (mask_mov_){
              auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
              std::transform(
                  samples_pre_mult.begin(),
                  samples_pre_mult.end(),
                  samples_mask_mov.begin(),
                  samples_pre_mult.begin(),
                  std::multiplies<float>()
                  );
            }
            // Generate the correct differentiated splines
            auto x_deriv_order = col == 0 ? 1 : 0;
            auto y_deriv_order = col == 1 ? 1 : 0;
            auto z_deriv_order = col == 2 ? 1 : 0;
            auto spline_x = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                x_deriv_order);
            auto spline_y = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                y_deriv_order);
            auto spline_z = BASISFIELD::Spline1D<float>(
                spline_1D_.Order(),
                spline_1D_.KnotSpacing(),
                z_deriv_order);
            auto spline_x_vals = spline_as_vec_(
                spline_x,
                sample_resolution_[0],
                x_deriv_order);
            auto spline_y_vals = spline_as_vec_(
                spline_y,
                sample_resolution_[1],
                y_deriv_order);
            auto spline_z_vals = spline_as_vec_(
                spline_z,
                sample_resolution_[2],
                z_deriv_order);
            // Get correct index into jte_dev (x, then y, then z)
            auto jte_dev_ptr = thrust::raw_pointer_cast(
                &jte_dev[row * warp_sz.second]);
            auto temp_jte_dev = thrust::device_vector<float>(warp_sz.second, 0.0f);
            calculate_sub_grad_(
                // Input
                samples_pre_mult,
                sample_dimensions_,
                spline_x_vals,
                spline_y_vals,
                spline_z_vals,
                warp_field_->get_dimensions(),
                std::vector<int>(3, spline_1D_.KnotSpacing()),
                // Output
                temp_jte_dev);
            // Add subgrad to appropriate position
            auto jte_dev_begin_it = jte_dev.begin() + row * warp_sz.second;
            thrust::transform(
                temp_jte_dev.begin(),
                temp_jte_dev.end(),
                jte_dev_begin_it,
                jte_dev_begin_it,
                thrust::plus<float>());
          }
        }
        // Copy data to host
        auto jte_host = thrust::host_vector<float>(jte_dev);
        // Return result
        auto jte_return = arma::fvec(jte_host.data(), jte_host.size());
        return jte_return;
      }

      /// Calculate the Hessian of the cost per spline coefficient
      MMORF::SparseDiagonalMatrixTiled calculate_hess_per_spline_(
          const std::vector<std::vector<std::vector<float> > >&
            grad_per_jacobian_element,
          const std::vector<std::vector<float> >&
            sample_positions_warped) const
      {
        // Determine sizes of splines etc.
        auto spline_sz = spline_1D_.KernelSize();
        auto field_sz = grad_per_jacobian_element[0][0].size();
        // Create sparse tiled matrix to store sub-Hessian
        auto sparse_jtj = create_empty_jtj_(warp_field_->get_dimensions());
        // Get size of warp field
        auto warp_sz = warp_field_->get_parameter_size();
        // Calculate Jacobian determinant for current parametrisation
        auto samples_jac_det = warp_field_->jacobian_determinants(sample_positions_warp_);
        // Calculate the possible combinations of both gradient images and splines. There
        // should be 9 combinations for each of the 9 "sub-Hessians" leading to 81 in total.
        // There is however symmetry involved here. 3 of each of the 9s are redundant (or
        // are the transpose of each other) and therefore there should only be 36 unique
        // sub-hessians to calculate (which is still admittedly a lot).
        //
        // Loop over all warp directions
        for (auto row_hess = 0; row_hess < warp_sz.first; ++row_hess){
          // Loop over all warp directions again
          for (auto col_hess = 0; col_hess < warp_sz.first; ++col_hess){
            // Check if this is a unique sub-jtj - we will keep the main diagonal and below
            //if (col_hess > row_hess) continue;
            auto field_pre_mult = std::vector<std::vector<float> >(9,
                std::vector<float>(field_sz, 0));

            auto spline_x_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_y_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_z_row_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);

            auto spline_x_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_y_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);
            auto spline_z_col_vals = std::vector<float>(
                warp_sz.first*warp_sz.first*spline_sz, 0);

            auto loop_count = 0;
            // Loop over the 3 Jacobian elements that contribute to the row gradient
            for (auto row_elem = 0; row_elem < warp_sz.first; ++row_elem){
              // Loop over the 3 Jabians elements that contribute to the col gradient
              for (auto col_elem = 0; col_elem < warp_sz.first; ++col_elem){
                // Check if this is a unique combination of elements - we will keep the main
                // diagonal and below
                // if (col_elem > row_elem) continue;
                // Generate the correct differentiated splines
                // Note that the modulo (%) operator converts the "elem" value to the column
                // position of that element in the Jacobian matrix
                auto x_deriv_order_row = row_elem == 0 ? 1 : 0;
                auto y_deriv_order_row = row_elem == 1 ? 1 : 0;
                auto z_deriv_order_row = row_elem == 2 ? 1 : 0;

                auto x_deriv_order_col = col_elem == 0 ? 1 : 0;
                auto y_deriv_order_col = col_elem == 1 ? 1 : 0;
                auto z_deriv_order_col = col_elem == 2 ? 1 : 0;

                auto spline_x_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    x_deriv_order_row);
                auto spline_y_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    y_deriv_order_row);
                auto spline_z_row = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    z_deriv_order_row);

                auto spline_x_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    x_deriv_order_col);
                auto spline_y_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    y_deriv_order_col);
                auto spline_z_col = BASISFIELD::Spline1D<float>(
                    spline_1D_.Order(),
                    spline_1D_.KnotSpacing(),
                    z_deriv_order_col);

                auto spline_x_row_vals_tmp = spline_as_vec_(
                    spline_x_row,
                    sample_resolution_[0],
                    x_deriv_order_row);
                auto spline_y_row_vals_tmp = spline_as_vec_(
                    spline_y_row,
                    sample_resolution_[1],
                    y_deriv_order_row);
                auto spline_z_row_vals_tmp = spline_as_vec_(
                    spline_z_row,
                    sample_resolution_[2],
                    z_deriv_order_row);

                auto spline_x_col_vals_tmp = spline_as_vec_(
                    spline_x_col,
                    sample_resolution_[0],
                    x_deriv_order_col);
                auto spline_y_col_vals_tmp = spline_as_vec_(
                    spline_y_col,
                    sample_resolution_[1],
                    y_deriv_order_col);
                auto spline_z_col_vals_tmp = spline_as_vec_(
                    spline_z_col,
                    sample_resolution_[2],
                    z_deriv_order_col);

                std::copy(
                    spline_x_row_vals_tmp.begin(),
                    spline_x_row_vals_tmp.end(),
                    std::next(spline_x_row_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_y_row_vals_tmp.begin(),
                    spline_y_row_vals_tmp.end(),
                    std::next(spline_y_row_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_z_row_vals_tmp.begin(),
                    spline_z_row_vals_tmp.end(),
                    std::next(spline_z_row_vals.begin(), loop_count*spline_sz));

                std::copy(
                    spline_x_col_vals_tmp.begin(),
                    spline_x_col_vals_tmp.end(),
                    std::next(spline_x_col_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_y_col_vals_tmp.begin(),
                    spline_y_col_vals_tmp.end(),
                    std::next(spline_y_col_vals.begin(), loop_count*spline_sz));
                std::copy(
                    spline_z_col_vals_tmp.begin(),
                    spline_z_col_vals_tmp.end(),
                    std::next(spline_z_col_vals.begin(), loop_count*spline_sz));

                // Pre-multiply the gradient_per_jacobian_element_fields
                // Loop over tensor elements
                for (auto dt_elem = 0; dt_elem < 6; ++dt_elem){
                  auto tmp_pre_mult = std::vector<float>(
                      field_sz);
                  auto tensor_elem = dt_elem; // First row of DT
                  // Create temporary sparse tiled matrix for this tensor element
                  if (dt_elem > 4) tensor_elem += 3; // Third row of DT
                  else if (dt_elem > 2) tensor_elem += 1; // Second row of DT
                  std::transform(
                      grad_per_jacobian_element[dt_elem][row_hess*warp_sz.first + row_elem].begin(),
                      grad_per_jacobian_element[dt_elem][row_hess*warp_sz.first + row_elem].end(),
                      grad_per_jacobian_element[dt_elem][col_hess*warp_sz.first + col_elem].begin(),
                      tmp_pre_mult.begin(),
                      std::multiplies<float>()
                      );
                  std::transform(
                      tmp_pre_mult.begin(),
                      tmp_pre_mult.end(),
                      field_pre_mult[loop_count].begin(),
                      field_pre_mult[loop_count].begin(),
                      std::plus<float>()
                      );
                  // Account for symmetry
                  if (dt_elem == 1 || dt_elem == 2 || dt_elem == 4){
                    std::transform(
                        tmp_pre_mult.begin(),
                        tmp_pre_mult.end(),
                        field_pre_mult[loop_count].begin(),
                        field_pre_mult[loop_count].begin(),
                        std::plus<float>()
                        );
                  }
                }
                // Pre-multiply by the Jacobian modulation
                // NB!!! NB!!! Note that a and b are swapped around here relative to all the
                // other places that we use this lambda!
                std::transform(
                    samples_jac_det.begin(),
                    samples_jac_det.end(),
                    field_pre_mult[loop_count].begin(),
                    field_pre_mult[loop_count].begin(),
                    [](float b, float a) -> float { return 0.5f*(1.0f + b)*a; }
                    );
                if (mask_ref_){
                  auto samples_mask_ref = mask_ref_->sample(sample_positions_ref_);
                  std::transform(
                      samples_mask_ref.begin(),
                      samples_mask_ref.end(),
                      field_pre_mult[loop_count].begin(),
                      field_pre_mult[loop_count].begin(),
                      std::multiplies<float>()
                      );
                }
                if (mask_mov_){
                  auto samples_mask_mov = mask_mov_->sample(sample_positions_warped);
                  std::transform(
                      samples_mask_mov.begin(),
                      samples_mask_mov.end(),
                      field_pre_mult[loop_count].begin(),
                      field_pre_mult[loop_count].begin(),
                      std::multiplies<float>()
                      );
                }
                // Increase loop counter
                ++loop_count;
              }
            }
            // Call kernel for this combination of elements
            calculate_sub_jtj_non_symmetrical_(
              // Input
              field_pre_mult,
              sample_dimensions_,
              spline_x_row_vals,
              spline_y_row_vals,
              spline_z_row_vals,
              spline_x_col_vals,
              spline_y_col_vals,
              spline_z_col_vals,
              warp_field_->get_dimensions(),
              std::vector<int>(3, spline_1D_.KnotSpacing()),
              row_hess,
              col_hess,
              sparse_jtj);
          }
        }
        return sparse_jtj;
      }

      /// Calculate a sub-vector of Jte
      void calculate_sub_grad_(
          // Input
          const std::vector<float>&    grad_jxx_ima,
          const std::vector<int>&      ima_sz,
          const std::vector<float>&    spline_x,
          const std::vector<float>&    spline_y,
          const std::vector<float>&    spline_z,
          const std::vector<int>&      coef_sz,
          const std::vector<int>&      ksp,
          // Output
          thrust::device_vector<float>& sub_jte) const
      {
        // Make everything a device vector using thrust::
        auto grad_jxx_ima_dev = thrust::device_vector<float>(grad_jxx_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(grad_jxx_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        float *sub_jte_raw = thrust::raw_pointer_cast(sub_jte.data());
        // Calculate parameters for running kernel
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_make_jte,
            0,
            0);
        unsigned int grid_size =
          (coef_sz[0]*coef_sz[1]*coef_sz[2] + block_size - 1)/block_size;
        auto grid_1d = dim3(grid_size);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jte<<<grid_1d,block_size,smem>>>(
            // Input
            ima_sz[0],
            ima_sz[1],
            ima_sz[2],
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp[0],
            ksp[1],
            ksp[2],
            coef_sz[0],
            coef_sz[1],
            coef_sz[2],
            // Output
            sub_jte_raw);
        /// \todo Add error checking to cudaDeviceSynchronize
        checkCudaErrors(cudaDeviceSynchronize());
      }

      // Calculate the offsets for a sparse diagonal matrix
      std::vector<int> calculate_offsets_(const std::vector<int>& coef_sz) const
      {
        auto offsets = std::vector<int>(343,0);
#pragma omp parallel for
        for (unsigned int i = 0; i < 343; ++i)
        {
            unsigned int first_row, first_col, last_row, last_col;
            MMORF::identify_diagonal(i,7,7,7,coef_sz[0],coef_sz[1],coef_sz[2],
                                       &first_row,&last_row,&first_col,&last_col);
            if (first_col == 0){
              offsets.at(i) = -static_cast<int>(first_row);
            }
            else{
              offsets.at(i) = static_cast<int>(first_col);
            }
        }
        return offsets;
      }

      // Create an empty JtJ matrix. NB!!! In this case the matrix only contains the main
      // diagonal! And therefore we need a separate method to "calculate offsets" for what
      // the true Hessian would look like. This is annoying, but it's an unfortunate side-
      // effect of efficiency in the kernel.
      MMORF::SparseDiagonalMatrixTiled create_empty_jtj_(
          const std::vector<int>& coef_sz) const
      {
        auto offsets = std::vector<int>(1,0);
        unsigned int max_diagonal = coef_sz[0]*coef_sz[1]*coef_sz[2];
        MMORF::SparseDiagonalMatrixTiled r_matrix(max_diagonal, max_diagonal, 3, offsets);
        return r_matrix;
      }

      // Calclulate sub-matrix of jtj for symmetrical splines
      void calculate_sub_jtj_symmetrical_(
          const std::vector<float>& prod_ima,
          const std::vector<int>& ima_sz,
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>& coef_sz,
          const std::vector<int>& ksp,
          const unsigned int sub_jtj_row,
          const unsigned int sub_jtj_col,
          MMORF::SparseDiagonalMatrixTiled& sparse_jtj) const
      {
        // Calculate real Heassian offsets as sparse_jtj is now only a main diagonal matrix
        auto offsets = calculate_offsets_(coef_sz);
        // Make everything a device vector using thrust::
        auto sparse_jtj_offsets_dev = thrust::device_vector<int>(offsets);
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = MMORF::TextureHandleLinear(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        int *sparse_jtj_offsets_raw = thrust::raw_pointer_cast(
            sparse_jtj_offsets_dev.data());
        // Note that in this case we save everything to the main diagonal, so we just use the
        // sub_jtj_row for both the row and column parameter
        float *sparse_jtj_raw_1 = sparse_jtj.get_raw_pointer(sub_jtj_row, sub_jtj_row);
        float *sparse_jtj_raw_2 = sparse_jtj.get_raw_pointer(sub_jtj_col, sub_jtj_col);
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_make_jtj_symmetrical,
            0,
            0);
        unsigned int blocks = 172;
        unsigned int chunks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_2d(chunks,blocks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        // Check if this is on the main diagonal of the tiled matrix or not
        if (sub_jtj_row == sub_jtj_col){
          MMORF::kernel_make_jtj_symmetrical_diag_hess_main<<<blocks_2d,threads,smem>>>(
              // Input
              ima_sz.at(0),
              ima_sz.at(1),
              ima_sz.at(2),
              tex,
              spline_x_raw,
              spline_y_raw,
              spline_z_raw,
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              sparse_jtj_offsets_raw,
              // Output
              sparse_jtj_raw_1);
          /// \todo Add error checking to cudaDeviceSynchronize
          checkCudaErrors(cudaDeviceSynchronize());
        }
        else{
          MMORF::kernel_make_jtj_symmetrical_diag_hess_off<<<blocks_2d,threads,smem>>>(
              // Input
              ima_sz.at(0),
              ima_sz.at(1),
              ima_sz.at(2),
              tex,
              spline_x_raw,
              spline_y_raw,
              spline_z_raw,
              ksp.at(0),
              ksp.at(1),
              ksp.at(2),
              coef_sz.at(0),
              coef_sz.at(1),
              coef_sz.at(2),
              sparse_jtj_offsets_raw,
              // Output
              sparse_jtj_raw_1,
              sparse_jtj_raw_2);
          /// \todo Add error checking to cudaDeviceSynchronize
          checkCudaErrors(cudaDeviceSynchronize());
        }
      }

      // Calclulate sub-matrix of jtj for symmetrical splines
      void calculate_sub_jtj_non_symmetrical_(
          const std::vector<std::vector<float> >& prod_ima,
          const std::vector<int>&                 ima_sz,
          const std::vector<float>&               spline_x_1,
          const std::vector<float>&               spline_y_1,
          const std::vector<float>&               spline_z_1,
          const std::vector<float>&               spline_x_2,
          const std::vector<float>&               spline_y_2,
          const std::vector<float>&               spline_z_2,
          const std::vector<int>&                 coef_sz,
          const std::vector<int>&                 ksp,
          const unsigned int                      sub_jtj_row,
          const unsigned int                      sub_jtj_col,
          MMORF::SparseDiagonalMatrixTiled&       sparse_jtj) const
      {
        // Calculate real Heassian offsets as sparse_jtj is now only a main diagonal matrix
        auto offsets = calculate_offsets_(coef_sz);
        // Make everything a device vector using thrust::
        auto sparse_jtj_offsets_dev = thrust::device_vector<int>(offsets);
        auto prod_ima_1_dev = thrust::device_vector<float>(prod_ima[0]);
        auto prod_ima_2_dev = thrust::device_vector<float>(prod_ima[1]);
        auto prod_ima_3_dev = thrust::device_vector<float>(prod_ima[2]);
        auto prod_ima_4_dev = thrust::device_vector<float>(prod_ima[3]);
        auto prod_ima_5_dev = thrust::device_vector<float>(prod_ima[4]);
        auto prod_ima_6_dev = thrust::device_vector<float>(prod_ima[5]);
        auto prod_ima_7_dev = thrust::device_vector<float>(prod_ima[6]);
        auto prod_ima_8_dev = thrust::device_vector<float>(prod_ima[7]);
        auto prod_ima_9_dev = thrust::device_vector<float>(prod_ima[8]);
        auto spline_x_1_dev = thrust::device_vector<float>(spline_x_1);
        auto spline_y_1_dev = thrust::device_vector<float>(spline_y_1);
        auto spline_z_1_dev = thrust::device_vector<float>(spline_z_1);
        auto spline_x_2_dev = thrust::device_vector<float>(spline_x_2);
        auto spline_y_2_dev = thrust::device_vector<float>(spline_y_2);
        auto spline_z_2_dev = thrust::device_vector<float>(spline_z_2);
        // Create texture handle. Note that as we are dealing with all combinations of
        // Jacobian elements which contribute to this sub-JTJ simultaneously we have to make
        // a vector of 9 different textures
        auto texture_handle_1 = MMORF::TextureHandleLinear(prod_ima_1_dev,ima_sz);
        auto texture_handle_2 = MMORF::TextureHandleLinear(prod_ima_2_dev,ima_sz);
        auto texture_handle_3 = MMORF::TextureHandleLinear(prod_ima_3_dev,ima_sz);
        auto texture_handle_4 = MMORF::TextureHandleLinear(prod_ima_4_dev,ima_sz);
        auto texture_handle_5 = MMORF::TextureHandleLinear(prod_ima_5_dev,ima_sz);
        auto texture_handle_6 = MMORF::TextureHandleLinear(prod_ima_6_dev,ima_sz);
        auto texture_handle_7 = MMORF::TextureHandleLinear(prod_ima_7_dev,ima_sz);
        auto texture_handle_8 = MMORF::TextureHandleLinear(prod_ima_8_dev,ima_sz);
        auto texture_handle_9 = MMORF::TextureHandleLinear(prod_ima_9_dev,ima_sz);
        auto tex_dev = thrust::device_vector<cudaTextureObject_t>(9);
        tex_dev[0] = texture_handle_1.get_texture();
        tex_dev[1] = texture_handle_2.get_texture();
        tex_dev[2] = texture_handle_3.get_texture();
        tex_dev[3] = texture_handle_4.get_texture();
        tex_dev[4] = texture_handle_5.get_texture();
        tex_dev[5] = texture_handle_6.get_texture();
        tex_dev[6] = texture_handle_7.get_texture();
        tex_dev[7] = texture_handle_8.get_texture();
        tex_dev[8] = texture_handle_9.get_texture();
        cudaTextureObject_t *tex_raw = thrust::raw_pointer_cast(tex_dev.data());
        // Get all the raw pointers ready for the Kernel
        float *spline_x_1_raw = thrust::raw_pointer_cast(spline_x_1_dev.data());
        float *spline_y_1_raw = thrust::raw_pointer_cast(spline_y_1_dev.data());
        float *spline_z_1_raw = thrust::raw_pointer_cast(spline_z_1_dev.data());
        float *spline_x_2_raw = thrust::raw_pointer_cast(spline_x_2_dev.data());
        float *spline_y_2_raw = thrust::raw_pointer_cast(spline_y_2_dev.data());
        float *spline_z_2_raw = thrust::raw_pointer_cast(spline_z_2_dev.data());
        int *sparse_jtj_offsets_raw = thrust::raw_pointer_cast(
            sparse_jtj_offsets_dev.data());

        float *sparse_jtj_raw_1 = sparse_jtj.get_raw_pointer(sub_jtj_row, sub_jtj_row);
        // Calculate parameters for running kernel
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            MMORF::kernel_make_jtj_non_symmetrical_diag_hess_spred,
            0,
            0);
        unsigned int grid_size_diags = 343;
        unsigned int grid_size_chunks =
          (coef_sz[0]*coef_sz[1]*coef_sz[2] + block_size - 1)/block_size;
        auto grid_2d = dim3(grid_size_chunks, grid_size_diags);
        unsigned int smem =
          (spline_x_1.size() + spline_y_1.size() + spline_z_1.size()
          + spline_x_2.size() + spline_y_2.size() + spline_z_2.size())
          * sizeof(float);
        // Call CUDA Kernel
        MMORF::kernel_make_jtj_non_symmetrical_diag_hess_spred<<<grid_2d,block_size,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex_raw,
            spline_x_1_raw,
            spline_y_1_raw,
            spline_z_1_raw,
            spline_x_2_raw,
            spline_y_2_raw,
            spline_z_2_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            sparse_jtj_offsets_raw,
            // Output
            sparse_jtj_raw_1);
        checkCudaErrors(cudaDeviceSynchronize());
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      // Private datamembers
      std::shared_ptr<MMORF::VolumeTensor>     vol_ref_;
      std::shared_ptr<MMORF::VolumeTensor>     vol_mov_;
      arma::fmat                               affine_ref_;
      arma::fmat                               affine_mov_;
      std::shared_ptr<MMORF::Volume>           mask_ref_;
      std::shared_ptr<MMORF::Volume>           mask_mov_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      std::vector<int>                         sample_dimensions_;
      BASISFIELD::Spline1D<float>              spline_1D_;
      std::vector<std::vector<float> >         sample_positions_warp_;
      std::vector<std::vector<float> >         sample_positions_ref_;
      std::vector<float>                       sample_resolution_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::~CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess() = default;
  /// Move ctor
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess(CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess&& rhs) = default;
  /// Move assignment operator
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess& CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::operator=(CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess&& rhs) = default;
  /// Copy ctor
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess(const CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess& CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::operator=(const CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
    else{
      *pimpl_ = *rhs.pimpl_;
    }
    return *this;
  }
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
  /// Construct by passing in fully constructed volumes, and defining warp field
  /// characteristics
  /// \param vol_ref Reference (stationary) volume
  /// \param vol_mov Transformed volume
  /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
  ///                   reference space
  /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
  ///                   moving space
  /// \param mask_ref 3D volume used to mask the reference volume
  /// \param mask_mov 3D volume used to mask the moving volume
  /// \param knot_spacing Warp field B-spline knot spacing (mm) in reference space
  /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
  ///             knot_spacing        = 10mm
  ///             sampling_frequency  = 5
  ///        warp field will be sampled every 2mm
  CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess(
          std::shared_ptr<MMORF::VolumeTensor>     vol_ref,
          std::shared_ptr<MMORF::VolumeTensor>     vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::Volume>           mask_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency
          )
    : pimpl_(MMORF::make_unique<Impl>(
          vol_ref,
          vol_mov,
          affine_ref,
          affine_mov,
          mask_ref,
          mask_mov,
          warp_field,
          sampling_frequency)
        )
  {}
  /// Get the current value of the parameters
  std::vector<std::vector<float> > CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the current value of the parameters
  void CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get cost under current parameterisation
  float CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::cost() const
  {
    return pimpl_->cost();
  }
  /// Get Jte under current parameterisation
  arma::fvec CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::grad() const
  {
    return pimpl_->grad();
  }
  /// Get JtJ under current parameterisation
  MMORF::SparseDiagonalMatrixTiled CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess::hess() const
  {
    return pimpl_->hess();
  }
} // MMORF
