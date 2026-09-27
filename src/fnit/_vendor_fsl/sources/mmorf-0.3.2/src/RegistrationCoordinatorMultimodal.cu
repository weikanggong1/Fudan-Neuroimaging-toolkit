//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Coordinate the exection of a volumetric registration involving both scalar and
///        tensor volumes.
/// \details This class is responsible for coordinating the objects required to follow a
///          particular registration "recipe". This involves things like iterating over
///          different levels of smoothing, regularisation, etc.
/// \author Frederik Lange
/// \date November 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif
#undef NDEBUG
#include "MMORFio.h"
#include "MmorfMemory.h"
#include "RegistrationCoordinatorMultimodal.cuh"
#include "VolumeBSpline.cuh"
#include "VolumeTensor.cuh"
#include "WarpFieldBSpline.cuh"
#include "BiasFieldBSpline.cuh"
#include "IntensityMapperPolynomial.cuh"
#include "CostFxn.h"
#include "CostFxnSSDWarpFieldSymmetricMaskedExcluded.cuh"
#include "CostFxnSSDWarpFieldSymmetricMaskedExcludedDiagHess.cuh"
#include "CostFxnSSDBiasFieldMaskedExcluded.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "CostFxnBendingEnergyBiasField.cuh"
#include "CostFxnLogJacobianSingularValues.cuh"
#include "CostFxnLogJacobianSingularValuesDiagHess.cuh"
#include "CostFxnTensorL2WarpFieldSymmetricMaskedExcluded.cuh"
#include "CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess.cuh"
#include "CostFxnCompoundWarpFieldVarianceScaled.cuh"
#include "CostFxnCompoundBiasFieldVarianceScaled.cuh"
#include "Optimiser.h"
#include "OptimiserLevenberg.cuh"
#include "OptimiserScaledConjugateGradient.cuh"
#include "LinearSolverCusp.cuh"

#include "newimage/newimageall.h"

#include <armadillo>

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class RegistrationCoordinatorMultimodal::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Complete ctor
      Impl(
          // Warp defining parameters
          const float                             warp_res_init,
          const std::vector<int>&                 warp_scaling,
          const std::string&                      f_img_warp_space,
          // Scalar image parameters
          const std::vector<std::string>&         f_img_ref_scalar,
          const std::vector<std::string>&         f_img_mov_scalar,
          const std::vector<std::string>&         f_aff_ref_scalar,
          const std::vector<std::string>&         f_aff_mov_scalar,
          const std::vector<bool>&                use_implicit_mask,
          const std::vector<std::string>&         f_mask_ref_scalar,
          const std::vector<std::string>&         f_mask_mov_scalar,
          const std::vector<std::vector<bool> >&  use_mask_ref_scalar,
          const std::vector<std::vector<bool> >&  use_mask_mov_scalar,
          const std::vector<std::vector<float> >& fwhm_ref_scalar,
          const std::vector<std::vector<float> >& fwhm_mov_scalar,
          const std::vector<std::vector<float> >& lambda_scalar,
          // Tensor image parameters
          const std::vector<std::string>&         f_img_ref_tensor,
          const std::vector<std::string>&         f_img_mov_tensor,
          const std::vector<std::string>&         f_aff_ref_tensor,
          const std::vector<std::string>&         f_aff_mov_tensor,
          const std::vector<std::string>&         f_mask_ref_tensor,
          const std::vector<std::string>&         f_mask_mov_tensor,
          const std::vector<std::vector<bool> >&  use_mask_ref_tensor,
          const std::vector<std::vector<bool> >&  use_mask_mov_tensor,
          const std::vector<std::vector<float> >& fwhm_ref_tensor,
          const std::vector<std::vector<float> >& fwhm_mov_tensor,
          const std::vector<std::vector<float> >& lambda_tensor,
          // Regularisation parameters
          const std::vector<float>&               lambda_reg,
          // Optimiser parameters
          const float                             hires,
          const MMORF::OptimiserType              optimiser_lowres,
          const int                               optimiser_max_it_lowres,
          const float                             optimiser_rel_tol_lowres,
          const MMORF::OptimiserType              optimiser_hires,
          const int                               optimiser_max_it_hires,
          const float                             optimiser_rel_tol_hires,
          // Solver parameters
          const int                               solver_max_it_lowres,
          const float                             solver_rel_tol_lowres,
          const int                               solver_max_it_hires,
          const float                             solver_rel_tol_hires,
          // Bias field parameters
          const std::vector<bool>&                bias_estimate,
          const std::vector<float>&               bias_res_init,
          const std::vector<std::vector<float> >& lambda_reg_bias,
          // Debugging option
          const bool                              debug
          // Miscelaneous parameters
          //const bool                              affines_are_inverted
          )
        : n_iterations_(warp_scaling.size())
        , n_img_scalar_(f_img_ref_scalar.size())
        , n_img_tensor_(f_img_ref_tensor.size())
        , warp_scaling_(warp_scaling)
        , f_img_warp_space_(f_img_warp_space)

        , fwhm_ref_scalar_(fwhm_ref_scalar)
        , fwhm_mov_scalar_(fwhm_mov_scalar)
        , fwhm_ref_tensor_(fwhm_ref_tensor)
        , fwhm_mov_tensor_(fwhm_mov_tensor)

        , use_implicit_mask_(use_implicit_mask)
        , use_mask_ref_scalar_(use_mask_ref_scalar)
        , use_mask_mov_scalar_(use_mask_mov_scalar)
        , use_mask_ref_tensor_(use_mask_ref_tensor)
        , use_mask_mov_tensor_(use_mask_mov_tensor)

        , lambda_scalar_(lambda_scalar)
        , lambda_tensor_(lambda_tensor)
        , lambda_reg_(lambda_reg)

        , hires_(hires)
        , optimiser_lowres_(optimiser_lowres)
        , optimiser_max_it_lowres_(optimiser_max_it_lowres)
        , optimiser_rel_tol_lowres_(optimiser_rel_tol_lowres)
        , optimiser_hires_(optimiser_hires)
        , optimiser_max_it_hires_(optimiser_max_it_hires)
        , optimiser_rel_tol_hires_(optimiser_rel_tol_hires)

        , solver_max_it_lowres_(solver_max_it_lowres)
        , solver_rel_tol_lowres_(solver_rel_tol_lowres)
        , solver_max_it_hires_(solver_max_it_hires)
        , solver_rel_tol_hires_(solver_rel_tol_hires)

        , b_bias_estimate_(bias_estimate)
        , bias_res_init_(bias_res_init)
        , lambda_reg_bias_(lambda_reg_bias)

        , debug_(debug)
      {
        /// \todo Replace asserts with exceptions
        // Check that the number of volumes match
        assert(f_img_ref_scalar.size() == n_img_scalar_
            && f_img_mov_scalar.size() == n_img_scalar_
            && f_aff_ref_scalar.size() == n_img_scalar_
            && f_aff_mov_scalar.size() == n_img_scalar_
            && f_mask_ref_scalar.size() == n_img_scalar_
            && f_mask_mov_scalar.size() == n_img_scalar_
            && fwhm_ref_scalar.size() == n_img_scalar_
            && fwhm_mov_scalar.size() == n_img_scalar_
            && use_implicit_mask.size() == n_img_scalar_
            && use_mask_ref_scalar.size() == n_img_scalar_
            && use_mask_mov_scalar.size() == n_img_scalar_
            && lambda_scalar.size() == n_img_scalar_
            && bias_estimate.size() == n_img_scalar_
            && bias_res_init.size() == n_img_scalar_
            && lambda_reg_bias.size() == n_img_scalar_);
        assert(f_img_ref_tensor.size() == n_img_tensor_
            && f_img_mov_tensor.size() == n_img_tensor_
            && f_aff_ref_tensor.size() == n_img_tensor_
            && f_aff_mov_tensor.size() == n_img_tensor_
            && f_mask_ref_tensor.size() == n_img_tensor_
            && f_mask_mov_tensor.size() == n_img_tensor_
            && use_mask_ref_tensor.size() == n_img_tensor_
            && use_mask_mov_tensor.size() == n_img_tensor_
            && fwhm_ref_tensor.size() == n_img_tensor_
            && fwhm_mov_tensor.size() == n_img_tensor_
            && lambda_tensor.size() == n_img_tensor_);
        // Check that the number of iterations match
        if(n_img_scalar_ != 0){
          assert(use_mask_ref_scalar[0].size() == n_iterations_
              && use_mask_mov_scalar[0].size() == n_iterations_
              && fwhm_ref_scalar[0].size() == n_iterations_
              && fwhm_mov_scalar[0].size() == n_iterations_
              && lambda_scalar[0].size() == n_iterations_
              && lambda_reg_bias[0].size() == n_iterations_);
        }
        if(n_img_tensor_ != 0){
          assert(use_mask_ref_tensor[0].size() == n_iterations_
              && use_mask_mov_tensor[0].size() == n_iterations_
              && fwhm_ref_tensor[0].size() == n_iterations_
              && fwhm_mov_tensor[0].size() == n_iterations_
              && lambda_tensor[0].size() == n_iterations_);
        }
        assert(lambda_reg.size() == n_iterations_);
        // Read in warp space information
        NEWIMAGE::read_volume_hdr_only(hdr_warp_space_, f_img_warp_space);
        // Create a warp field which covers the reference space
        create_warp_field_(f_img_warp_space, warp_res_init);
        // Create a bias field which covers the reference space
        create_bias_field_(f_img_warp_space, bias_res_init);
        // Read reference volumes
        read_img_ref_scalar_(f_img_ref_scalar);
        read_img_ref_tensor_(f_img_ref_tensor);
        // Read moving volumes
        read_img_mov_scalar_(f_img_mov_scalar);
        read_img_mov_tensor_(f_img_mov_tensor);
        // Read mask volumes
        read_mask_ref_scalar_(f_mask_ref_scalar);
        read_mask_mov_scalar_(f_mask_mov_scalar);
        read_mask_ref_tensor_(f_mask_ref_tensor);
        read_mask_mov_tensor_(f_mask_mov_tensor);
        // Create affine matrices
        create_affine_matrices_scalar_(
            f_aff_ref_scalar,
            f_aff_mov_scalar,
            true);
        create_affine_matrices_tensor_(
            f_aff_ref_tensor,
            f_aff_mov_tensor,
            true);
        // Calculate the maximum resolution across all volumes for use in regularisation
        max_res_ = get_max_resolution_all_();
        if (debug_){
          auto tmp_debug_dir = "mkdir -p " + debug_dir_;
          auto tmp_sysval = std::system(tmp_debug_dir.c_str());
        }
      }
      /// Run through the registration process
      void register_volumes()
      {
        omp_set_num_threads(6);
        std::cout
          << "##### MAXIMUM THREADS AVAILABLE IS: "
          << omp_get_max_threads()
          << " #####"
          << std::endl;
        #pragma omp parallel
        {
          #pragma omp single
          {
                std::cout << "##### THREADS BEING USED IS: "
                  << omp_get_num_threads()
                  << " #####"
                  << std::endl;
          }
        }
        auto timer = arma::wall_clock();
        timer.tic();
        auto optimised_parameters = warp_field_->get_parameters();
        // Loop over iterations
        for (auto i_iteration = 0; i_iteration < n_iterations_; ++i_iteration){
          // Scale warp field if necessary
          if (warp_scaling_[i_iteration] > 1){
            auto extents_old = warp_field_->get_extents();
            warp_field_->reparameterise(warp_scaling_[i_iteration]);
            warp_field_->crop(extents_warp_space_);
            auto extents_new = warp_field_->get_extents();
            std::cout
              << "Extents Old: ["
              << extents_old.first[0]
              << " "
              << extents_old.first[1]
              << " "
              << extents_old.first[2]
              << "]"
              << " : ["
              << extents_old.second[0]
              << " "
              << extents_old.second[1]
              << " "
              << extents_old.second[2]
              << "]"
              << std::endl
              << "Extents New: ["
              << extents_new.first[0]
              << " "
              << extents_new.first[1]
              << " "
              << extents_new.first[2]
              << "]"
              << " : ["
              << extents_new.second[0]
              << " "
              << extents_new.second[1]
              << " "
              << extents_new.second[2]
              << "]"
              << std::endl;
          }
          // Create smoothed versions of volumes for this iteration
          auto ref_scalar_vols = create_ref_scalar_vols_(i_iteration);
          auto mov_scalar_vols = create_mov_scalar_vols_(i_iteration);
          auto ref_tensor_vols = create_ref_tensor_vols_(i_iteration);
          auto mov_tensor_vols = create_mov_tensor_vols_(i_iteration);
          // Estimate bias field
          estimate_bias_field_(
            i_iteration,
            ref_scalar_vols,
            mov_scalar_vols);
          auto knot_spacing = warp_field_->get_knot_spacing();
          // Get initial parameters to optimise over
          optimised_parameters = warp_field_->get_parameters();
          // Create individual cost functions
          auto cost_fxns = create_cost_fxns_(
            i_iteration,
            knot_spacing,
            ref_scalar_vols,
            mov_scalar_vols,
            ref_tensor_vols,
            mov_tensor_vols);
          // Create regulariser
          auto regulariser = create_reg_(i_iteration, knot_spacing);
          // Combine lambdas
          auto lambda_combined = std::vector<float>();
          for (auto i_s = 0; i_s < n_img_scalar_; ++i_s){
            lambda_combined.push_back(lambda_scalar_[i_s][i_iteration]);
          }
          for (auto i_t = 0; i_t < n_img_tensor_; ++i_t){
            lambda_combined.push_back(lambda_tensor_[i_t][i_iteration]);
          }
          // Create the single compound cost function
          auto cost_fxn_compound = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
              cost_fxns,
              lambda_combined,
              regulariser,
              lambda_reg_[i_iteration],
              warp_field_);
          // Create linear solver
          auto linear_solver = std::shared_ptr<MMORF::LinearSolverCusp>(nullptr);
          // Choose what type of optimiser to use
          auto optimiser = std::unique_ptr<MMORF::Optimiser>(nullptr);
          auto max_iter = 0;
          auto rel_tol = 0.0f;
          if (knot_spacing >= hires_){
            linear_solver = std::make_shared<MMORF::LinearSolverCusp>(
                solver_max_it_lowres_,
                solver_rel_tol_lowres_);
            switch (optimiser_lowres_){
              case MMORF::OptimiserType::LM:
                optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
                break;
              case MMORF::OptimiserType::MM:
                optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
                break;
              case MMORF::OptimiserType::SCG:
                optimiser = MMORF::make_unique<MMORF::OptimiserScaledConjugateGradient>();
                break;
              default:
                std::cout<< "This should never happen!" << std::endl;
                break;
            }
            max_iter = optimiser_max_it_lowres_;
            rel_tol = optimiser_rel_tol_lowres_;
          }
          else{
            linear_solver = std::make_shared<MMORF::LinearSolverCusp>(
                solver_max_it_hires_,
                solver_rel_tol_hires_);
            switch (optimiser_hires_){
              case MMORF::OptimiserType::LM:
                optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
                break;
              case MMORF::OptimiserType::MM:
                optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
                break;
              case MMORF::OptimiserType::SCG:
                optimiser = MMORF::make_unique<MMORF::OptimiserScaledConjugateGradient>();
                break;
              default:
                std::cout<< "This should never happen!" << std::endl;
                break;
            }
            max_iter = optimiser_max_it_hires_;
            rel_tol = optimiser_rel_tol_hires_;
          }
          // Solve system
          optimised_parameters = optimiser->optimise(
              cost_fxn_compound,
              warp_field_->get_parameters(),
              max_iter,
              rel_tol);
          // Update parameters of warp field
          warp_field_->set_parameters(optimised_parameters);

          if (debug_){
            // Create folder
            auto debug_dir = debug_dir_ + "iter_" + std::to_string(i_iteration) + "/";
            auto mkdir_command =
              "mkdir -p " +
              debug_dir;
            auto tmp_sysval = std::system(mkdir_command.c_str());
            save_warp_field(debug_dir + "warp");
            save_jacobian_determinants(debug_dir + "jac");
            save_warped_volumes(debug_dir + "vol");
            save_affine_transformed_volumes(debug_dir + "aff");
            save_bias_field(debug_dir + "bias");
          }
        }
        estimate_bias_field_no_smoothing_();
        auto runtime = timer.toc();
        std::cout << "\n\n REGISTRATION TOOK " << runtime << " SECONDS" << std::endl;
      }

      /// Save warp field as 4D nifti
      void save_warp_field(std::string filename_warp_field)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(f_img_warp_space_);
        auto samples_vec = warp_field_->sample_warp(
            warp_space_vol.get_original_sample_positions());
        auto n_samples = samples_vec[0].size();
        auto mm_to_vox_rot = warp_space_vol.get_mm_to_vox_rotation();
        auto samples_arma = arma::fmat(3, n_samples);
        for (auto dim = 0; dim < 3; ++dim){
          samples_arma.row(dim) = arma::frowvec(samples_vec[dim]);
        }
        samples_arma = mm_to_vox_rot * samples_arma;
        for (auto dim = 0; dim < 3; ++dim){
          samples_vec[dim] = arma::conv_to<std::vector<float> >::from(samples_arma.row(dim));
        }
        MMORF::save_as_nifti(
            samples_vec,
            warp_space_vol.get_dimensions(),
            hdr_warp_space_,
            filename_warp_field);
      }

      /// Save bias_field
      void save_bias_field(std::string bias_prefix)
      {
        auto bias_space_vol = MMORF::VolumeBSpline(f_img_warp_space_);
        for (auto i = 0; i < n_img_scalar_; ++i){
          if (b_bias_estimate_[i]){
            auto bias_samples = bias_field_[i]->sample_bias(
                bias_space_vol.get_original_sample_positions());
            MMORF::save_as_nifti(
                bias_samples,
                bias_space_vol.get_dimensions(),
                hdr_warp_space_,
                bias_prefix + "_" + std::to_string(i));
          }
        }
      }

      /// Save warped moving volumes
      void save_warped_volumes(std::string vol_prefix)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(f_img_warp_space_);
        for (auto i = 0; i < n_img_scalar_; ++i){
          auto vol_mov = MMORF::VolumeBSpline(img_mov_scalar_[i]);
          auto sample_positions = warp_field_->apply_warp_then_affine(
              warp_space_vol.get_original_sample_positions(), aff_mov_scalar_[i]);
          MMORF::save_as_nifti(
              vol_mov,
              sample_positions,
              warp_space_vol.get_dimensions(),
              hdr_warp_space_,
              vol_prefix + "_" + std::to_string(i));
        }
      }

      /// Save affine transformed volumes
      void save_affine_transformed_volumes(std::string vol_prefix)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(f_img_warp_space_);
        for (auto i = 0; i < n_img_scalar_; ++i){
          auto vol_mov = MMORF::VolumeBSpline(img_mov_scalar_[i]);
          auto sample_positions = warp_space_vol.get_original_sample_positions();
          auto sample_positions_arma = arma::fmat(3, sample_positions[0].size());
          sample_positions_arma.row(0) = arma::frowvec(sample_positions[0]);
          sample_positions_arma.row(1) = arma::frowvec(sample_positions[1]);
          sample_positions_arma.row(2) = arma::frowvec(sample_positions[2]);
          arma::fmat affine_positions_arma = arma::affmul(
              aff_mov_scalar_[i],
              sample_positions_arma);
          sample_positions[0] = arma::conv_to<std::vector<float> >::from(
              affine_positions_arma.row(0));
          sample_positions[1] = arma::conv_to<std::vector<float> >::from(
              affine_positions_arma.row(1));
          sample_positions[2] = arma::conv_to<std::vector<float> >::from(
              affine_positions_arma.row(2));
          MMORF::save_as_nifti(
              vol_mov,
              sample_positions,
              warp_space_vol.get_dimensions(),
              hdr_warp_space_,
              vol_prefix + "_" + std::to_string(i));
        }
      }

      /// Save Jacobian determinant of the warp field
      void save_jacobian_determinants(std::string filename_jacobian)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(f_img_warp_space_);
        auto jacobian =
          warp_field_->jacobian_determinants(warp_space_vol.get_original_sample_positions());
        MMORF::save_as_nifti(
            jacobian,
            warp_space_vol.get_dimensions(),
            hdr_warp_space_,
            filename_jacobian);
      }

      /// Save full extents of warp field
      void save_warp_field_full(std::string f_warp)
      {
        MMORF::save_full_warp_as_nifti(
            *warp_field_,
            1.0f,
            f_warp);
      }

      /// Save full extents of volumes
      void save_warped_volumes_full(std::string f_vols)
      {
        for (auto i = 0; i < n_img_scalar_; ++i){
          auto vol_mov = MMORF::VolumeBSpline(img_mov_scalar_[i]);
          MMORF::save_full_vol_as_nifti(
              *warp_field_,
              vol_mov,
              aff_mov_scalar_[i],
              1.0f,
              f_vols + "_" + std::to_string(i));
        }
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Create a warp field covering the warp space
      void create_warp_field_(
          const std::string& f_img_warp_space,
          const float        knot_spacing)
      {
        auto reference_volume = MMORF::VolumeBSpline(f_img_warp_space);
        extents_warp_space_ = reference_volume.get_extents();
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            extents_warp_space_,
            knot_spacing);
      }

      /// Create a bias field covering the warp space
      void create_bias_field_(
          const std::string&        f_img_warp_space,
          const std::vector<float>  bias_res_init)
      {
        auto reference_volume = MMORF::VolumeBSpline(f_img_warp_space);
        auto extents_bias_space_ = reference_volume.get_extents();

        for (auto i = 0; i < n_img_scalar_; ++i){
          if (b_bias_estimate_[i]){
            bias_field_.push_back(
                std::make_shared<MMORF::BiasFieldBSpline>(
                  extents_bias_space_,
                  bias_res_init[i]));
          }
          else{
            bias_field_.push_back(std::shared_ptr<MMORF::BiasFieldBSpline>(nullptr));
          }
        }
      }

      /// Read in scalar reference newimage volumes
      void read_img_ref_scalar_(const std::vector<std::string>& f_img_ref_scalar)
      {
        for (const auto& filename : f_img_ref_scalar){
          auto volume = NEWIMAGE::volume<float>();
          NEWIMAGE::read_volume(volume, filename);
          img_ref_scalar_.push_back(volume);
        }
      }

      /// Read in tensor reference newimage volumes
      void read_img_ref_tensor_(const std::vector<std::string>& f_img_ref_tensor)
      {
        for (const auto& filename : f_img_ref_tensor){
          auto volume = NEWIMAGE::volume4D<float>();
          NEWIMAGE::read_volume4D(volume, filename);
          img_ref_tensor_.push_back(volume);
        }
      }

      /// Read in scalar moving newimage volumes
      void read_img_mov_scalar_(const std::vector<std::string>& f_img_mov_scalar)
      {
        for (const auto& filename : f_img_mov_scalar){
          auto volume = NEWIMAGE::volume<float>();
          NEWIMAGE::read_volume(volume, filename);
          img_mov_scalar_.push_back(volume);
        }
      }

      /// Read in tensor moving newimage volumes
      void read_img_mov_tensor_(const std::vector<std::string>& f_img_mov_tensor)
      {
        for (const auto& filename : f_img_mov_tensor){
          auto volume = NEWIMAGE::volume4D<float>();
          NEWIMAGE::read_volume4D(volume, filename);
          img_mov_tensor_.push_back(volume);
        }
      }

      /// Read in scalar reference mask newimage volumes
      void read_mask_ref_scalar_(const std::vector<std::string>& f_mask_ref_scalar)
      {
        for (const auto& filename : f_mask_ref_scalar){
          if (filename != "NULL"){
            auto volume = std::make_shared<NEWIMAGE::volume<float> >();
            NEWIMAGE::read_volume(*volume, filename);
            mask_ref_scalar_.push_back(volume);
          }
          else{
            mask_ref_scalar_.push_back(std::shared_ptr<NEWIMAGE::volume<float> >(nullptr));
          }
        }
      }

      /// Read in scalar moving mask newimage volumes
      void read_mask_mov_scalar_(const std::vector<std::string>& f_mask_mov_scalar)
      {
        for (const auto& filename : f_mask_mov_scalar){
          if (filename != "NULL"){
            auto volume = std::make_shared<NEWIMAGE::volume<float> >();
            NEWIMAGE::read_volume(*volume, filename);
            mask_mov_scalar_.push_back(volume);
          }
          else{
            mask_mov_scalar_.push_back(std::shared_ptr<NEWIMAGE::volume<float> >(nullptr));
          }
        }
      }

      /// Read in tensor reference mask newimage volumes
      void read_mask_ref_tensor_(const std::vector<std::string>& f_mask_ref_tensor)
      {
        for (const auto& filename : f_mask_ref_tensor){
          if (filename != "NULL"){
            auto volume = std::make_shared<NEWIMAGE::volume<float> >();
            NEWIMAGE::read_volume(*volume, filename);
            mask_ref_tensor_.push_back(volume);
          }
          else{
            mask_ref_tensor_.push_back(std::shared_ptr<NEWIMAGE::volume<float> >(nullptr));
          }
        }
      }

      /// Read in tensor moving mask newimage volumes
      void read_mask_mov_tensor_(const std::vector<std::string>& f_mask_mov_tensor)
      {
        for (const auto& filename : f_mask_mov_tensor){
          if (filename != "NULL"){
            auto volume = std::make_shared<NEWIMAGE::volume<float> >();
            NEWIMAGE::read_volume(*volume, filename);
            mask_mov_tensor_.push_back(volume);
          }
          else{
            mask_mov_tensor_.push_back(std::shared_ptr<NEWIMAGE::volume<float> >(nullptr));
          }
        }
      }

      /// \details We are making the assumption that the matrices were calculated using FLIRT,
      ///          and therefore describe a transform between the volumes in "scaled mm
      ///          space". In MMORF we make the assumption that all coordinates and positions
      ///          are given in "real world mm space". As such, there is some serious
      ///          shuffling that needs to be done in order to get the FLIRT affine matrices
      ///          into a form that works with MMORF's coordinate system.
      void create_affine_matrices_scalar_(
          const std::vector<std::string>& f_aff_ref_scalar,
          const std::vector<std::string>& f_aff_mov_scalar,
          const bool                      affines_are_inverted)
      {
        // Read in ref matrices
        for (auto i = 0; i < img_ref_scalar_.size(); ++i){
          auto affine_mat_ref = flirt_mat_to_real_mat_scalar_(
              hdr_warp_space_,
              img_ref_scalar_[i],
              f_aff_ref_scalar[i],
              affines_are_inverted);
          aff_ref_scalar_.push_back(affine_mat_ref);
        }
        // Read in mov matrices
        for (auto i = 0; i < img_mov_scalar_.size(); ++i){
          // Read in reference volume header
          auto affine_mat_ref = flirt_mat_to_real_mat_scalar_(
              hdr_warp_space_,
              img_mov_scalar_[i],
              f_aff_mov_scalar[i],
              affines_are_inverted);
          aff_mov_scalar_.push_back(affine_mat_ref);
        }
      }

      /// \details We are making the assumption that the matrices were calculated using FLIRT,
      ///          and therefore describe a transform between the volumes in "scaled mm
      ///          space". In MMORF we make the assumption that all coordinates and positions
      ///          are given in "real world mm space". As such, there is some serious
      ///          shuffling that needs to be done in order to get the FLIRT affine matrices
      ///          into a form that works with MOORF's coordinate system.
      void create_affine_matrices_tensor_(
          const std::vector<std::string>& f_aff_ref_tensor,
          const std::vector<std::string>& f_aff_mov_tensor,
          const bool                      affines_are_inverted)
      {
        // Read in ref matrices
        for (auto i = 0; i < img_ref_tensor_.size(); ++i){
          auto affine_mat_ref = flirt_mat_to_real_mat_tensor_(
              hdr_warp_space_,
              img_ref_tensor_[i],
              f_aff_ref_tensor[i],
              affines_are_inverted);
          aff_ref_tensor_.push_back(affine_mat_ref);
        }
        // Read in mov matrices
        for (auto i = 0; i < img_mov_tensor_.size(); ++i){
          // Read in reference volume header
          auto affine_mat_ref = flirt_mat_to_real_mat_tensor_(
              hdr_warp_space_,
              img_mov_tensor_[i],
              f_aff_mov_tensor[i],
              affines_are_inverted);
          aff_mov_tensor_.push_back(affine_mat_ref);
        }
      }

      /// Take a matrix output by flirt and convert it to real world coordinates
      arma::fmat flirt_mat_to_real_mat_scalar_(
          NEWIMAGE::volume<float>& vol_source,
          NEWIMAGE::volume<float>& vol_dest,
          std::string              filename_affine_flirt,
          bool                     mat_inverted)
      {
        auto vox_to_mm_source = affine_newmat_to_arma_(vol_source.newimagevox2mm_mat());
        auto vox_to_mm_dest = affine_newmat_to_arma_(vol_dest.newimagevox2mm_mat());
        auto sampling_source = affine_newmat_to_arma_(vol_source.sampling_mat());
        auto sampling_dest = affine_newmat_to_arma_(vol_dest.sampling_mat());
        auto flirt_mat = arma::fmat();
        flirt_mat.load(filename_affine_flirt, arma::raw_ascii);
        if (mat_inverted){
          flirt_mat = flirt_mat.i();
        }
        arma::fmat real_mat =
          vox_to_mm_dest
          * sampling_dest.i()
          * flirt_mat
          * sampling_source
          * vox_to_mm_source.i();
        return real_mat;
      }

      /// Take a matrix output by flirt and convert it to real world coordinates
      arma::fmat flirt_mat_to_real_mat_tensor_(
          NEWIMAGE::volume<float>&   vol_source,
          NEWIMAGE::volume4D<float>& vol_dest,
          std::string                filename_affine_flirt,
          bool                       mat_inverted)
      {
        auto vox_to_mm_source = affine_newmat_to_arma_(vol_source.newimagevox2mm_mat());
        auto vox_to_mm_dest = affine_newmat_to_arma_(vol_dest.newimagevox2mm_mat());
        auto sampling_source = affine_newmat_to_arma_(vol_source.sampling_mat());
        auto sampling_dest = affine_newmat_to_arma_(vol_dest.sampling_mat());
        auto flirt_mat = arma::fmat();
        flirt_mat.load(filename_affine_flirt, arma::raw_ascii);
        if (mat_inverted){
          flirt_mat = flirt_mat.i();
        }
        arma::fmat real_mat =
          vox_to_mm_dest
          * sampling_dest.i()
          * flirt_mat
          * sampling_source
          * vox_to_mm_source.i();
        return real_mat;
      }

      /// Convert an affine matrix from newmat to armadillo matrix
      /// \todo Get rid of templating by finding out what the actual type should be.
      template<typename T>
      arma::fmat affine_newmat_to_arma_(const T& newmat_mat) const
      {
        auto arma_mat = arma::Mat<float>(4, 4, arma::fill::zeros);
        for (auto j = 0; j < 4; ++j){
          for (auto i = 0; i < 4; ++i){
            arma_mat(i,j) = newmat_mat.element(i,j);
          }
        }
        return arma_mat;
      }

      /// Estimate the bias field for each scalar modality pair
      void estimate_bias_field_(
        const int                                                   i_iteration,
        const std::vector<std::shared_ptr<MMORF::VolumeBSpline> >&  ref_scalar_vols,
        const std::vector<std::shared_ptr<MMORF::VolumeBSpline> >&  mov_scalar_vols)
      {
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          if (b_bias_estimate_[i_volume]){
            auto max_res = std::min(
                get_max_resolution_(ref_scalar_vols[i_volume]),
                get_max_resolution_(mov_scalar_vols[i_volume]));
            auto sampling_freq = calculate_sampling_frequency_bias_field_(
                i_iteration,
                i_volume,
                max_res);
            auto mask_ref_vol = use_mask_ref_scalar_[i_volume][i_iteration] ?
                std::make_shared<MMORF::VolumeBSpline>(*mask_ref_scalar_[i_volume]) :
                nullptr;
            auto mask_mov_vol = use_mask_mov_scalar_[i_volume][i_iteration] ?
                std::make_shared<MMORF::VolumeBSpline>(*mask_mov_scalar_[i_volume]) :
                nullptr;
            auto cost_fxn_bias_ssd = std::make_shared<MMORF::CostFxnSSDBiasFieldMaskedExcluded>(
                ref_scalar_vols[i_volume],
                mov_scalar_vols[i_volume],
                aff_ref_scalar_[i_volume],
                aff_mov_scalar_[i_volume],
                mask_ref_vol,
                mask_mov_vol,
                warp_field_,
                bias_field_[i_volume],
                sampling_freq,
                debug_);
            auto cost_fxn_bias_reg = std::make_shared<MMORF::CostFxnBendingEnergyBiasField>(
                bias_field_[i_volume],
                sampling_freq);
            auto cost_fxn_bias_compound = MMORF::CostFxnCompoundBiasFieldVarianceScaled(
                std::vector<std::shared_ptr<MMORF::CostFxn> >(1,cost_fxn_bias_ssd),
                std::vector<float>(1,1.0f),
                cost_fxn_bias_reg,
                lambda_reg_bias_[i_volume][i_iteration],
                bias_field_[i_volume]);
            auto linear_solver = std::make_shared<MMORF::LinearSolverCusp>(
                100,
                1e-3f);
            auto optimiser = MMORF::OptimiserLevenberg(linear_solver);
            auto optimised_parameters = bias_field_[i_volume]->get_parameters();
            optimised_parameters = optimiser.optimise(
                cost_fxn_bias_compound,
                optimised_parameters,
                1,
                1e-1f);
            bias_field_[i_volume]->set_parameters(optimised_parameters);
          }
        }
      }

      /// Estimate the bias field without smoothing once before exiting
      void estimate_bias_field_no_smoothing_()
      {
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          if (b_bias_estimate_[i_volume]){
            auto ref_vol = std::make_shared<MMORF::VolumeBSpline>(
                img_ref_scalar_[i_volume]);
            auto mov_vol = std::make_shared<MMORF::VolumeBSpline>(
                img_mov_scalar_[i_volume]);
            auto max_res = std::min(
                get_max_resolution_(ref_vol),get_max_resolution_(mov_vol));
            auto sampling_freq = static_cast<int>(
                bias_res_init_[i_volume]/max_res);
            sampling_freq = std::max(sampling_freq, 1);
            auto mask_ref_vol = use_mask_ref_scalar_[i_volume].back() ?
                std::make_shared<MMORF::VolumeBSpline>(*mask_ref_scalar_[i_volume]) :
                nullptr;
            auto mask_mov_vol = use_mask_mov_scalar_[i_volume].back() ?
                std::make_shared<MMORF::VolumeBSpline>(*mask_mov_scalar_[i_volume]) :
                nullptr;
            auto cost_fxn_bias_ssd = std::make_shared<MMORF::CostFxnSSDBiasFieldMaskedExcluded>(
                ref_vol,
                mov_vol,
                aff_ref_scalar_[i_volume],
                aff_mov_scalar_[i_volume],
                mask_ref_vol,
                mask_mov_vol,
                warp_field_,
                bias_field_[i_volume],
                sampling_freq);
            auto cost_fxn_bias_reg = std::make_shared<MMORF::CostFxnBendingEnergyBiasField>(
                bias_field_[i_volume],
                sampling_freq);
            auto cost_fxn_bias_compound = MMORF::CostFxnCompoundBiasFieldVarianceScaled(
                std::vector<std::shared_ptr<MMORF::CostFxn> >(1,cost_fxn_bias_ssd),
                std::vector<float>(1,1.0f),
                cost_fxn_bias_reg,
                lambda_reg_bias_[i_volume][n_iterations_ - 1],
                bias_field_[i_volume]);
            auto linear_solver = std::make_shared<MMORF::LinearSolverCusp>(
                100,
                1e-3f);
            auto optimiser = MMORF::OptimiserLevenberg(linear_solver);
            auto optimised_parameters = bias_field_[i_volume]->get_parameters();
            optimised_parameters = optimiser.optimise(
                cost_fxn_bias_compound,
                optimised_parameters,
                5,
                1e-2f);
            bias_field_[i_volume]->set_parameters(optimised_parameters);
          }
        }
      }

      /// Choose a sensible sampling frequency
      /// \details Smoothing introduces some filtering, and leads to an acceptable level of
      ///          sampling rate of about f_samp > 1/FWHM (to satisfy Nyquist at -3dB). This
      ///          leads to a sample_step < FWHM. The sample step is equal to
      ///          knot_spacing/sampling_freq. Therefore we need:
      ///
      ///                       sampling_freq > knot_spacing/FWHM
      ///
      ///         Practically speaking, we would prefer to have a sampling frequency of at
      ///         least 2. We will therefore take the maximum of the required frequency
      ///         as calculated above, and 2. Additionally we will limit the number of samples
      ///         to a maximum of 10 for computational reasons.
      /// \todo Replace the hard max of 10 with the minimum sampling frequency which gives a
      ///       sample step smaller than the highest resolution in either of the volumes
      int calculate_sampling_frequency_scalar_(
          const unsigned int i_iteration,
          const unsigned int i_volume,
          const float        max_resolution) const
      {
        auto knot_spacing = warp_field_->get_knot_spacing();
        auto smooth_fwhm_ref = fwhm_ref_scalar_[i_volume][i_iteration];
        auto smooth_fwhm_mov = fwhm_mov_scalar_[i_volume][i_iteration];
        auto sampling_freq_ref = (int)0;
        auto sampling_freq_mov = (int)0;
        // Calculate sampling frequency for ref
        if (smooth_fwhm_ref > 0){
          sampling_freq_ref = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_ref));
        }
        sampling_freq_ref = std::max(sampling_freq_ref, 1);
        sampling_freq_ref = std::min(sampling_freq_ref, 4);
        // Calculate sampling frequency for mov
        if (smooth_fwhm_mov > 0){
          sampling_freq_mov = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_mov));
        }
        sampling_freq_mov = std::max(sampling_freq_mov, 1);
        sampling_freq_mov = std::min(sampling_freq_mov, 4);
        // Calculate the maximum between the ref and mov samples
        auto sampling_freq_final = std::max(sampling_freq_ref, sampling_freq_mov);
        // If sampling rate would mean we sample at a higher resolution than out highest
        // resolution volume, reduce sampling frequency to 1
        if (static_cast<float>(sampling_freq_final) > knot_spacing/max_resolution){
          sampling_freq_final = static_cast<int>(std::ceil(knot_spacing/max_resolution));
        }
        std::cout << "SCALAR SAMPLING FREQUENCY of iteration " << i_iteration << ", volume "
          << i_volume << " is: " << sampling_freq_final << std::endl;
        return sampling_freq_final;
      }

      /// Choose a sensible sampling frequency
      /// \details Smoothing introduces some filtering, and leads to an acceptable level of
      ///          sampling rate of about f_samp > 1/FWHM (to satisfy Nyquist at -3dB). This
      ///          leads to a sample_step < FWHM. The sample step is equal to
      ///          knot_spacing/sampling_freq. Therefore we need:
      ///
      ///                       sampling_freq > knot_spacing/FWHM
      ///
      ///         Practically speaking, we would prefer to have a sampling frequency of at
      ///         least 2. We will therefore take the maximum of the required frequency
      ///         as calculated above, and 2. Additionally we will limit the number of samples
      ///         to a maximum of 10 for computational reasons.
      /// \todo Replace the hard max of 10 with the minimum sampling frequency which gives a
      ///       sample step smaller than the highest resolution in either of the volumes
      int calculate_sampling_frequency_tensor_(
          const unsigned int i_iteration,
          const unsigned int i_volume,
          const float        max_resolution) const
      {
        auto knot_spacing = warp_field_->get_knot_spacing();
        auto smooth_fwhm_ref = fwhm_ref_tensor_[i_volume][i_iteration];
        auto smooth_fwhm_mov = fwhm_mov_tensor_[i_volume][i_iteration];
        auto sampling_freq_ref = (int)0;
        auto sampling_freq_mov = (int)0;
        // Calculate sampling frequency for ref
        if (smooth_fwhm_ref > 0){
          sampling_freq_ref = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_ref));
        }
        sampling_freq_ref = std::max(sampling_freq_ref, 1);
        sampling_freq_ref = std::min(sampling_freq_ref, 4);
        // Calculate sampling frequency for mov
        if (smooth_fwhm_mov > 0){
          sampling_freq_mov = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_mov));
        }
        sampling_freq_mov = std::max(sampling_freq_mov, 1);
        sampling_freq_mov = std::min(sampling_freq_mov, 4);
        // Calculate the maximum between scalarthe ref and mov samples
        auto sampling_freq_final = std::max(sampling_freq_ref, sampling_freq_mov);
        // If sampling rate would mean we sample at a higher resolution than out highest
        // resolution volume, reduce sampling frequency to 1
        if (static_cast<float>(sampling_freq_final) > knot_spacing/max_resolution){
          sampling_freq_final = static_cast<int>(std::ceil(knot_spacing/max_resolution));
        }
        std::cout << "TENSOR SAMPLING FREQUENCY of iteration " << i_iteration << ", volume "
          << i_volume << " is: " << sampling_freq_final << std::endl;
        return sampling_freq_final;
      }

      /// Create a vector of cost functions for each pair of volumes
      std::vector<std::shared_ptr<MMORF::CostFxn> > create_cost_fxns_(
          const unsigned int                                          i_iteration,
          const float                                                 knot_spacing,
          const std::vector<std::shared_ptr<MMORF::VolumeBSpline> >&  ref_scalar_vols,
          const std::vector<std::shared_ptr<MMORF::VolumeBSpline> >&  mov_scalar_vols,
          const std::vector<std::shared_ptr<MMORF::VolumeTensor> >&   ref_tensor_vols,
          const std::vector<std::shared_ptr<MMORF::VolumeTensor> >&   mov_tensor_vols)
      {
        // Create scalar cost functions
        auto cost_fxns = std::vector<std::shared_ptr<MMORF::CostFxn> >();
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          auto max_res = std::min(
            get_max_resolution_(ref_scalar_vols[i_volume]),
            get_max_resolution_(mov_scalar_vols[i_volume]));
          auto sampling_frequency = calculate_sampling_frequency_scalar_(
              i_iteration,
              i_volume,
              max_res);
          auto mask_ref_vol = use_mask_ref_scalar_[i_volume][i_iteration] ?
              std::make_shared<MMORF::VolumeBSpline>(*mask_ref_scalar_[i_volume]) :
              nullptr;
          auto mask_mov_vol = use_mask_mov_scalar_[i_volume][i_iteration] ?
              std::make_shared<MMORF::VolumeBSpline>(*mask_mov_scalar_[i_volume]) :
              nullptr;
          auto optimiser_type = knot_spacing >= hires_ ? optimiser_lowres_ : optimiser_hires_;
          switch (optimiser_type){
            case MMORF::OptimiserType::LM:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnSSDWarpFieldSymmetricMaskedExcluded>(
                    ref_scalar_vols[i_volume],
                    mov_scalar_vols[i_volume],
                    aff_ref_scalar_[i_volume],
                    aff_mov_scalar_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency,
                    bias_field_[i_volume],
                    debug_));
              break;
            case MMORF::OptimiserType::MM:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnSSDWarpFieldSymmetricMaskedExcludedDiagHess>(
                    ref_scalar_vols[i_volume],
                    mov_scalar_vols[i_volume],
                    aff_ref_scalar_[i_volume],
                    aff_mov_scalar_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency,
                    bias_field_[i_volume],
                    debug_));
              break;
            case MMORF::OptimiserType::SCG:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnSSDWarpFieldSymmetricMaskedExcludedDiagHess>(
                    ref_scalar_vols[i_volume],
                    mov_scalar_vols[i_volume],
                    aff_ref_scalar_[i_volume],
                    aff_mov_scalar_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency,
                    bias_field_[i_volume],
                    debug_));
              break;
            default:
              std::cout<< "This should never happen!" << std::endl;
              break;
          }
        }
        // Create tensor cost functions
        for (auto i_volume = 0; i_volume < n_img_tensor_; ++i_volume){
          auto max_res = std::min(
            get_max_resolution_(ref_tensor_vols[i_volume]),
            get_max_resolution_(mov_tensor_vols[i_volume]));
          auto sampling_frequency = calculate_sampling_frequency_tensor_(
              i_iteration,
              i_volume,
              max_res);
          auto mask_ref_vol = use_mask_ref_tensor_[i_volume][i_iteration] ?
              std::make_shared<MMORF::VolumeBSpline>(*mask_ref_tensor_[i_volume]) :
              nullptr;
          auto mask_mov_vol = use_mask_mov_tensor_[i_volume][i_iteration] ?
              std::make_shared<MMORF::VolumeBSpline>(*mask_mov_tensor_[i_volume]) :
              nullptr;
          auto optimiser_type = knot_spacing >= hires_ ? optimiser_lowres_ : optimiser_hires_;
          switch (optimiser_type){
            case MMORF::OptimiserType::LM:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnTensorL2WarpFieldSymmetricMaskedExcluded>(
                    ref_tensor_vols[i_volume],
                    mov_tensor_vols[i_volume],
                    aff_ref_tensor_[i_volume],
                    aff_mov_tensor_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency));
              break;
            case MMORF::OptimiserType::MM:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess>(
                    ref_tensor_vols[i_volume],
                    mov_tensor_vols[i_volume],
                    aff_ref_tensor_[i_volume],
                    aff_mov_tensor_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency));
              break;
            case MMORF::OptimiserType::SCG:
              cost_fxns.push_back(
                  std::make_shared<MMORF::CostFxnTensorL2WarpFieldSymmetricMaskedExcludedDiagHess>(
                    ref_tensor_vols[i_volume],
                    mov_tensor_vols[i_volume],
                    aff_ref_tensor_[i_volume],
                    aff_mov_tensor_[i_volume],
                    mask_ref_vol,
                    mask_mov_vol,
                    warp_field_,
                    sampling_frequency));
              break;
            default:
              std::cout<< "This should never happen!" << std::endl;
              break;
          }
        }
        return cost_fxns;
      }

      /// Calculate regularisation sampling frequency
      /// \details  The sampling frequency for the regularisation does not change the value
      ///           of the regularisation, merely its accuracy. Therefore the choice of
      ///           sampling frequency is a matter of balancing speed and accuracy.
      ///           Sampling will occur at the knot spacing for sub-mm warps, approximately
      ///           every mm for 1-4mm warps, and at 1/4 the knot spacing for knot spacings
      ///           greater than 4mm.
      int calculate_sampling_frequency_reg_(
          const float knot_spacing) const
      {
        auto sampling_frequency = static_cast<int>(std::floor(knot_spacing/max_res_));
        sampling_frequency = std::max(sampling_frequency, 1);
        sampling_frequency = std::min(sampling_frequency, 4);
        std::cout << "REGULARISER SAMPLING FREQUENCY is: " << sampling_frequency << std::endl;
        return sampling_frequency;
      }

      /// Create the regularisation cost function
      std::shared_ptr<MMORF::CostFxn> create_reg_(
          const unsigned int i_iteration,
          const float knot_spacing)
      {
        auto regulariser = std::shared_ptr<MMORF::CostFxn>(nullptr);
        auto sampling_frequency = calculate_sampling_frequency_reg_(knot_spacing);
        if (i_iteration == 0){
          regulariser =
              std::make_shared<MMORF::CostFxnBendingEnergy>(
                warp_field_,
                sampling_frequency);
        }
        else{
          if (knot_spacing >= hires_){
            regulariser =
                std::make_shared<MMORF::CostFxnLogJacobianSingularValues>(
                  warp_field_,
                  sampling_frequency,
                  debug_);
          }
          else{
            regulariser =
                std::make_shared<MMORF::CostFxnLogJacobianSingularValuesDiagHess>(
                  warp_field_,
                  sampling_frequency);
          }
        }
        return regulariser;
      }

      /// Choose a sensible sampling frequency
      /// \details Smoothing introduces some filtering, and leads to an acceptable level of
      ///          sampling rate of about f_samp > 1/FWHM (to satisfy Nyquist at -3dB). This
      ///          leads to a sample_step < FWHM. The sample step is equal to
      ///          knot_spacing/sampling_freq. Therefore we need:
      ///
      ///                       sampling_freq > knot_spacing/FWHM
      ///
      ///         Practically speaking, we would prefer to have a sampling frequency of at
      ///         least 2. We will therefore take the maximum of the required frequency
      ///         as calculated above, and 2. Additionally we will limit the number of samples
      ///         to a maximum of 10 for computational reasons.
      /// \todo Replace the hard max of 10 with the minimum sampling frequency which gives a
      ///       sample step smaller than the highest resolution in either of the volumes
      int calculate_sampling_frequency_bias_field_(
          const unsigned int i_iteration,
          const unsigned int i_volume,
          const float        max_resolution) const
      {
        auto knot_spacing = bias_res_init_[i_volume];
        auto smooth_fwhm_ref = fwhm_ref_scalar_[i_volume][i_iteration];
        auto smooth_fwhm_mov = fwhm_mov_scalar_[i_volume][i_iteration];
        auto sampling_freq_ref = (int)0;
        auto sampling_freq_mov = (int)0;
        // Calculate sampling frequency for ref
        if (smooth_fwhm_ref > 0){
          sampling_freq_ref = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_ref));
        }
        sampling_freq_ref = std::max(sampling_freq_ref, 1);
        sampling_freq_ref = std::min(sampling_freq_ref, 16);
        // Calculate sampling frequency for mov
        if (smooth_fwhm_mov > 0){
          sampling_freq_mov = static_cast<int>(std::floor(2.0f*knot_spacing/smooth_fwhm_mov));
        }
        sampling_freq_mov = std::max(sampling_freq_mov, 1);
        sampling_freq_mov = std::min(sampling_freq_mov, 16);
        // Calculate the maximum between the ref and mov samples
        auto sampling_freq_final = std::max(sampling_freq_ref, sampling_freq_mov);
        // If sampling rate would mean we sample at a higher resolution than out highest
        // resolution volume, reduce sampling frequency to 1
        if (static_cast<float>(sampling_freq_final) > knot_spacing/max_resolution){
          sampling_freq_final = static_cast<int>(std::ceil(knot_spacing/max_resolution));
        }
        std::cout << "BIAS FIELD SAMPLING FREQUENCY of iteration " << i_iteration << ", volume "
          << i_volume << " is: " << sampling_freq_final << std::endl;
        return sampling_freq_final;
      }

      /// Calculate the highest resolution in 1D for a given volume
      template<typename T>
      float get_max_resolution_(const std::shared_ptr<T> vol)
      {
        auto res = vol->get_resolution();
        auto max_res = std::min(std::min(std::abs(res[0]),std::abs(res[1])),std::abs(res[2]));
        return max_res;
      }

      /// Calculate the highest resolution in 1D across all scalar and tensor modalities
      float get_max_resolution_all_()
      {
        auto max_res = std::numeric_limits<float>::infinity();
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          auto ref_vol = std::make_shared<MMORF::VolumeBSpline>(
              img_ref_scalar_[i_volume]);
          auto mov_vol = std::make_shared<MMORF::VolumeBSpline>(
              img_mov_scalar_[i_volume]);
          auto max_res_scalar = std::min(get_max_resolution_(ref_vol),get_max_resolution_(mov_vol));
          max_res = std::min(max_res_scalar,max_res);
        }
        for (auto i_volume = 0; i_volume < n_img_tensor_; ++i_volume){
          auto ref_vol = std::make_shared<MMORF::VolumeTensor>(
              img_ref_tensor_[i_volume]);
          auto mov_vol = std::make_shared<MMORF::VolumeTensor>(
              img_mov_tensor_[i_volume]);
          auto max_res_tensor = std::min(get_max_resolution_(ref_vol),get_max_resolution_(mov_vol));
          max_res = std::min(max_res_tensor,max_res);
        }
        return max_res;
      }

      /// Create a vector of shared pointers to smoothed reference scalar volumes
      std::vector<std::shared_ptr<MMORF::VolumeBSpline> > create_ref_scalar_vols_(
        const unsigned int i_iteration) const
      {
        auto smoothed_vols = std::vector<std::shared_ptr<MMORF::VolumeBSpline> >(n_img_scalar_);
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          smoothed_vols[i_volume] = std::make_shared<MMORF::VolumeBSpline>(
              img_ref_scalar_[i_volume],
              fwhm_ref_scalar_[i_volume][i_iteration],
              use_implicit_mask_[i_volume],
              false,
              debug_);
        }
        return smoothed_vols;
      }

      /// Create a vector of shared pointers to smoothed moving scalar volumes
      std::vector<std::shared_ptr<MMORF::VolumeBSpline> > create_mov_scalar_vols_(
        const unsigned int i_iteration) const
      {
        auto smoothed_vols = std::vector<std::shared_ptr<MMORF::VolumeBSpline> >(n_img_scalar_);
        for (auto i_volume = 0; i_volume < n_img_scalar_; ++i_volume){
          smoothed_vols[i_volume] = std::make_shared<MMORF::VolumeBSpline>(
              img_mov_scalar_[i_volume],
              fwhm_mov_scalar_[i_volume][i_iteration],
              use_implicit_mask_[i_volume],
              false,
              debug_);
        }
        return smoothed_vols;
      }

      /// Create a vector of shared pointers to smoothed reference tensor volumes
      std::vector<std::shared_ptr<MMORF::VolumeTensor> > create_ref_tensor_vols_(
        const unsigned int i_iteration) const
      {
        auto smoothed_vols = std::vector<std::shared_ptr<MMORF::VolumeTensor> >(n_img_tensor_);
        for (auto i_volume = 0; i_volume < n_img_tensor_; ++i_volume){
          smoothed_vols[i_volume] = std::make_shared<MMORF::VolumeTensor>(
              img_ref_tensor_[i_volume],
              fwhm_ref_tensor_[i_volume][i_iteration]);
        }
        return smoothed_vols;
      }

      /// Create a vector of shared pointers to smoothed moving tensor volumes
      std::vector<std::shared_ptr<MMORF::VolumeTensor> > create_mov_tensor_vols_(
        const unsigned int i_iteration) const
      {
        auto smoothed_vols = std::vector<std::shared_ptr<MMORF::VolumeTensor> >(n_img_tensor_);
        for (auto i_volume = 0; i_volume < n_img_tensor_; ++i_volume){
          smoothed_vols[i_volume] = std::make_shared<MMORF::VolumeTensor>(
              img_mov_tensor_[i_volume],
              fwhm_mov_tensor_[i_volume][i_iteration]);
        }
        return smoothed_vols;
      }

      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      // Warp
      unsigned int                                            n_iterations_;
      unsigned int                                            n_img_scalar_;
      unsigned int                                            n_img_tensor_;
      std::vector<int>                                        warp_scaling_;
      std::string                                             f_img_warp_space_;
      std::shared_ptr<MMORF::WarpFieldBSpline>                warp_field_;
      NEWIMAGE::volume<float>                                 hdr_warp_space_;
      std::pair<
        std::vector<float>,
        std::vector<float> >                                  extents_warp_space_;
      // Scalar
      std::vector<std::vector<float> >                        lambda_scalar_;
      std::vector<std::vector<float> >                        fwhm_ref_scalar_;
      std::vector<std::vector<float> >                        fwhm_mov_scalar_;
      std::vector<NEWIMAGE::volume<float> >                   img_ref_scalar_;
      std::vector<NEWIMAGE::volume<float> >                   img_mov_scalar_;
      std::vector<bool>                                       use_implicit_mask_;
      std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > mask_ref_scalar_;
      std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > mask_mov_scalar_;
      std::vector<std::vector<bool> >                         use_mask_ref_scalar_;
      std::vector<std::vector<bool> >                         use_mask_mov_scalar_;
      std::vector<arma::fmat>                                 aff_ref_scalar_;
      std::vector<arma::fmat>                                 aff_mov_scalar_;
      // Tensor
      std::vector<std::vector<float> >                        lambda_tensor_;
      std::vector<std::vector<float> >                        fwhm_ref_tensor_;
      std::vector<std::vector<float> >                        fwhm_mov_tensor_;
      std::vector<NEWIMAGE::volume4D<float> >                 img_ref_tensor_;
      std::vector<NEWIMAGE::volume4D<float> >                 img_mov_tensor_;
      std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > mask_ref_tensor_;
      std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > mask_mov_tensor_;
      std::vector<std::vector<bool> >                         use_mask_ref_tensor_;
      std::vector<std::vector<bool> >                         use_mask_mov_tensor_;
      std::vector<arma::fmat>                                 aff_ref_tensor_;
      std::vector<arma::fmat>                                 aff_mov_tensor_;
      // Regularisation
      std::vector<float>                                      lambda_reg_;
      float                                                   max_res_;
      // Optimiser
      float                                                   hires_;
      MMORF::OptimiserType                                    optimiser_lowres_;
      int                                                     optimiser_max_it_lowres_;
      float                                                   optimiser_rel_tol_lowres_;
      MMORF::OptimiserType                                    optimiser_hires_;
      int                                                     optimiser_max_it_hires_;
      float                                                   optimiser_rel_tol_hires_;
      // Linear Solver
      int                                                     solver_max_it_lowres_;
      float                                                   solver_rel_tol_lowres_;
      int                                                     solver_max_it_hires_;
      float                                                   solver_rel_tol_hires_;
      // Bias Field
      std::vector<std::shared_ptr<MMORF::BiasFieldBSpline> >  bias_field_;
      std::vector<bool>                                       b_bias_estimate_;
      std::vector<float>                                      bias_res_init_;
      std::vector<std::vector<float> >                        lambda_reg_bias_;
      // Debugging option
      bool                                                    debug_;
      const static std::string                                debug_dir_;
  };

  /// Static debug directory initialisation
  const std::string RegistrationCoordinatorMultimodal::Impl::debug_dir_ =
    "debug/RegistrationCoordinatorMultimodal/";
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  RegistrationCoordinatorMultimodal::~RegistrationCoordinatorMultimodal() = default;
  /// Move ctor
  RegistrationCoordinatorMultimodal::RegistrationCoordinatorMultimodal(RegistrationCoordinatorMultimodal&& rhs) = default;
  /// Move assignment operator
  RegistrationCoordinatorMultimodal& RegistrationCoordinatorMultimodal::operator=(RegistrationCoordinatorMultimodal&& rhs) = default;
  /// Copy ctor
  RegistrationCoordinatorMultimodal::RegistrationCoordinatorMultimodal(const RegistrationCoordinatorMultimodal& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  RegistrationCoordinatorMultimodal& RegistrationCoordinatorMultimodal::operator=(const RegistrationCoordinatorMultimodal& rhs)
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
  /// Complete ctor
  RegistrationCoordinatorMultimodal::RegistrationCoordinatorMultimodal(
      // Warp defining parameters
      const float                             warp_res_init,
      const std::vector<int>&                 warp_scaling,
      const std::string&                      f_img_warp_space,
      // Scalar image parameters
      const std::vector<std::string>&         f_img_ref_scalar,
      const std::vector<std::string>&         f_img_mov_scalar,
      const std::vector<std::string>&         f_aff_ref_scalar,
      const std::vector<std::string>&         f_aff_mov_scalar,
      const std::vector<bool>&                use_implicit_mask,
      const std::vector<std::string>&         f_mask_ref_scalar,
      const std::vector<std::string>&         f_mask_mov_scalar,
      const std::vector<std::vector<bool> >&  use_mask_ref_scalar,
      const std::vector<std::vector<bool> >&  use_mask_mov_scalar,
      const std::vector<std::vector<float> >& fwhm_ref_scalar,
      const std::vector<std::vector<float> >& fwhm_mov_scalar,
      const std::vector<std::vector<float> >& lambda_scalar,
      // Tensor image parameters
      const std::vector<std::string>&         f_img_ref_tensor,
      const std::vector<std::string>&         f_img_mov_tensor,
      const std::vector<std::string>&         f_aff_ref_tensor,
      const std::vector<std::string>&         f_aff_mov_tensor,
      const std::vector<std::string>&         f_mask_ref_tensor,
      const std::vector<std::string>&         f_mask_mov_tensor,
      const std::vector<std::vector<bool> >&  use_mask_ref_tensor,
      const std::vector<std::vector<bool> >&  use_mask_mov_tensor,
      const std::vector<std::vector<float> >& fwhm_ref_tensor,
      const std::vector<std::vector<float> >& fwhm_mov_tensor,
      const std::vector<std::vector<float> >& lambda_tensor,
      // Regularisation parameters
      const std::vector<float>&               lambda_reg,
      // Optimiser parameters
      const float                             hires,
      const MMORF::OptimiserType              optimiser_lowres,
      const int                               optimiser_max_it_lowres,
      const float                             optimiser_rel_tol_lowres,
      const MMORF::OptimiserType              optimiser_hires,
      const int                               optimiser_max_it_hires,
      const float                             optimiser_rel_tol_hires,
      // Solver parameters
      const int                               solver_max_it_lowres,
      const float                             solver_rel_tol_lowres,
      const int                               solver_max_it_hires,
      const float                             solver_rel_tol_hires,
      // Bias field parameters
      const std::vector<bool>&                bias_estimate,
      const std::vector<float>&               bias_res_init,
      const std::vector<std::vector<float> >& lambda_reg_bias,
      // Debugging option
      const bool                              debug
      // Miscelaneous parameters
      //const bool                              affines_are_inverted
      )
    : pimpl_(
        MMORF::make_unique<Impl>(
          warp_res_init,
          warp_scaling,
          f_img_warp_space,
          f_img_ref_scalar,
          f_img_mov_scalar,
          f_aff_ref_scalar,
          f_aff_mov_scalar,
          use_implicit_mask,
          f_mask_ref_scalar,
          f_mask_mov_scalar,
          use_mask_ref_scalar,
          use_mask_mov_scalar,
          fwhm_ref_scalar,
          fwhm_mov_scalar,
          lambda_scalar,
          f_img_ref_tensor,
          f_img_mov_tensor,
          f_aff_ref_tensor,
          f_aff_mov_tensor,
          f_mask_ref_tensor,
          f_mask_mov_tensor,
          use_mask_ref_tensor,
          use_mask_mov_tensor,
          fwhm_ref_tensor,
          fwhm_mov_tensor,
          lambda_tensor,
          lambda_reg,
          hires,
          optimiser_lowres,
          optimiser_max_it_lowres,
          optimiser_rel_tol_lowres,
          optimiser_hires,
          optimiser_max_it_hires,
          optimiser_rel_tol_hires,
          solver_max_it_lowres,
          solver_rel_tol_lowres,
          solver_max_it_hires,
          solver_rel_tol_hires,
          bias_estimate,
          bias_res_init,
          lambda_reg_bias,
          debug
          )
        )
  {}
  /// Run through the registration process
//  void RegistrationCoordinatorMultimodal::register_volumes(bool save_all_steps, std::string folder_name)
  void RegistrationCoordinatorMultimodal::register_volumes()
  {
    pimpl_->register_volumes();
  }
  /// Save warp field as 4D nifti
  void RegistrationCoordinatorMultimodal::save_warp_field(std::string f_warp_field_out)
  {
    pimpl_->save_warp_field(f_warp_field_out);
  }
  /// Save bias field
  void RegistrationCoordinatorMultimodal::save_bias_field(std::string f_bias_field_out)
  {
    pimpl_->save_bias_field(f_bias_field_out);
  }
  /// Save warped moving volumes
//  void RegistrationCoordinatorMultimodal::save_warped_volumes(std::string warp_prefix)
//  {
//    pimpl_->save_warped_volumes(warp_prefix);
//  }
  /// Save Jacobian determinant of the warp field
  void RegistrationCoordinatorMultimodal::save_jacobian_determinants(std::string f_jacobian_out)
  {
    pimpl_->save_jacobian_determinants(f_jacobian_out);
  }
} // MMORF
