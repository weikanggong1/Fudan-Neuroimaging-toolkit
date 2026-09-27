//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Coordinate the exection of a volumetric registration involving DTI volumes
/// \details This class is responsible for coordinating the objects required to follow a
///          particular registration "recipe". This involves things like iterating over
///          different levels of smoothing, regularisation, etc.
/// \author Frederik Lange
/// \date October 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif
#include "MMORFio.h"
#include "MmorfMemory.h"
#include "RegistrationCoordinatorTensor.cuh"
#include "VolumeBSpline.cuh"
#include "WarpFieldBSpline.cuh"
#include "CostFxn.h"
#include "CostFxnSSDWarpFieldMasked.cuh"
#include "CostFxnSSDWarpFieldMaskedDiagHess.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "CostFxnLogJacobian.cuh"
#include "CostFxnLogJacobianSingularValues.cuh"
#include "CostFxnLogJacobianSingularValuesDiagHess.cuh"
#include "CostFxnLogJacobianSingularValuesExact.cuh"
#include "CostFxnTensorL2WarpFieldMasked.cuh"
#include "CostFxnCompoundWarpField.cuh"
#include "CostFxnCompoundWarpFieldVarianceScaled.cuh"
#include "Optimiser.h"
#include "OptimiserLevenbergMarquardt.cuh"
#include "OptimiserLevenberg.cuh"
#include "OptimiserScaledConjugateGradient.cuh"
#include "LinearSolverCusp.cuh"

#include "newimage/newimageall.h"

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
  class RegistrationCoordinatorTensor::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Complete ctor
      Impl(
          const std::string&                      filename_warp_space,
          // Parameters the size of the number of volumes being used
          const std::vector<std::string>&         filename_volumes_reference,
          const std::vector<std::string>&         filename_affine_transforms_reference,
          const std::vector<std::string>&         filename_volumes_moving,
          const std::vector<std::string>&         filename_affine_transforms_moving,
          const std::vector<std::string>&         filename_masks_reference,
          // Parameters the size of the number of iterations being performed
          const std::vector<std::vector<float> >& lambda_volumes,
          const std::vector<std::vector<float> >& smoothing_reference,
          const std::vector<std::vector<float> >& smoothing_moving,
          const std::vector<float>&               lambda_regularisation,
          const std::vector<int>&                 warp_scaling,
          // Paramters with a single value
          const float                             knot_spacing_initial,
          const bool                              affines_are_inverted
          )
        : n_volumes_(filename_volumes_reference.size())
        , n_iterations_(lambda_volumes.size())
        , filename_warp_space_(filename_warp_space)
        , lambda_volumes_(lambda_volumes)
        , smoothing_reference_(smoothing_reference)
        , smoothing_moving_(smoothing_moving)
        , lambda_regularisation_(lambda_regularisation)
        , warp_scaling_(warp_scaling)
      {
        /// \todo Replace asserts with exceptions
        // Check that the number of volumes match
        assert(filename_affine_transforms_reference.size() == n_volumes_
            && filename_volumes_moving.size() == n_volumes_
            && filename_affine_transforms_moving.size() == n_volumes_
            && filename_masks_reference.size() == n_volumes_);
        // Check that the number of iterations match
        assert(smoothing_reference.size() == n_iterations_
            && smoothing_moving.size() == n_iterations_
            && lambda_regularisation.size() == n_iterations_
            && warp_scaling.size() == n_iterations_);
        // Read in warp space information
        NEWIMAGE::read_volume_hdr_only(header_warp_space_,filename_warp_space);
        // Create a warp field which covers the reference space
        create_warp_field_(filename_warp_space, knot_spacing_initial);
        // Read reference volumes
        read_volumes_reference_(filename_volumes_reference);
        // Read moving volumes
        read_volumes_moving_(filename_volumes_moving);
        // Read mask volumes
        read_masks_reference_(filename_masks_reference);
        // Create affine matrices
        create_affine_matrices_(
            filename_affine_transforms_reference,
            filename_affine_transforms_moving,
            affines_are_inverted);
      }
      /// Run through the registration process
      void register_volumes(bool save_all_steps, std::string folder_name)
      {
        auto optimised_parameters = warp_field_->get_parameters();
        // Loop over iterations
        for (auto i_iteration = 0; i_iteration < n_iterations_; ++i_iteration){
          // Scale warp field if necessary
          if (warp_scaling_[i_iteration] > 1){

            auto extents_old = warp_field_->get_extents();
            std::cout << "Old extents: ";
            for (const auto& point : extents_old.first){
              std::cout << point << " ";
            }
            std::cout << "   :    ";
            for (const auto& point : extents_old.second){
              std::cout << point << " ";
            }
            std::cout << std::endl;

            auto warp_field_reparameterised =
              warp_field_->reparameterise(warp_scaling_[i_iteration]);
            *warp_field_ = warp_field_reparameterised.crop(extents_warp_space_);

            auto extents_new = warp_field_->get_extents();
            std::cout << "New extents: ";
            for (const auto& point : extents_new.first){
              std::cout << point << " ";
            }
            std::cout << "   :    ";
            for (const auto& point : extents_new.second){
              std::cout << point << " ";
            }
            std::cout << std::endl;
          }
          auto knot_spacing = warp_field_->get_knot_spacing();
          //save_warp_field("data_test_RegistrationCoordinatorTensor/warp_iter_"
          //    + std::to_string(i_iteration) + "_start");
          // Get initial parameters to optimise over
          optimised_parameters = warp_field_->get_parameters();
          auto dims = warp_field_->get_dimensions();
          // Create individual cost functions
          auto cost_fxns = create_cost_fxns_(i_iteration,knot_spacing);
          // Create regularisation
          // NOTE: The sampling frequency for the regularisation does not change the value
          //       of the regularisation, merely its accuracy. Therefore the choice of
          //       sampling frequency is a matter of balancing speed and accuracy.
          auto regulariser = std::shared_ptr<MMORF::CostFxn>(nullptr);
          auto knot_spacing_floored = static_cast<int>(
              std::floor(warp_field_->get_knot_spacing()));
          knot_spacing_floored = std::max(knot_spacing_floored, 1);
          knot_spacing_floored = std::min(knot_spacing_floored, 4);
          if (i_iteration == 0){
            regulariser =
                std::make_shared<MMORF::CostFxnBendingEnergy>(
                  warp_field_,
                  knot_spacing_floored);
          }
          else{
            if (knot_spacing > 3.9f){
              regulariser =
                  std::make_shared<MMORF::CostFxnLogJacobianSingularValues>(
                    warp_field_,
                    knot_spacing_floored);
            }
            else{
              regulariser =
                  std::make_shared<MMORF::CostFxnLogJacobianSingularValuesDiagHess>(
                    warp_field_,
                    knot_spacing_floored);
            }
          }
          // Create the single compound cost function
          //auto cost_fxn_compound = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
          auto cost_fxn_compound = MMORF::CostFxnCompoundWarpField(
              cost_fxns,
              lambda_volumes_[i_iteration],
              regulariser,
              lambda_regularisation_[i_iteration],
              warp_field_);
          // Create linear solver
          auto linear_solver = std::make_shared<MMORF::LinearSolverCusp>();
          // Choose what type of optimiser to use
          auto optimiser = std::unique_ptr<MMORF::Optimiser>(nullptr);
          auto max_iter = 0;
          if (knot_spacing >= 3.9f){
            optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
            max_iter = 5;
          }
          else{
            optimiser = MMORF::make_unique<MMORF::OptimiserLevenberg>(linear_solver);
            //optimiser = MMORF::make_unique<MMORF::OptimiserScaledConjugateGradient>();
            max_iter = 5;
          }
          // Create optimiser
          //auto optimiser = MMORF::OptimiserLevenbergMarquardt();
          //auto optimiser = MMORF::OptimiserLevenberg();
          // Solve system
          optimised_parameters = optimiser->optimise(
              cost_fxn_compound,
              optimised_parameters,
              max_iter,
              1.0e-3f);
          // Update parameters of warp field
          warp_field_->set_parameters(optimised_parameters);
          // Save results if required
          if (save_all_steps){
            save_warp_field(folder_name + "/warp_iter_"
                + std::to_string(i_iteration));
            save_warped_volumes(folder_name + "/vol_iter_"
                + std::to_string(i_iteration));
            save_jacobian_determinants(folder_name + "/jacobian_iter_"
                + std::to_string(i_iteration));
          }
        }
        //auto final_params = warp_field_->get_parameters();
        //auto x_params = arma::fvec(final_params[0]);
        //auto y_params = arma::fvec(final_params[1]);
        //auto z_params = arma::fvec(final_params[2]);
        //x_params.save("data/DTI/x_params.txt",arma::raw_ascii);
        //y_params.save("data/DTI/y_params.txt",arma::raw_ascii);
        //z_params.save("data/DTI/z_params.txt",arma::raw_ascii);
        //auto final_jacobians = warp_field_->get_jacobian_elements(warp_field_->get_robust_sample_positions(4));
        //auto j11 = arma::fvec(final_jacobians[0]);
        //auto j12 = arma::fvec(final_jacobians[1]);
        //auto j13 = arma::fvec(final_jacobians[2]);
        //auto j21 = arma::fvec(final_jacobians[3]);
        //auto j22 = arma::fvec(final_jacobians[4]);
        //auto j23 = arma::fvec(final_jacobians[5]);
        //auto j31 = arma::fvec(final_jacobians[6]);
        //auto j32 = arma::fvec(final_jacobians[7]);
        //auto j33 = arma::fvec(final_jacobians[8]);
        //j11.save("data/DTI/j11.txt",arma::raw_ascii);
        //j12.save("data/DTI/j12.txt",arma::raw_ascii);
        //j13.save("data/DTI/j13.txt",arma::raw_ascii);
        //j21.save("data/DTI/j21.txt",arma::raw_ascii);
        //j22.save("data/DTI/j22.txt",arma::raw_ascii);
        //j23.save("data/DTI/j23.txt",arma::raw_ascii);
        //j31.save("data/DTI/j31.txt",arma::raw_ascii);
        //j32.save("data/DTI/j32.txt",arma::raw_ascii);
        //j33.save("data/DTI/j33.txt",arma::raw_ascii);
      }
      /// Save warp field as 4D nifti
      void save_warp_field(std::string filename_warp_field)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(filename_warp_space_);
        auto warp_samples = warp_field_->sample_warp(
            warp_space_vol.get_original_sample_positions());
        MMORF::save_as_nifti(
            warp_samples,
            warp_space_vol.get_dimensions(),
            header_warp_space_,
            filename_warp_field);
      }
      /// Save warped moving volumes
      void save_warped_volumes(std::string warp_prefix)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(filename_warp_space_);
        for (auto i = 0; i < n_volumes_; ++i){
          auto vol_mov = MMORF::VolumeTensor(volumes_moving_[i]);
          auto sample_positions = warp_field_->apply_warp_then_affine(
              warp_space_vol.get_original_sample_positions(), affine_matrices_moving_[i]);
          //MMORF::save_as_nifti(
          //    vol_mov,
          //    sample_positions,
          //    warp_space_vol.get_dimensions(),
          //    header_warp_space_,
          //    warp_prefix + std::to_string(i));
        }
      }
      /// Save Jacobian determinant of the warp field
      void save_jacobian_determinants(std::string filename_jacobian)
      {
        auto warp_space_vol = MMORF::VolumeBSpline(filename_warp_space_);
        //std::vector<float> jacobian =
        auto jacobian =
          warp_field_->jacobian_determinants(warp_space_vol.get_original_sample_positions());
        MMORF::save_as_nifti(
            jacobian,
            warp_space_vol.get_dimensions(),
            header_warp_space_,
            filename_jacobian);
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Create a warp field covering the warp space
      void create_warp_field_(
          const std::string& filename_warp_space,
          const float        knot_spacing)
      {
        auto reference_volume = MMORF::VolumeBSpline(filename_warp_space);
        extents_warp_space_ = reference_volume.get_extents();
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            extents_warp_space_,
            knot_spacing);
      }
      /// Read in reference newimage volumes
      void read_volumes_reference_(const std::vector<std::string>& filename_volumes_reference)
      {
        for (const auto& filename : filename_volumes_reference){
          auto volume = NEWIMAGE::volume4D<float>();
          NEWIMAGE::read_volume4D(volume, filename);
          volumes_reference_.push_back(volume);
        }
      }
      /// Read in moving newimage volumes
      void read_volumes_moving_(const std::vector<std::string>& filename_volumes_moving)
      {
        for (const auto& filename : filename_volumes_moving){
          auto volume = NEWIMAGE::volume4D<float>();
          NEWIMAGE::read_volume4D(volume, filename);
          volumes_moving_.push_back(volume);
        }
      }
      /// Read in reference mask newimage volumes
      void read_masks_reference_(const std::vector<std::string>& filename_masks_reference)
      {
        for (const auto& filename : filename_masks_reference){
          auto volume = NEWIMAGE::volume<float>();
          NEWIMAGE::read_volume(volume, filename);
          masks_reference_.push_back(volume);
        }
      }
      /// \details We are making the assumption that the matrices were calculated using FLIRT,
      ///          and therefore describe a transform between the volumes in "scaled mm
      ///          space". In MMORF we make the assumption that all coordinates and positions
      ///          are given in "real world mm space". As such, there is some serious
      ///          shuffling that needs to be done in order to get the FLIRT affine matrices
      ///          into a form that works with MOORF's coordinate system.
      void create_affine_matrices_(
          const std::vector<std::string>& filename_affine_transforms_reference,
          const std::vector<std::string>& filename_affine_transforms_moving,
          const bool                      affines_are_inverted)
      {
        // Read in ref matrices
        for (auto i = 0; i < volumes_reference_.size(); ++i){
          auto affine_mat_ref = flirt_mat_to_real_mat_(
              header_warp_space_,
              volumes_reference_[i],
              filename_affine_transforms_reference[i],
              affines_are_inverted);
          affine_matrices_reference_.push_back(affine_mat_ref);
        }
        // Read in mov matrices
        for (auto i = 0; i < volumes_moving_.size(); ++i){
          // Read in reference volume header
          auto affine_mat_ref = flirt_mat_to_real_mat_(
              header_warp_space_,
              volumes_moving_[i],
              filename_affine_transforms_moving[i],
              affines_are_inverted);
          affine_matrices_moving_.push_back(affine_mat_ref);
        }
      }
      /// Take a matrix output by flirt and convert it to real world coordinates
      arma::fmat flirt_mat_to_real_mat_(
          NEWIMAGE::volume<float>& vol_source,
          NEWIMAGE::volume4D<float>& vol_dest,
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
          affine_invert_(flirt_mat);
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
      /// Calculate in inverse matrix for an affine transform
      void affine_invert_(arma::fmat& affine_mat) const
      {
        affine_mat.submat(0,0,2,2) = affine_mat.submat(0,0,2,2).i();
        affine_mat.submat(0,3,2,3) =
            -affine_mat.submat(0,0,2,2)
            *affine_mat.submat(0,3,2,3);
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
      float calculate_sampling_frequency_(
          const unsigned int i_iteration,
          const unsigned int i_volume,
          const float        max_resolution) const
      {
        auto knot_spacing = warp_field_->get_knot_spacing();
        auto smooth_fwhm_ref = smoothing_reference_[i_iteration][i_volume];
        auto smooth_fwhm_mov = smoothing_moving_[i_iteration][i_volume];
        auto sampling_freq_ref = (int)0;
        auto sampling_freq_mov = (int)0;
        // Calculate sampling frequency for ref
        if (smooth_fwhm_ref > 0){
          sampling_freq_ref = static_cast<int>(std::ceil(knot_spacing/smooth_fwhm_ref));
        }
        sampling_freq_ref = std::max(sampling_freq_ref, 1);
        sampling_freq_ref = std::min(sampling_freq_ref, 5);
        // Calculate sampling frequency for mov
        if (smooth_fwhm_mov > 0){
          sampling_freq_mov = static_cast<int>(std::ceil(knot_spacing/smooth_fwhm_mov));
        }
        sampling_freq_mov = std::max(sampling_freq_mov, 1);
        sampling_freq_mov = std::min(sampling_freq_mov, 5);
        // Calculate the maximum between the ref and mov samples
        auto sampling_freq_final = std::max(sampling_freq_ref, sampling_freq_mov);
        // If sampling rate would mean we sample at a higher resolution than out highest
        // resolution volume, reduce sampling frequency to 1
        if (static_cast<float>(sampling_freq_final) > knot_spacing/max_resolution){
          sampling_freq_final = 1;
        }
        return sampling_freq_final;
      }
      /// Create a vector of cost functions for each pair of volumes
      std::vector<std::shared_ptr<MMORF::CostFxn> > create_cost_fxns_(
          const unsigned int i_iteration,
          const float knot_spacing)
      {
        auto cost_fxns = std::vector<std::shared_ptr<MMORF::CostFxn> >();
        for (auto i_volume = 0; i_volume < n_volumes_; ++i_volume){
          auto ref_vol = std::make_shared<MMORF::VolumeTensor>(
              volumes_reference_[i_volume],
              smoothing_reference_[i_iteration][i_volume]);
          auto mov_vol = std::make_shared<MMORF::VolumeTensor>(
              volumes_moving_[i_volume],
              smoothing_moving_[i_iteration][i_volume]);
          auto mask_vol = std::make_shared<MMORF::VolumeBSpline>(
              masks_reference_[i_volume],
              smoothing_reference_[i_iteration][i_volume]);
          auto max_res = std::max(get_max_resolution_(ref_vol),get_max_resolution_(mov_vol));
          auto sampling_frequency = calculate_sampling_frequency_(
              i_iteration,
              i_volume,
              max_res);
          if (knot_spacing > 3.9f){
            cost_fxns.push_back(
                std::make_shared<MMORF::CostFxnTensorL2WarpFieldMasked>(
                  ref_vol,
                  mov_vol,
                  affine_matrices_reference_[i_volume],
                  affine_matrices_moving_[i_volume],
                  mask_vol,
                  warp_field_,
                  sampling_frequency));
          }
          else{
            cost_fxns.push_back(
                std::make_shared<MMORF::CostFxnTensorL2WarpFieldMasked>(
                  ref_vol,
                  mov_vol,
                  affine_matrices_reference_[i_volume],
                  affine_matrices_moving_[i_volume],
                  mask_vol,
                  warp_field_,
                  sampling_frequency));
          }
        }
        return cost_fxns;
      }
      /// Calculate the highest resolution in 1D for a given volume
      float get_max_resolution_(const std::shared_ptr<MMORF::VolumeTensor> vol)
      {
        auto res = vol->get_resolution();
        auto max_res = std::max(std::max(std::abs(res[0]),std::abs(res[1])),std::abs(res[2]));
        return max_res;
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      unsigned int                             n_volumes_;
      unsigned int                             n_iterations_;
      std::string                              filename_warp_space_;
      std::vector<std::vector<float> >         lambda_volumes_;
      std::vector<std::vector<float> >         smoothing_reference_;
      std::vector<std::vector<float> >         smoothing_moving_;
      std::vector<float>                       lambda_regularisation_;
      std::vector<int>                         warp_scaling_;
      std::vector<NEWIMAGE::volume4D<float> >  volumes_reference_;
      std::vector<NEWIMAGE::volume4D<float> >  volumes_moving_;
      std::vector<NEWIMAGE::volume<float> >    masks_reference_;
      std::vector<arma::fmat>                  affine_matrices_reference_;
      std::vector<arma::fmat>                  affine_matrices_moving_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      NEWIMAGE::volume<float>                  header_warp_space_;
      std::pair<
        std::vector<float>,
        std::vector<float> >                   extents_warp_space_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  RegistrationCoordinatorTensor::~RegistrationCoordinatorTensor() = default;
  /// Move ctor
  RegistrationCoordinatorTensor::RegistrationCoordinatorTensor(RegistrationCoordinatorTensor&& rhs) = default;
  /// Move assignment operator
  RegistrationCoordinatorTensor& RegistrationCoordinatorTensor::operator=(RegistrationCoordinatorTensor&& rhs) = default;
  /// Copy ctor
  RegistrationCoordinatorTensor::RegistrationCoordinatorTensor(const RegistrationCoordinatorTensor& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  RegistrationCoordinatorTensor& RegistrationCoordinatorTensor::operator=(const RegistrationCoordinatorTensor& rhs)
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
  RegistrationCoordinatorTensor::RegistrationCoordinatorTensor(
      const std::string&                      filename_warp_space,
      // Parameters the size of the number of volumes being used
      const std::vector<std::string>&         filename_volumes_reference,
      const std::vector<std::string>&         filename_affine_transforms_reference,
      const std::vector<std::string>&         filename_volumes_moving,
      const std::vector<std::string>&         filename_affine_transforms_moving,
      const std::vector<std::string>&         filename_masks_reference,
      // Parameters the size of the number of iterations being performed
      const std::vector<std::vector<float> >& lambda_volumes,
      const std::vector<std::vector<float> >& smoothing_reference,
      const std::vector<std::vector<float> >& smoothing_moving,
      const std::vector<float>&               lambda_regularisation,
      const std::vector<int>&                 warp_scaling,
      // Paramters with a single value
      const float                             knot_spacing_initial,
      const bool                              affines_are_inverted
      )
    : pimpl_(
        MMORF::make_unique<Impl>(
          filename_warp_space,
          filename_volumes_reference,
          filename_affine_transforms_reference,
          filename_volumes_moving,
          filename_affine_transforms_moving,
          filename_masks_reference,
          lambda_volumes,
          smoothing_reference,
          smoothing_moving,
          lambda_regularisation,
          warp_scaling,
          knot_spacing_initial,
          affines_are_inverted
          )
        )
  {}
  /// Run through the registration process
  void RegistrationCoordinatorTensor::register_volumes(bool save_all_steps, std::string folder_name)
  {
    pimpl_->register_volumes(save_all_steps, folder_name);
  }
  /// Save warp field as 4D nifti
  void RegistrationCoordinatorTensor::save_warp_field(std::string filename_warp_field)
  {
    pimpl_->save_warp_field(filename_warp_field);
  }
  /// Save warped moving volumes
  void RegistrationCoordinatorTensor::save_warped_volumes(std::string warp_prefix)
  {
    pimpl_->save_warped_volumes(warp_prefix);
  }
  /// Save Jacobian determinant of the warp field
  void RegistrationCoordinatorTensor::save_jacobian_determinants(std::string filename_jacobian)
  {
    pimpl_->save_jacobian_determinants(filename_jacobian);
  }
} // MMORF
