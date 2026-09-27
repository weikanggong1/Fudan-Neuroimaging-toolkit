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
#ifndef REGISTRATION_COORDINATOR_MULTIMODAL_CUH
#define REGISTRATION_COORDINATOR_MULTIMODAL_CUH

#include <memory>
#include <vector>
#include <string>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Enum for selection the optimisation strategy
  enum OptimiserType{
    /// Levenberg-Marquardt
    LM,
    /// Majorise-Minimise
    MM,
    /// Sclaled Conjugate Gradient
    SCG
  };

  class RegistrationCoordinatorMultimodal
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~RegistrationCoordinatorMultimodal();
      /// Move ctor
      RegistrationCoordinatorMultimodal(RegistrationCoordinatorMultimodal&& rhs);
      /// Move assignment operator
      RegistrationCoordinatorMultimodal& operator=(RegistrationCoordinatorMultimodal&& rhs);
      /// Copy ctor
      RegistrationCoordinatorMultimodal(const RegistrationCoordinatorMultimodal& rhs);
      /// Copy assignment operator
      RegistrationCoordinatorMultimodal& operator=(const RegistrationCoordinatorMultimodal& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Complete ctor
      /// \details The constructor is used to completely define all the parameters for the
      ///          registration algorithm, or "recipe". Certain parameters must be defined
      ///          for each volume being registered, while others need to be defined for each
      ///          iteration of the optimisation. Note that all volumes should have been
      ///          affine registered to the warp space beforehand.
      ///
      /// \param warp_res_init  Initial resolution of warp field (mm isotropic)
      ///
      /// \param warp_scaling Scaling factor for the knot spacing at each iteration. These
      ///                     scalings compound, therefore ensure that the final knot
      ///                     spacing remains sensible.
      ///
      /// \param f_img_warp_space Volume filename representing the reference space to which
      ///                         all  volumes have been affinely registered.
      ///                         Note that this MAY be the same volume as one of the
      ///                         other reference volumes. This is additionally the
      ///                         "native" space for the final warp field.
      ///
      /// \param f_img_ref_scalar List of all the scalar reference volume filenames
      ///
      /// \param f_img_mov_scalar List of all the scalar moving volume filenames
      ///
      /// \param f_aff_ref_scalar Affine transforms FROM reference space
      ///                         TO each reference volume
      ///
      /// \param f_aff_mov_scalar Affine transforms FROM reference space TO
      ///                         each moving volume
      ///
      /// \param f_mask_ref_scalar Volume mask in the reference image space
      ///
      /// \param f_mask_mov_scalar Volume mask in the moving image space
      ///
      /// \param fwhm_ref_scalar  Amount of Gaussian blur to apply for each
      ///                         iteration of the optimisation (FWHM in mm)
      ///
      /// \param fwhm_mov_scalar  Amount of Gaussian blur to apply for each
      ///                         iteration of the optimisation (FWHM in mm)
      ///
      /// \param lambda_scalar  Relative weighting of the cost function for each volume
      ///
      /// \param f_img_ref_tensor List of all the tensor reference volume filenames
      ///
      /// \param f_img_mov_tensor List of all the tensor moving volume filenames
      ///
      /// \param f_aff_ref_tensor Affine transforms FROM reference space
      ///                         TO each reference volume
      ///
      /// \param f_aff_mov_tensor Affine transforms FROM reference space TO
      ///                         each moving volume
      ///
      /// \param f_mask_ref_tensor Volume mask in the reference image space
      ///
      /// \param f_mask_mov_tensor Volume mask in the moving image space
      ///
      /// \param fwhm_ref_tensor  Amount of Gaussian blur to apply for each
      ///                         iteration of the optimisation (FWHM in mm)
      ///
      /// \param fwhm_mov_tensor  Amount of Gaussian blur to apply for each
      ///                         iteration of the optimisation (FWHM in mm)
      ///
      /// \param lambda_tensor  Relative weighting of the cost function for each volume
      ///
      /// \param smoothing_reference Amount of Gaussian blur to apply for each
      ///                            iteration of the optimisation (FWHM in mm)
      ///
      /// \param lambda_reg Relative weighting of the regularisation term of the
      ///                   cost function for each iteration of the optimisation
      ///
      /// \param affines_are_inverted Determines whether or not the affine transform needs to
      ///                             be inverted. Note: FSL by default saves a
      ///                             moving_to_reference transform and we will therefore need
      ///                             to be inverted
      RegistrationCoordinatorMultimodal(
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
          const bool                              debug=false
          // Miscelaneous parameters
          //const bool                              affines_are_inverted
          );
      /// Run through the registration process
      //void register_volumes(bool save_all_steps = false, std::string folder_name = ".");
      void register_volumes();
      /// Save warp field as 4D nifti
      void save_warp_field(std::string f_warp_field_out);
      /// Save bias field
      void save_bias_field(std::string f_bias_field_out);
      /// Save warped moving volumes
      //void save_warped_volumes(std::string warp_prefix);
      /// Save Jacobian determinant of the warp field
      void save_jacobian_determinants(std::string f_jacobian_out);
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // RegistrationCoordinatorMultimodal
} // MMORF
#endif // REGISTRATION_COORDINATOR_MULTIMODAL_CUH
