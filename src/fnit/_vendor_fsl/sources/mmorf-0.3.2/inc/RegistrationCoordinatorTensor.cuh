//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Coordinate the exection of a volumetric registration
/// \details This class is responsible for coordinating the objects required to follow a
///          particular registration "recipe". This involves things like iterating over
///          different levels of smoothing, regularisation, etc.
/// \author Frederik Lange
/// \date July 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef REGISTRATION_COORDINATOR_CUH
#define REGISTRATION_COORDINATOR_CUH

#include <memory>
#include <vector>
#include <string>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class RegistrationCoordinatorTensor
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~RegistrationCoordinatorTensor();
      /// Move ctor
      RegistrationCoordinatorTensor(RegistrationCoordinatorTensor&& rhs);
      /// Move assignment operator
      RegistrationCoordinatorTensor& operator=(RegistrationCoordinatorTensor&& rhs);
      /// Copy ctor
      RegistrationCoordinatorTensor(const RegistrationCoordinatorTensor& rhs);
      /// Copy assignment operator
      RegistrationCoordinatorTensor& operator=(const RegistrationCoordinatorTensor& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Complete ctor
      /// \details The constructor is used to completely define all the parameters for the
      ///          registration algorithm, or "recipe". Certain parameters must be defined
      ///          for each volume being registered, while others need to be defined for each
      ///          iteration of the optimisation. Note that this assumes that all reference
      ///          volumes have been coregistered beforehand (and similary so for the moving
      ///          volumes). Additionally an affine registration between the reference and
      ///          moving volumes should already have been calculated.
      /// \param filename_warp_space Volume representing the reference space to which
      ///                            all reference volumes have been affine registered.
      ///                            note that this MAY be the same volume as one of the
      ///                            other reference volumes. This is additionally the
      ///                            "native" space for the final warp field.
      /// \param filename_volumes_reference List of all the reference volume file names
      /// \param filename_affine_transforms_reference Affine transforms FROM reference space
      ///                                             TO each reference volume
      /// \param filename_volumes_moving List of all the moving volume file names
      /// \param filename_affine_transforms_moving Affine transforms FROM reference space TO
      ///                                          each moving volume
      /// \param filename_makes_reference Volume mask in the reference inmage space
      /// \param lambda_volumes Relative weighting of the cost function for each volume
      /// \param smoothing_reference Amount of Gaussian blur to apply for each
      ///                            iteration of the optimisation (FWHM in mm)
      /// \param smoothing_moving Amount of Gaussian blur to apply for each iteration
      ///                         of the optimisatoin (FMWH in mm)
      /// \param lambda_regularisation Relative weighting of the regularisation term of the
      ///                              cost function for each iteration of the optimisation
      /// \param warp_scaling Scaling factor for the knot spacing at each iteration. These
      ///                     scalings compound, therefore ensure that the final knot
      ///                     spacing remains sensible.
      /// \param knot_spacing B-spline knot spacing. Note this is the same in all directions
      /// \param filename_affine_transform Name of file containting the pre-calculated affine
      ///                                  transform between the reference and moving images
      /// \param affines_are_inverted Determines whether or not the affine transform needs to
      ///                             be inverted. Note: FSL by default saves a
      ///                             moving_to_reference transform and we will therefore need
      ///                             to be inverted
      RegistrationCoordinatorTensor(
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
          );
      /// Run through the registration process
      void register_volumes(bool save_all_steps = false, std::string folder_name = ".");
      /// Save warp field as 4D nifti
      void save_warp_field(std::string filename_warp_field);
      /// Save warped moving volumes
      void save_warped_volumes(std::string warp_prefix);
      /// Save Jacobian determinant of the warp field
      void save_jacobian_determinants(std::string filename_jacobian);
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // RegistrationCoordinatorTensor
} // MMORF
#endif // REGISTRATION_COORDINATOR_CUH
