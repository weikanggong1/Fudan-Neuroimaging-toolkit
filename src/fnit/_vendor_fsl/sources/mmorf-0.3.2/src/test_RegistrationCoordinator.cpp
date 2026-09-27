// Unit Test File

#include "MmorfMemory.h"
#include "RegistrationCoordinator.cuh"
#include "gtest/gtest.h"

namespace
{

  class RegistrationCoordinatorTest : public ::testing::Test
  {
    public:
      RegistrationCoordinatorTest()
        //: filename_warp_space_("./data/MNI152_T1_1mm.nii.gz")
        //, filename_volumes_moving_({"./data/HCP_T1_2_norm.nii.gz"})
        //, filename_affine_transforms_moving_({"./data/HCP_T1_2_flirt.mat"})
        //, filename_volumes_reference_({"./data/HCP_T1_norm.nii.gz"})
        //, filename_affine_transforms_reference_({"./data/HCP_T1_flirt.mat"})
        //, filename_masks_reference_({"./data/HCP_T1_brain_mask.nii.gz"})
        : filename_warp_space_("./data/NIREP/na01/na01_cropped.nii.gz")
        , filename_volumes_reference_({"./data/NIREP/na01/na01_cropped.nii.gz"})
        //, filename_volumes_reference_({"./data/HCP_T1_2_norm.nii.gz"})
        , filename_affine_transforms_reference_({"./data/MNI_flirt.mat"})
        //, filename_affine_transforms_reference_({"./data/HCP_T1_2_flirt.mat"})
        , filename_volumes_moving_({"./data/NIREP/na02/na02_cropped.nii.gz"})
        , filename_affine_transforms_moving_({"./data/NIREP/na02/na02_to_na01.mat"})
        , filename_masks_reference_({"./data/NIREP/na_mask.nii"})
        //, filename_masks_reference_({"./data/HCP_T1_2_brain_mask.nii.gz"})
        //: filename_warp_space_("./data/MNI152_T1_1mm.nii.gz")
        //, filename_volumes_reference_({"./data/HCP_T1.nii.gz"})
        //, filename_affine_transforms_reference_({"./data/HCP_T1_flirt.mat"})
        //, filename_volumes_moving_({"./data/HCP_T1_2.nii.gz"})
        //, filename_affine_transforms_moving_({"./data/HCP_T1_2_flirt.mat"})
        //, filename_masks_reference_({"./data/HCP_T1_brain_mask.nii.gz"})
        //: filename_warp_space_("./data/MNI152_T1_1mm.nii.gz")
        //, filename_volumes_reference_({"./data/MNI152_T1_1mm_brain.nii.gz"})
        //, filename_affine_transforms_reference_({"./data/affine_identity.mat"})
        //, filename_volumes_moving_({"./data/HCP_T1_brain.nii.gz"})
        //, filename_affine_transforms_moving_({"./data/HCP_T1_flirt.mat"})
        , lambda_volumes_({
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f}),
            std::vector<float>({1.0f})
            })
        , smoothing_reference_({
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({20.0f/4.0f}),
            std::vector<float>({10.0f/4.0f}),
            std::vector<float>({5.0f/4.0f}),
            std::vector<float>({2.5f/4.0f}),
            std::vector<float>({1.25f/4.0f})
            })
        , smoothing_moving_({
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({20.0f/4.0f}),
            std::vector<float>({10.0f/4.0f}),
            std::vector<float>({5.0f/4.0f}),
            std::vector<float>({2.5f/4.0f}),
            std::vector<float>({1.25f/4.0f})
            })
        , lambda_regularisation_({1.6e5f, 1.5e-1f, 1.1125e-1f, 8.4375e-2f, 6.328125e-2f, 4.746094e-2f, 3.55957e-2f})
        //, lambda_regularisation_({1.6e5f, 8.0e4f, 4.0e4f, 2.0e4f, 1.0e4f, 5.0e3f})
        //, lambda_regularisation_({1.6e5f, 4.0e-1f, 4.0e-1f})
        //, lambda_regularisation_({1.0e4f, 1.0e4f, 1.0e4f,})
        , warp_scaling_({1, 1, 2, 2, 2, 2, 2})
        //, warp_scaling_({1, 2, 2})
        , knot_spacing_initial_(40.0f)
        , affines_are_inverted_(true)
      {
        reg_coordinator_ = MMORF::make_unique<MMORF::RegistrationCoordinator>(
            filename_warp_space_,
            filename_volumes_reference_,
            filename_affine_transforms_reference_,
            filename_volumes_moving_,
            filename_affine_transforms_moving_,
            filename_masks_reference_,
            lambda_volumes_,
            smoothing_reference_,
            smoothing_moving_,
            lambda_regularisation_,
            warp_scaling_,
            knot_spacing_initial_,
            affines_are_inverted_);
      }

    protected:
      const std::string                               filename_warp_space_;
      // Parameters the size of the number of volumes being used
      const std::vector<std::string>                  filename_volumes_reference_;
      const std::vector<std::string>                  filename_affine_transforms_reference_;
      const std::vector<std::string>                  filename_volumes_moving_;
      const std::vector<std::string>                  filename_affine_transforms_moving_;
      const std::vector<std::string>                  filename_masks_reference_;
      // Parameters the size of the number of iterations being performed
      const std::vector<std::vector<float> >          lambda_volumes_;
      const std::vector<std::vector<float> >          smoothing_reference_;
      const std::vector<std::vector<float> >          smoothing_moving_;
      const std::vector<float>                        lambda_regularisation_;
      const std::vector<int>                          warp_scaling_;
      // Paramters with a single value
      const float                                     knot_spacing_initial_;
      const bool                                      affines_are_inverted_;
      // Actual registration coordinator
      std::unique_ptr<MMORF::RegistrationCoordinator> reg_coordinator_;

  }; // RegistrationCoordinatorTest


  TEST_F(RegistrationCoordinatorTest, func_register_volumes)
  {
    reg_coordinator_->register_volumes(true,"data_test_RegistrationCoordinator");
    reg_coordinator_->save_warp_field("data_test_RegistrationCoordinator/final_warp_field");
    reg_coordinator_->save_warped_volumes("data_test_RegistrationCoordinator/final_warped_");
  } // func_optimise_costfxn_bending_energy

} // namespace
