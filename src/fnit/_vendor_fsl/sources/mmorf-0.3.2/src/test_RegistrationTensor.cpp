// Unit Test File

#include "MmorfMemory.h"
#include "RegistrationCoordinatorTensor.cuh"
#include "gtest/gtest.h"

namespace
{

  class RegistrationCoordinatorTest : public ::testing::Test
  {
    public:
      RegistrationCoordinatorTest()
        //: filename_warp_space_("./data/DTI/dti_mask.nii.gz")
        //, filename_volumes_reference_({"./data/DTI/dti_reference.nii.gz"})
        //, filename_affine_transforms_reference_({"./data/MNI_flirt.mat"})
        //, filename_volumes_moving_({"./data/DTI/dti_rot_x.nii.gz"})
        //, filename_affine_transforms_moving_({"./data/MNI_flirt.mat"})
        //, filename_masks_reference_({"./data/DTI/dti_mask.nii.gz"})
        : filename_warp_space_("./data/HCP/100307_diffusion/T1w/T1w_acpc_dc_restore_1.25.nii.gz")
        , filename_volumes_reference_({"./data/HCP/100307_diffusion/T1w/Diffusion/dti_tensor.nii.gz"})
        , filename_affine_transforms_reference_({"./data/MNI_flirt.mat"})
        , filename_volumes_moving_({"./data/HCP/102614_diffusion/T1w/Diffusion/dti_tensor.nii.gz"})
        , filename_affine_transforms_moving_({"./data/HCP/102614_diffusion/T1w/102614_to_100307.mat"})
        , filename_masks_reference_({"./data/HCP/100307_diffusion/T1w/100307_mask.nii.gz"})
        , lambda_volumes_({
            std::vector<float>({10.0e9f}),
            std::vector<float>({10.0e9f}),
            std::vector<float>({10.0e9f}),
            std::vector<float>({10.0e9f}),
            std::vector<float>({10.0e9f})
            //std::vector<float>({1000.0e9f}),
            //std::vector<float>({10000.0e9f})
            })
        , smoothing_reference_({
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({20.0f/4.0f}),
            std::vector<float>({10.0f/4.0f}),
            std::vector<float>({5.0f/4.0f})
            //std::vector<float>({0.0f/4.0f}),
            //std::vector<float>({0.0f/4.0f})
            })
        , smoothing_moving_({
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({40.0f/4.0f}),
            std::vector<float>({20.0f/4.0f}),
            std::vector<float>({10.0f/4.0f}),
            std::vector<float>({5.0f/4.0f})
            //std::vector<float>({0.0f/4.0f}),
            //std::vector<float>({0.0f/4.0f})
            })
        , lambda_regularisation_({3.2e5f, 4.0e-1f, 3.0e-1f, 2.0e-1f, 1.0e-1f})
        //, lambda_regularisation_({3.2e6f, 1.0e-1f})
        , warp_scaling_({1, 1, 2, 2, 2})
        //, warp_scaling_({1, 1})
        , knot_spacing_initial_(40.0f)
        //, knot_spacing_initial_(4.0f)
        , affines_are_inverted_(true)
      {
        reg_coordinator_ = MMORF::make_unique<MMORF::RegistrationCoordinatorTensor>(
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
      std::unique_ptr<MMORF::RegistrationCoordinatorTensor> reg_coordinator_;

  }; // RegistrationCoordinatorTest


  TEST_F(RegistrationCoordinatorTest, func_register_volumes)
  {
    reg_coordinator_->register_volumes();
    //reg_coordinator_->register_volumes(true,"data_test_RegistrationMICCAI");
    reg_coordinator_->save_warp_field("data_test_RegistrationTensor/final_warp_field");
    reg_coordinator_->save_jacobian_determinants("data_test_RegistrationTensor/final_jacobian_field");
    //reg_coordinator_->save_warped_volumes("data_test_RegistrationMICCAI/final_warped_");
  } // func_optimise_costfxn_bending_energy

} // namespace
