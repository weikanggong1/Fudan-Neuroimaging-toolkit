// Unit Test File

#include "CostFxnTensorL2WarpFieldMasked.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "VolumeTensor.cuh"
#include "Volume.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <utility>
#include <memory>
#include <iostream>

namespace
{
  class CostFxnTensorL2WarpFieldMaskedTest : public ::testing::Test
  {
    public:
      CostFxnTensorL2WarpFieldMaskedTest()
        : reference_filename_("./data/DTI/dti_reference.nii.gz")
        , moving_filename_("./data/DTI/dti_rot_x.nii.gz")
        , mask_filename_("./data/DTI/dti_mask.nii.gz")
        , reference_vol_(reference_filename_)
        , moving_vol_(moving_filename_)
        , mask_vol_(mask_filename_)
        , affine_{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1}}
      {
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            mask_vol_.get_extents(),
            knot_spacing_);
      }
    protected:
      const float knot_spacing_ = 4.0f;
      const int sampling_frequency_ = 4;
      const std::string reference_filename_;
      const std::string moving_filename_;
      const std::string mask_filename_;
      MMORF::VolumeTensor reference_vol_;
      MMORF::VolumeTensor moving_vol_;
      MMORF::VolumeBSpline mask_vol_;
      arma::Mat<float> affine_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
  }; // CostFxnTensorL2WarpFieldMaskedTest

  TEST_F(CostFxnTensorL2WarpFieldMaskedTest, func_cost)
  {
    auto my_cost_fxn = MMORF::CostFxnTensorL2WarpFieldMasked(
        std::make_shared<MMORF::VolumeTensor>(reference_vol_),
        std::make_shared<MMORF::VolumeTensor>(moving_vol_),
        affine_,
        affine_.i(),
        std::make_shared<MMORF::VolumeBSpline>(mask_vol_),
        warp_field_,
        sampling_frequency_);
    auto cost = my_cost_fxn.cost();
    auto parameters = my_cost_fxn.get_parameters();
    auto new_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(), 5));
    my_cost_fxn.set_parameters(new_parameters);
    auto new_cost = my_cost_fxn.cost();
    //EXPECT_LT(cost,new_cost);
    std::cout << "Zero cost = " << cost << "\nShifted cost = " << new_cost << "\n";
  } // func_cost

  TEST_F(CostFxnTensorL2WarpFieldMaskedTest, func_grad)
  {
    auto my_cost_fxn = MMORF::CostFxnTensorL2WarpFieldMasked(
        std::make_shared<MMORF::VolumeTensor>(reference_vol_),
        std::make_shared<MMORF::VolumeTensor>(moving_vol_),
        affine_,
        affine_.i(),
        std::make_shared<MMORF::VolumeBSpline>(mask_vol_),
        warp_field_,
        sampling_frequency_);
    auto jte = my_cost_fxn.grad();
    jte.save("data/DTI/grad_mmorf.txt",arma::raw_ascii);
  } // func_grad

  TEST_F(CostFxnTensorL2WarpFieldMaskedTest, func_hess)
  {
    auto my_cost_fxn = MMORF::CostFxnTensorL2WarpFieldMasked(
        std::make_shared<MMORF::VolumeTensor>(reference_vol_),
        std::make_shared<MMORF::VolumeTensor>(moving_vol_),
        affine_,
        affine_.i(),
        std::make_shared<MMORF::VolumeBSpline>(mask_vol_),
        warp_field_,
        sampling_frequency_);
    auto jtj = my_cost_fxn.hess();
    auto jtj_arma = jtj.convert_to_csc();
    jtj_arma.save("data/DTI/hess_mmorf.txt",arma::coord_ascii);
  } // func_jtj

} // namespace
