// Unit Test File

#include "CostFxnSSDWarpField.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class CostFxnSSDWarpFieldTest : public ::testing::Test
  {
    public:
      CostFxnSSDWarpFieldTest()
        : reference_filename_("./data/MNI152_T1_2mm.nii.gz")
        , moving_filename_("./data/MNI152_T1_2mm.nii.gz")
        , reference_vol_(reference_filename_)
        , moving_vol_(moving_filename_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
      {
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            reference_vol_.get_extents(),
            knot_spacing_);
      }
    protected:
      const float knot_spacing_ = 10.0f;
      const int sampling_frequency_ = 5;
      const std::string reference_filename_;
      const std::string moving_filename_;
      MMORF::VolumeBSpline reference_vol_;
      MMORF::VolumeBSpline moving_vol_;
      arma::Mat<float> affine_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
  }; // CostFxnSSDWarpFieldTest

  TEST_F(CostFxnSSDWarpFieldTest, func_get_parameters)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto parameters = my_cost_fxn.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),0));
    EXPECT_EQ(parameters,expected_parameters);
  } // func_get_parameters

  TEST_F(CostFxnSSDWarpFieldTest, func_set_parameters)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto original_parameters = my_cost_fxn.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_cost_fxn.set_parameters(expected_parameters);
    auto new_parameters = my_cost_fxn.get_parameters();
    EXPECT_EQ(new_parameters,expected_parameters);
  } // func_set_parameters

  TEST_F(CostFxnSSDWarpFieldTest, func_cost)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto cost = my_cost_fxn.cost();
    EXPECT_EQ(cost,0);
    auto parameters = my_cost_fxn.get_parameters();
    auto new_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),1));
    my_cost_fxn.set_parameters(new_parameters);
    auto new_cost = my_cost_fxn.cost();
    EXPECT_NE(new_cost,0);
  } // func_cost

  TEST_F(CostFxnSSDWarpFieldTest, func_grad)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto original_parameters = my_cost_fxn.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_cost_fxn.set_parameters(expected_parameters);
    auto jte = my_cost_fxn.grad();
    //jte.save("data/grad.txt",arma::raw_ascii);
  } // func_jte
  TEST_F(CostFxnSSDWarpFieldTest, func_hess)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto original_parameters = my_cost_fxn.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_cost_fxn.set_parameters(expected_parameters);
    auto jtj = my_cost_fxn.hess();
    //auto jtj_arma = jtj.convert_to_csc();
    //jtj_arma.save("data/hess_from_sdmt.txt",arma::coord_ascii);
  } // func_jtj
} // namespace
