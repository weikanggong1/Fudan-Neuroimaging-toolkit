// Unit Test File

#include "CostFxnRegularised.cuh"
#include "CostFxnSSDWarpField.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
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
        : filename_mov_("./data/MNI152_T1_2mm.nii.gz")
        , filename_ref_("./data/MNI152_T1_2mm.nii.gz")
        , vol_mov_mo_(filename_mov_)
        , vol_ref_mo_(filename_ref_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
      {
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            vol_ref_mo_.get_extents(),
            knot_spacing_);
        NEWIMAGE::read_volume(vol_ref_ni_,filename_ref_);
        NEWIMAGE::read_volume(vol_mov_ni_,filename_mov_);
      }
    protected:
      const std::string filename_ref_;
      const std::string filename_mov_;
      MMORF::VolumeBSpline vol_mov_mo_;
      MMORF::VolumeBSpline vol_ref_mo_;
      arma::Mat<float> affine_;
      const float knot_spacing_ = 10.0f;
      const int sampling_frequency_ = 5;
      const float lambda_reg_ = 1e6f;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      NEWIMAGE::volume<float> vol_ref_ni_;
      NEWIMAGE::volume<float> vol_mov_ni_;
  }; // CostFxnSSDWarpFieldTest

  TEST_F(CostFxnSSDWarpFieldTest, func_get_parameters)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    auto parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),0));
    EXPECT_EQ(parameters,expected_parameters);
  } // func_get_parameters

  TEST_F(CostFxnSSDWarpFieldTest, func_set_parameters)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto new_parameters = my_reg_cf.get_parameters();
    EXPECT_EQ(new_parameters,expected_parameters);
  } // func_set_parameters

  TEST_F(CostFxnSSDWarpFieldTest, func_cost)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    auto cost = my_reg_cf.cost();
    EXPECT_EQ(cost,0);
    auto parameters = my_reg_cf.get_parameters();
    auto new_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),1));
    my_reg_cf.set_parameters(new_parameters);
    auto new_cost = my_reg_cf.cost();
    EXPECT_NE(new_cost,0);
  } // func_cost

  TEST_F(CostFxnSSDWarpFieldTest, func_grad)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto jte = my_reg_cf.grad();
    //jte.save("data_test_CostFxnRegularised/grad.txt",arma::raw_ascii);
  } // func_jte

  TEST_F(CostFxnSSDWarpFieldTest, func_hess)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto jtj = my_reg_cf.hess();
    //auto jtj_arma = jtj.convert_to_csc();
    //jtj_arma.save("data_test_CostFxnRegularised/hess_from_shared_warpfield.txt",arma::coord_ascii);
  } // func_jtj

} // namespace

