// Unit Test File

#include "CostFxnCompoundWarpFieldVarianceScaled.cuh"
#include "CostFxnSSDWarpField.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "Volume.h"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class CostFxnCompoundWarpFieldVarianceScaledTest : public ::testing::Test
  {
    public:
      CostFxnCompoundWarpFieldVarianceScaledTest()
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
        auto zero_params = warp_field_->get_parameters();
        auto small_params = std::vector<std::vector<float> >(
            zero_params.size(),
            std::vector<float>(zero_params[0].size(),1e-2f));
        warp_field_->set_parameters(small_params);

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
  }; // CostFxnCompoundWarpFieldVarianceScaled

  TEST_F(CostFxnCompoundWarpFieldVarianceScaledTest, func_get_parameters)
  {
    auto vol_1 = std::make_shared<MMORF::VolumeBSpline>(vol_ref_mo_);
    auto vol_2 = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_1,
        vol_2,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
        std::vector<std::shared_ptr<MMORF::CostFxn> >({my_cost_fxn}),
        std::vector<float>({1.0f}),
        my_reg,
        lambda_reg_,
        warp_field_);
    auto parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),1e-2f));
    EXPECT_EQ(parameters,expected_parameters);
  } // func_get_parameters

  TEST_F(CostFxnCompoundWarpFieldVarianceScaledTest, func_set_parameters)
  {
    auto vol_1 = std::make_shared<MMORF::VolumeBSpline>(vol_ref_mo_);
    auto vol_2 = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_1,
        vol_2,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
        std::vector<std::shared_ptr<MMORF::CostFxn> >({my_cost_fxn}),
        std::vector<float>({1.0f}),
        my_reg,
        lambda_reg_,
        warp_field_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto new_parameters = my_reg_cf.get_parameters();
    EXPECT_EQ(new_parameters,expected_parameters);
  } // func_set_parameters

  TEST_F(CostFxnCompoundWarpFieldVarianceScaledTest, func_cost)
  {
    auto vol_1 = std::make_shared<MMORF::VolumeBSpline>(vol_ref_mo_);
    auto vol_2 = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_1,
        vol_2,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
        std::vector<std::shared_ptr<MMORF::CostFxn> >({my_cost_fxn}),
        std::vector<float>({1.0f}),
        my_reg,
        lambda_reg_,
        warp_field_);
    auto cost = my_reg_cf.cost();
    //EXPECT_EQ(cost,0);
    auto parameters = my_reg_cf.get_parameters();
    auto new_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),1));
    my_reg_cf.set_parameters(new_parameters);
    auto new_cost = my_reg_cf.cost();
    EXPECT_NE(new_cost,cost);
  } // func_cost

  TEST_F(CostFxnCompoundWarpFieldVarianceScaledTest, func_grad)
  {
    auto vol_1 = std::make_shared<MMORF::VolumeBSpline>(vol_ref_mo_);
    auto vol_2 = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_1,
        vol_2,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
        std::vector<std::shared_ptr<MMORF::CostFxn> >({my_cost_fxn}),
        std::vector<float>({1.0f}),
        my_reg,
        lambda_reg_,
        warp_field_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto jte = my_reg_cf.grad();
    //jte.save("data_test_CostFxnCompoundWarpFieldVarianceScaled/grad.txt",arma::raw_ascii);
  } // func_jte

  TEST_F(CostFxnCompoundWarpFieldVarianceScaledTest, func_hess)
  {
    auto vol_1 = std::make_shared<MMORF::VolumeBSpline>(vol_ref_mo_);
    auto vol_2 = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_1,
        vol_2,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnCompoundWarpFieldVarianceScaled(
        std::vector<std::shared_ptr<MMORF::CostFxn> >({my_cost_fxn}),
        std::vector<float>({1.0f}),
        my_reg,
        lambda_reg_,
        warp_field_);
    auto original_parameters = my_reg_cf.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),1));
    my_reg_cf.set_parameters(expected_parameters);
    auto jtj = my_reg_cf.hess();
    //auto jtj_arma = jtj.convert_to_csc();
    //jtj_arma.save("data_test_CostFxnCompoundWarpFieldVarianceScaled/hess_from_shared_warpfield.txt",arma::coord_ascii);
  } // func_jtj

} // namespace

