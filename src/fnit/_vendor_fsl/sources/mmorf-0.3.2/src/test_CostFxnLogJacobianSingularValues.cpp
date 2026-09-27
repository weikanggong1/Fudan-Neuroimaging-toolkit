// Unit Test File

#include "CostFxnLogJacobianSingularValues.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>
#include <cstdlib>

namespace
{
  class CostFxnLogJacobianSingularValuesTest : public ::testing::Test
  {
    public:
      CostFxnLogJacobianSingularValuesTest()
        : reference_filename_("./data/test_vol.nii")
        , reference_vol_(reference_filename_)
      {
        srand(24);
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            reference_vol_.get_extents(),
            knot_spacing_);

        auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
            warp_field_,
            sampling_frequency_);
        auto original_parameters = my_cost_fxn.get_parameters();
        params_ = std::vector<std::vector<float> >(
            original_parameters.size(),
            std::vector<float>(original_parameters[0].size()));
        for (auto& dim : params_){
          for (auto& param : dim){
            param = 4.0f*((float)rand()/float(RAND_MAX) - 0.5f);
          }
        }
      }
    protected:
      const std::string reference_filename_;
      MMORF::VolumeBSpline reference_vol_;
      const float knot_spacing_ = 4.0f;
      const int sampling_frequency_ = 4;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      std::vector<std::vector<float> >params_;
  }; // CostFxnLogJacobianSingularValuesTest

  TEST_F(CostFxnLogJacobianSingularValuesTest, save_params)
  {
    auto params_0 = arma::fvec(params_[0]);
    auto params_1 = arma::fvec(params_[1]);
    auto params_2 = arma::fvec(params_[2]);
    params_0 = arma::join_cols(params_0,params_1);
    params_0 = arma::join_cols(params_0,params_2);
    params_0.save("data_test_CostFxnLogJacobian/params.txt",arma::raw_ascii);

  } // func_get_parameters

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_get_parameters)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
        warp_field_,
        sampling_frequency_);
    auto parameters = my_cost_fxn.get_parameters();
    auto expected_parameters = std::vector<std::vector<float> >(
        parameters.size(),
        std::vector<float>(parameters[0].size(),0));
    EXPECT_EQ(parameters,expected_parameters);
  } // func_get_parameters

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_set_parameters)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
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

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_cost)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
        warp_field_,
        sampling_frequency_);
    auto cost = my_cost_fxn.cost();
    std::cout << "Cost is: " << cost << ", should be 0\n";
    EXPECT_LT(cost,1.0e-3f);
    my_cost_fxn.set_parameters(params_);
    auto new_cost = my_cost_fxn.cost();
    std::cout << "Cost is: " << new_cost << ", should be big\n";
    EXPECT_NE(new_cost,0);
  } // func_cost

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_grad)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
        warp_field_,
        sampling_frequency_);
    my_cost_fxn.set_parameters(params_);
    auto jte = my_cost_fxn.grad();
    jte.save("data_test_CostFxnLogJacobian/grad.txt",arma::raw_ascii);
  } // func_jte

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_hess)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
        warp_field_,
        sampling_frequency_);
    my_cost_fxn.set_parameters(params_);
    auto jtj = my_cost_fxn.hess();
    std::cout << "Parameters per warp: " << warp_field_->get_parameter_size().second << std::endl;
    auto jtj_arma = jtj.convert_to_csc();
    jtj_arma.save("data_test_CostFxnLogJacobian/hess.txt",arma::coord_ascii);
  } // func_jtj

} // namespace
