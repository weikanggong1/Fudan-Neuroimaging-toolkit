// Unit Test File

#include "CostFxnLogJacobianSingularValues.cuh"
#include "CostFxnLogJacobianSingularValuesDiagHess.cuh"
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
        : reference_filename_("./data/NIREP/na01/na01_cropped.nii.gz")
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
      const float knot_spacing_ = 20.0f;
      const int sampling_frequency_ = 5;
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
    params_0.save("data_test_CostFxnLogJacobianDiagHess/params.txt",arma::raw_ascii);
  } // func_get_parameters

  TEST_F(CostFxnLogJacobianSingularValuesTest, func_hess)
  {
    auto my_cost_fxn = MMORF::CostFxnLogJacobianSingularValues(
        warp_field_,
        sampling_frequency_);
    auto my_cost_fxn_diag = MMORF::CostFxnLogJacobianSingularValuesDiagHess(
        warp_field_,
        sampling_frequency_);
    my_cost_fxn.set_parameters(params_);
    my_cost_fxn_diag.set_parameters(params_);
    auto jtj = my_cost_fxn.hess();
    auto jtj_diag = my_cost_fxn_diag.hess();
    std::cout << "Parameters per warp: " << warp_field_->get_parameter_size().second << std::endl;
    auto jtj_arma = jtj.convert_to_csc();
    auto jtj_arma_diag = jtj_diag.convert_to_csc();
    jtj_arma.save("data_test_CostFxnLogJacobianDiagHess/hess_full.txt",arma::coord_ascii);
    jtj_arma_diag.save("data_test_CostFxnLogJacobianDiagHess/hess_diag.txt",arma::coord_ascii);
  } // func_jtj

} // namespace
