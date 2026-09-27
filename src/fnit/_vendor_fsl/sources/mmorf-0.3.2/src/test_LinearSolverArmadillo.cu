// Unit Test File

#include "LinearSolverArmadillo.cuh"
#include "CostFxnRegularised.cuh"
#include "CostFxnSSDWarpField.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"

#include "newimage/newimageall.h"

#include "gtest/gtest.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class LinearSolverArmadilloTest : public ::testing::Test
  {
    public:
      LinearSolverArmadilloTest()
        : filename_mov_("./data/MNI152_T1_2mm_smooth_10mmFWHM.nii.gz")
        , filename_ref_("./data/MNI152_T1_2mm_smooth_10mmFWHM_warped.nii.gz")
        , vol_mov_mo_(filename_mov_)
        , vol_ref_mo_(filename_ref_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
        , warp_field_(vol_ref_mo_.get_extents(),knot_spacing_,affine_)
      {
        NEWIMAGE::read_volume(vol_ref_ni_,filename_ref_);
        NEWIMAGE::read_volume(vol_mov_ni_,filename_mov_);
      }
    protected:
      const std::string filename_mov_;
      const std::string filename_ref_;
      MMORF::VolumeBSpline vol_mov_mo_;
      MMORF::VolumeBSpline vol_ref_mo_;
      arma::Mat<float> affine_;
      const float knot_spacing_ = 20.0f;
      const int sampling_frequency_ = 5;
      const float lambda_reg_ = 1e6f;
      MMORF::WarpFieldBSpline warp_field_;
      NEWIMAGE::volume<float> vol_ref_ni_;
      NEWIMAGE::volume<float> vol_mov_ni_;
  }; // LinearSolverArmadilloTest

  TEST_F(LinearSolverArmadilloTest, class_LinearSolverArmadill_func_solve)
  {
    std::cout << "1\n";
    auto linear_solver_armadillo = MMORF::LinearSolverArmadillo();
    std::cout << "2\n";
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        knot_spacing_,
        sampling_frequency_);
    std::cout << "3\n";
    auto my_reg = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    std::cout << "4\n";
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_reg,
        lambda_reg_);
    std::cout << "5\n";
    auto original_parameters = my_reg_cf.get_parameters();
    std::cout << "6\n";
    auto expected_parameters = std::vector<std::vector<float> >(
        original_parameters.size(),
        std::vector<float>(original_parameters[0].size(),0));
    my_reg_cf.set_parameters(expected_parameters);
    std::cout << "7\n";
    auto jtj = my_reg_cf.hess();
    std::cout << "8\n";
    auto jte = my_reg_cf.grad();
    std::cout << "9\n";
    auto solution = linear_solver_armadillo.solve(jtj, -jte);
    solution.save("data_test_LinearSolver/solution_arma.txt",arma::raw_ascii);
  } // class_LinearSolverArmadill_func_solve
} // namespace
