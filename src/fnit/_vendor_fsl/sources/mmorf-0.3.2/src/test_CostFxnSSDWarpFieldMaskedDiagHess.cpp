// Unit Test File

#include "CostFxnSSDWarpFieldMasked.cuh"
#include "CostFxnSSDWarpFieldMaskedDiagHess.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "Volume.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class CostFxnSSDWarpFieldMaskedDiagHessTest : public ::testing::Test
  {
    public:
      CostFxnSSDWarpFieldMaskedDiagHessTest()
        : reference_filename_("./data/NIREP/na01/na01_cropped.nii.gz")
        , moving_filename_("./data/NIREP/na02/na02_cropped.nii.gz")
        , mask_filename_("./data/NIREP/na_mask.nii")
        , reference_vol_(reference_filename_)
        , moving_vol_(moving_filename_)
        , mask_vol_(mask_filename_)
        , affine_{
          {1.051976546,0.01969135107,-8.186239832e-05,-7.38074941},
          {-0.006709146721,1.031131345,0.07429944165,-10.94198449},
          {0.05688616047,-0.02706785188,1.016494529,0.05605069444},
          {0,0,0,1}}
      {
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            reference_vol_.get_extents(),
            knot_spacing_);
      }
    protected:
      const float knot_spacing_ = 20.0f;
      const int sampling_frequency_ = 5;
      const std::string reference_filename_;
      const std::string moving_filename_;
      const std::string mask_filename_;
      MMORF::VolumeBSpline reference_vol_;
      MMORF::VolumeBSpline moving_vol_;
      MMORF::VolumeBSpline mask_vol_;
      arma::Mat<float> affine_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
  }; // CostFxnSSDWarpFieldMaskedDiagHessTest

  TEST_F(CostFxnSSDWarpFieldMaskedDiagHessTest, func_hess)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpFieldMasked(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        warp_field_,
        sampling_frequency_);
    auto my_cost_fxn_diag = MMORF::CostFxnSSDWarpFieldMaskedDiagHess(
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        std::make_shared<MMORF::VolumeBSpline>(moving_vol_),
        affine_,
        affine_,
        std::make_shared<MMORF::VolumeBSpline>(reference_vol_),
        warp_field_,
        sampling_frequency_);
    auto jtj = my_cost_fxn.hess();
    auto jtj_diag = my_cost_fxn_diag.hess();
    auto jtj_arma = jtj.convert_to_csc();
    auto jtj_arma_diag = jtj_diag.convert_to_csc();
    jtj_arma.save("data_test_CostFxnSSDWarpFieldMaskedDiagHess/hess_full.txt",arma::coord_ascii);
    jtj_arma_diag.save("data_test_CostFxnSSDWarpFieldMaskedDiagHess/hess_diag.txt",arma::coord_ascii);
  } // func_jtj
} // namespace
