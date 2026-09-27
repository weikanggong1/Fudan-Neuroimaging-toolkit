// Unit Test File

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include "MMORFio.h"
#include "CostFxn.h"
#include "LinearSolverCusp.cuh"
#include "OptimiserLevenbergMarquardt.cuh"
#include "CostFxnBendingEnergy.cuh"
#include "CostFxnSSDWarpField.cuh"
#include "CostFxnRegularised.cuh"
#include "WarpFieldBSpline.cuh"
#include "Volume.h"
#include "VolumeBSpline.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstdlib>

namespace
{
  template<typename T>
  arma::fmat affine_newmat_to_arma(const T& newmat_mat)
  {
    auto arma_mat = arma::Mat<float>(4, 4, arma::fill::zeros);
    for (auto j = 0; j < 4; ++j){
      for (auto i = 0; i < 4; ++i){
        arma_mat(i,j) = newmat_mat.element(i,j);
      }
    }
    return arma_mat;
  }

  arma::fmat flirt_mat_to_real_mat(
      std::string file_vol_ref,
      std::string file_vol_mov,
      std::string file_mat_flirt,
      bool        mat_inverted)
  {
    auto header_ref = NEWIMAGE::volume<float>();
    auto header_mov = NEWIMAGE::volume<float>();
    NEWIMAGE::read_volume_hdr_only(header_ref, file_vol_ref);
    NEWIMAGE::read_volume_hdr_only(header_mov, file_vol_mov);
    auto vox_to_mm_ref = affine_newmat_to_arma(header_ref.newimagevox2mm_mat());
    auto vox_to_mm_mov = affine_newmat_to_arma(header_mov.newimagevox2mm_mat());
    auto sampling_ref = affine_newmat_to_arma(header_ref.sampling_mat());
    auto sampling_mov = affine_newmat_to_arma(header_mov.sampling_mat());
    auto flirt_mat = arma::fmat();
    flirt_mat.load(file_mat_flirt, arma::raw_ascii);
    if (mat_inverted){
        flirt_mat.submat(0,0,2,2) = flirt_mat.submat(0,0,2,2).i();
        flirt_mat.submat(0,3,2,3) =
            -flirt_mat.submat(0,0,2,2)*flirt_mat.submat(0,3,2,3);
    }
    arma::fmat real_mat =
      vox_to_mm_mov
      * sampling_mov.i()
      * flirt_mat
      * sampling_ref
      * vox_to_mm_ref.i();
    return real_mat;
  }

  std::vector<std::vector<float> > add_parameters(
      const std::vector<std::vector<float> >& p1,
      arma::fvec& p2)
  {
    auto result = std::vector<std::vector<float> >(p1);
    for (auto dim = 0; dim < p1.size(); ++dim){
      for (auto param = 0; param < p1[0].size(); ++param){
        result[dim][param] += p2[dim*p1[0].size() + param];
      }
    }
    return result;
  }

  void save_parameters(
      const std::vector<std::vector<float> >& parameters,
      std::string filename)
  {
    auto all_params = arma::fmat(parameters[0].size(),parameters.size());
    auto col = 0;
    for (const auto& params : parameters){
      all_params.col(col) = arma::fvec(params);
      ++col;
    }
    all_params.save(filename,arma::raw_ascii);
  }

  class LMOptimisation : public ::testing::Test
  {
    public:
      LMOptimisation()
        : filename_mov_("./data/MNI152_T1_2mm_smooth_10mmFWHM.nii.gz")
        , filename_ref_("./data/MNI152_T1_2mm_smooth_10mmFWHM_warped.nii.gz")
        , filename_real_("./data/patient_x_flirt_smooth_10mmFWHM.nii.gz")
        , filename_mov_hires_("./data/MNI152_T1_2mm.nii.gz")
        , filename_ref_hires_("./data/MNI152_T1_2mm_warped.nii.gz")
        , filename_real_hires_("./data/patient_x_robust_flirt.nii.gz")
        , filename_real_full_res_("./data/patient_x_robust.nii.gz")
        , filename_warp_params_("./data/actual_warp_params")
        , filename_real_affine_("./data/patient_x_robust_flirt.mat")
        , vol_mov_mo_(filename_mov_)
        , vol_ref_mo_(filename_ref_)
        , vol_real_mo_(filename_real_)
        , vol_mov_hires_mo_(filename_mov_hires_)
        , vol_ref_hires_mo_(filename_ref_hires_)
        , vol_real_hires_mo_(filename_real_hires_)
        , vol_real_full_res_mo_(filename_real_full_res_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
        , real_affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
      {
        warp_field_ = std::make_shared<MMORF::WarpFieldBSpline>(
            vol_ref_mo_.get_extents(),
            knot_spacing_);
        real_affine_ = flirt_mat_to_real_mat(
            filename_ref_hires_,
            filename_real_full_res_,
            filename_real_affine_,
            true);
        NEWIMAGE::read_volume(vol_mov_ni_,filename_mov_);
        NEWIMAGE::read_volume(vol_ref_ni_,filename_ref_);
        // Generate set of valid sample points
        generate_sample_points();
        // Generate random warp parameters
        generate_random_parameters();
        // Create dummy reference volume
      }

      void generate_sample_points()
      {
        std::vector<float> x_points;
        std::vector<float> y_points;
        std::vector<float> z_points;
        for (auto z = sample_start_z; z <= sample_end_z; z+=2){
          for (auto y = sample_start_y; y <= sample_end_y; y+=2){
            for (auto x = sample_start_x; x >= sample_end_x; x-=2){
              x_points.push_back((static_cast<float>(x) + 0.0)/1.0);
              y_points.push_back((static_cast<float>(y) + 0.0)/1.0);
              z_points.push_back((static_cast<float>(z) + 0.0)/1.0);
            }
          }
        }
        sample_points_.push_back(x_points);
        sample_points_.push_back(y_points);
        sample_points_.push_back(z_points);
      }

      void generate_random_parameters()
      {
        auto params = arma::fmat();
        params.load(filename_warp_params_);
        for (auto col = 0; col < 3; ++col){
          actual_warp_params_.push_back(arma::conv_to<std::vector<float> >::from(params.col(col)));
        }
      }

    protected:
      const float knot_spacing_ = 10.0f;
      const int sampling_frequency_ = 5;
      const std::string filename_mov_;
      const std::string filename_ref_;
      const std::string filename_real_;
      const std::string filename_mov_hires_;
      const std::string filename_ref_hires_;
      const std::string filename_real_hires_;
      const std::string filename_real_full_res_;
      const std::string filename_warp_params_;
      const std::string filename_real_affine_;
      MMORF::VolumeBSpline vol_mov_mo_;
      MMORF::VolumeBSpline vol_ref_mo_;
      MMORF::VolumeBSpline vol_real_mo_;
      MMORF::VolumeBSpline vol_mov_hires_mo_;
      MMORF::VolumeBSpline vol_ref_hires_mo_;
      MMORF::VolumeBSpline vol_real_hires_mo_;
      MMORF::VolumeBSpline vol_real_full_res_mo_;
      arma::Mat<float> affine_;
      arma::Mat<float> real_affine_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      NEWIMAGE::volume<float> vol_mov_ni_;
      NEWIMAGE::volume<float> vol_ref_ni_;
      std::vector<std::vector<float> > sample_points_;
      std::vector<std::vector<float> > actual_warp_params_;
      const int sample_start_x = 90;
      const int sample_end_x = -90;
      const int sample_start_y = -126;
      const int sample_end_y = 90;
      const int sample_start_z = -72;
      const int sample_end_z = 108;
      const float lambda_reg_ = 1e4f;

  }; // LMOptimisation

/*
  TEST_F(LMOptimisation, func_optimise_costfxn_bending_energy)
  {
    auto my_regulariser = MMORF::CostFxnBendingEnergy(
        warp_field_,
        sampling_frequency_);
    auto my_optimiser = MMORF::OptimiserLevenbergMarquardt();
    auto final_params = my_optimiser.optimise(my_regulariser, actual_warp_params_);
    save_parameters(final_params,"test_OptimiserLevenbergMarquardt_data/final_params_bending_energy");
  } // func_optimise_costfxn_bending_energy
*/
/*
  TEST_F(LMOptimisation, func_optimise_costfxn_ssd_warp_field)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_optimiser = MMORF::OptimiserLevenbergMarquardt();
    auto init_params = my_cost_fxn.get_parameters();
    auto my_linear_solver = MMORF::LinearSolverCusp();
    auto final_params = my_optimiser.optimise(my_cost_fxn, init_params, my_linear_solver);

    my_cost_fxn = MMORF::CostFxnSSDWarpField(
        vol_ref_hires_mo_,
        vol_mov_hires_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    final_params = my_optimiser.optimise(my_cost_fxn, final_params, my_linear_solver);

    save_parameters(final_params,"data_test_OptimiserLevenbergMarquardt/final_params_ssd_warp_field");
    warp_field_->set_parameters(final_params);
    auto warped_samples = warp_field_->apply_warp(sample_points_);
    // Save smoothed registration result
    MMORF::save_as_nifti(
        vol_mov_mo_,
        warped_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_ssd_warp_field_smooth"));
    // Save high resolution registration result
    MMORF::save_as_nifti(
        vol_mov_hires_mo_,
        warped_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_ssd_warp_field"));
    // Save final warp fields
    std::vector<std::vector<float> > warp_samples = warp_field_->sample_warp(sample_points_);
    MMORF::save_as_nifti(
        warp_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/ssd_warp_field_warp"));
    // Save Jacobian determinant of final warp field
    std::vector<float> jacobian_determinants = warp_field_->jacobian_determinants(sample_points_);
    MMORF::save_as_nifti(
        jacobian_determinants,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/ssd_warp_field_jacobian_determinants"));
  } // func_optimise_costfxn_ssd_warp_field
*/
/*
  TEST_F(LMOptimisation, func_optimise_costfxn_regularised)
  {
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    auto my_regulariser = std::make_shared<MMORF::CostFxnBendingEnergy>(
        warp_field_,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_regulariser,
        lambda_reg_);
    auto my_optimiser = MMORF::OptimiserLevenbergMarquardt();
    auto my_linear_solver = MMORF::LinearSolverCusp();
    auto init_params = my_reg_cf.get_parameters();
    auto final_params = my_optimiser.optimise(my_reg_cf, init_params, my_linear_solver);

    my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_ref_hires_mo_,
        vol_mov_hires_mo_,
        affine_,
        affine_,
        warp_field_,
        sampling_frequency_);
    my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_regulariser,
        lambda_reg_);
        //0.01*lambda_reg_);
    final_params = my_optimiser.optimise(my_reg_cf, final_params, my_linear_solver);

    save_parameters(
        final_params,
        "data_test_OptimiserLevenbergMarquardt/final_params_regularised");
    warp_field_->set_parameters(final_params);
    auto warped_samples = warp_field_->apply_warp(sample_points_);
    MMORF::save_as_nifti(
        vol_mov_mo_,
        warped_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_regularised_smooth"));
    MMORF::save_as_nifti(
        vol_mov_hires_mo_,
        warped_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_regularised"));
    // Save final warp fields
    std::vector<std::vector<float> > warp_samples = warp_field_->sample_warp(sample_points_);
    MMORF::save_as_nifti(
        warp_samples,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/regularised_warp"));
    // Save Jacobian determinant of final warp field
    std::vector<float> jacobian_determinants = warp_field_->jacobian_determinants(sample_points_);
    MMORF::save_as_nifti(
        jacobian_determinants,
        std::vector<int>{
            vol_ref_ni_.xsize(),
            vol_ref_ni_.ysize(),
            vol_ref_ni_.zsize()},
        vol_ref_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/regularised_jacobian_determinants"));
  } // func_optimise_costfxn_ssd_warp_field
*/

  TEST_F(LMOptimisation, func_optimise_costfxn_regularised_real)
  {
    vol_mov_mo_ = MMORF::VolumeBSpline(
        filename_mov_hires_, 3.0*(knot_spacing_/sampling_frequency_));
    vol_real_mo_ = MMORF::VolumeBSpline(
        filename_real_full_res_, 3.0*(knot_spacing_/sampling_frequency_));
    auto vol_mov_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    auto vol_real_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_real_mo_);
    auto real_warp_field = std::make_shared<MMORF::WarpFieldBSpline>(
        vol_mov_mo_.get_extents(),
        knot_spacing_);
    auto my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_mov_mo_ptr,
        vol_real_mo_ptr,
        affine_,
        real_affine_,
        real_warp_field,
        sampling_frequency_);
    auto my_regulariser = std::make_shared<MMORF::CostFxnBendingEnergy>(
        real_warp_field,
        sampling_frequency_);
    auto my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_regulariser,
        500.0f*lambda_reg_);
    auto my_linear_solver = std::make_shared<MMORF::LinearSolverCusp>();
    auto my_optimiser = MMORF::OptimiserLevenbergMarquardt(my_linear_solver);
    auto init_params = my_reg_cf.get_parameters();
    auto final_params = my_optimiser.optimise(my_reg_cf, init_params);

    vol_mov_mo_ = MMORF::VolumeBSpline(
        filename_mov_hires_, 2.0f*(knot_spacing_/sampling_frequency_));
    vol_real_mo_ = MMORF::VolumeBSpline(
        filename_real_full_res_, 2.0f*(knot_spacing_/sampling_frequency_));
    vol_mov_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    vol_real_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_real_mo_);
    my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_mov_mo_ptr,
        vol_real_mo_ptr,
        affine_,
        real_affine_,
        real_warp_field,
        sampling_frequency_);
    my_regulariser = std::make_shared<MMORF::CostFxnBendingEnergy>(
        real_warp_field,
        sampling_frequency_);
    my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_regulariser,
        250.0f*lambda_reg_);
    final_params = my_optimiser.optimise(my_reg_cf, final_params);

    vol_mov_mo_ = MMORF::VolumeBSpline(
        filename_mov_hires_, knot_spacing_/sampling_frequency_);
    vol_real_mo_ = MMORF::VolumeBSpline(
        filename_real_full_res_, knot_spacing_/sampling_frequency_);
    vol_mov_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_mov_mo_);
    vol_real_mo_ptr = std::make_shared<MMORF::VolumeBSpline>(vol_real_mo_);
    my_cost_fxn = std::make_shared<MMORF::CostFxnSSDWarpField>(
        vol_mov_mo_ptr,
        vol_real_mo_ptr,
        affine_,
        real_affine_,
        real_warp_field,
        sampling_frequency_);
    my_regulariser = std::make_shared<MMORF::CostFxnBendingEnergy>(
        real_warp_field,
        sampling_frequency_);
    my_reg_cf = MMORF::CostFxnRegularised(
        my_cost_fxn,
        my_regulariser,
        50.0f*lambda_reg_);
    final_params = my_optimiser.optimise(my_reg_cf, final_params);

    save_parameters(
        final_params,
        "data_test_OptimiserLevenbergMarquardt/final_params_real");

    real_warp_field->set_parameters(final_params);
    auto warped_samples = real_warp_field->apply_warp_then_affine(sample_points_,real_affine_);
    MMORF::save_as_nifti(
        vol_real_mo_,
        warped_samples,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_real_smooth"));

    vol_real_mo_ = MMORF::VolumeBSpline(
        filename_real_full_res_, 2.0f);
    MMORF::save_as_nifti(
        vol_real_mo_,
        warped_samples,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/final_vol_real"));
    // Save final warp fields
    std::vector<std::vector<float> > warp_samples = real_warp_field->sample_warp(sample_points_);

    MMORF::save_as_nifti(
        warp_samples,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/real_warp"));
    // Save Jacobian determinant of final warp field
    std::vector<float> jacobian_determinants = real_warp_field->jacobian_determinants(sample_points_);
    MMORF::save_as_nifti(
        jacobian_determinants,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/real_jacobian_determinants"));
  } // func_optimise_costfxn_regularised_real


  TEST_F(LMOptimisation, test_affine)
  {
    auto real_warp_field = std::make_shared<MMORF::WarpFieldBSpline>(
        vol_mov_hires_mo_.get_extents(),
        knot_spacing_);
    auto warped_points = real_warp_field->apply_warp_then_affine(sample_points_,real_affine_);

    MMORF::save_as_nifti(
        vol_real_full_res_mo_,
        warped_points,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data_test_OptimiserLevenbergMarquardt/patient_x_robust_affine"));
  }

} // namespace
