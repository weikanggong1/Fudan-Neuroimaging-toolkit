// Unit Test File

#include "MMORFio.h"
#include "CostFxnBendingEnergy.cuh"
#include "CostFxnSSDWarpField.cuh"
#include "WarpFieldBSpline.cuh"
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

  class BasicOptimisation : public ::testing::Test
  {
    public:
      BasicOptimisation()
        : filename_mov_("./data/MNI152_T1_2mm_smooth_10mmFWHM.nii.gz")
        , filename_ref_("./data/MNI152_T1_2mm_smooth_10mmFWHM_warped.nii.gz")
        , filename_warp_params_("./data/actual_warp_params")
        , vol_mov_mo_(filename_mov_)
        , vol_ref_mo_(filename_ref_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
        , warp_field_(vol_mov_mo_.get_extents(),knot_spacing_,affine_)
      {
        NEWIMAGE::read_volume(vol_mov_ni_,filename_mov_);
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
      const float knot_spacing_ = 20.0f;
      const int sampling_frequency_ = 10;
      const std::string filename_mov_;
      const std::string filename_ref_;
      const std::string filename_warp_params_;
      MMORF::VolumeBSpline vol_mov_mo_;
      MMORF::VolumeBSpline vol_ref_mo_;
      arma::Mat<float> affine_;
      MMORF::WarpFieldBSpline warp_field_;
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

  }; // BasicOptimisation

  TEST_F(BasicOptimisation, basic_test)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        knot_spacing_,
        sampling_frequency_);
    auto my_regulariser = MMORF::CostFxnBendingEnergy(
        warp_field_,
        sampling_frequency_);
    auto parameters = my_cost_fxn.get_parameters();
    MMORF::save_as_nifti(
        vol_mov_mo_,
        sample_points_,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data/moving_vol"));
    MMORF::save_as_nifti(
        vol_ref_mo_,
        sample_points_,
        std::vector<int>{
            vol_mov_ni_.xsize(),
            vol_mov_ni_.ysize(),
            vol_mov_ni_.zsize()},
        vol_mov_ni_,
        std::string("data/reference_vol"));
    // Try combinations of the actual warp and see the cost
    for (float i = 0; i < 2.1; i+=0.1){
      auto params_guess = std::vector<std::vector<float> >();
      for (const auto& sub_params : actual_warp_params_){
        auto sub_params_guess = std::vector<float>();
        for (const auto& param : sub_params){
          sub_params_guess.push_back(i*param);
        }
        std::cout << sub_params_guess[0] << std::endl;
        params_guess.push_back(sub_params_guess);
      }
      my_cost_fxn.set_parameters(params_guess);
      my_regulariser.set_parameters(params_guess);
      warp_field_.set_parameters(params_guess);
      auto warped_points = warp_field_.apply_warp(sample_points_);
      if (i > 0.99f && i < 1.01f){
        MMORF::save_as_nifti(
            vol_mov_mo_,
            warped_points,
            std::vector<int>{
                vol_mov_ni_.xsize(),
                vol_mov_ni_.ysize(),
                vol_mov_ni_.zsize()},
            vol_mov_ni_,
            std::string("data/unwarped_" + std::to_string(i)));
      }
      std::cout << "For i = " << i << ":\n";
      std::cout << "cf cost = " << my_cost_fxn.cost() << std::endl;
      std::cout << "re cost = " << my_regulariser.cost() << std::endl << std::endl;
    }
  } // func_basic_test

  TEST_F(BasicOptimisation, linear_solve)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        knot_spacing_,
        sampling_frequency_);
    auto my_regulariser = MMORF::CostFxnBendingEnergy(
        warp_field_,
        sampling_frequency_);
    auto parameters = my_cost_fxn.get_parameters();
    //auto ones = std::vector<std::vector<float> >(parameters.size(),std::vector<float>(parameters[0].size(),1.0f));
    //my_cost_fxn.set_parameters(actual_warp_params_);
    //my_regulariser.set_parameters(actual_warp_params_);
    //my_cost_fxn.set_parameters(ones);
    //my_regulariser.set_parameters(ones);
    auto cf_hess = my_cost_fxn.hess();
    cf_hess.save("test_BasicOptimisation_data/cf_hess_zero.txt",arma::coord_ascii);
    auto be_hess = my_regulariser.hess();
    be_hess.save("test_BasicOptimisation_data/be_hess_zero.txt",arma::coord_ascii);
    auto cf_grad = my_cost_fxn.grad();
    cf_grad.save("test_BasicOptimisation_data/cf_grad_zero.txt",arma::raw_ascii);
    auto be_grad = my_regulariser.grad();
    be_grad.save("test_BasicOptimisation_data/be_grad_zero.txt",arma::raw_ascii);
    //arma::fvec result = arma::spsolve(cf_hess + be_hess, -(cf_grad + be_grad));
    //result.save("data/linear_solve_actual.txt",arma::raw_ascii);
  } // func_basic_test

  TEST_F(BasicOptimisation, levenberg_marquardt)
  {
    auto my_cost_fxn = MMORF::CostFxnSSDWarpField(
        vol_ref_mo_,
        vol_mov_mo_,
        affine_,
        knot_spacing_,
        sampling_frequency_);
    auto my_regulariser = MMORF::CostFxnBendingEnergy(
        warp_field_,
        sampling_frequency_);
    auto parameters = my_cost_fxn.get_parameters();
    auto lambda_reg = 1e8f;
    auto lambda_lm = 0.001f;
    auto solver_opts = arma::superlu_opts();
    solver_opts.symmetric = true;
    solver_opts.permutation = arma::superlu_opts::MMD_AT_PLUS_A;
    solver_opts.pivot_thresh = 0.0;
    solver_opts.refine = arma::superlu_opts::REF_SINGLE;

    // Try a few iterations
    for (auto i = 0; i < 5; ++i){
      // Calculate initial cost
      std::cout << i << std::endl;
      auto cf_cost_last = my_cost_fxn.cost();
      auto cf_cost_next = cf_cost_last;
      auto be_cost_last = my_regulariser.cost();
      auto be_cost_next = be_cost_last;
      auto total_cost_next = cf_cost_next + lambda_reg*be_cost_next;
      auto total_cost_last = cf_cost_last + lambda_reg*be_cost_last;
      // Make sure we actually decrease the cost
      do{
        // Update parameters
        my_cost_fxn.set_parameters(parameters);
        my_regulariser.set_parameters(parameters);
        // Calculate hessians and gradients
        auto cf_hess = my_cost_fxn.hess();
        auto be_hess = my_regulariser.hess();
        auto cf_grad = my_cost_fxn.grad();
        auto be_grad = my_regulariser.grad();
        // Add regularisation
        arma::sp_fmat total_hess = cf_hess + lambda_reg*be_hess;
        total_hess.save("test_BasicOptimisation_data/total_hess",arma::coord_ascii);
        total_hess.diag() *= (1.0f + lambda_lm);
        arma::fvec total_grad = cf_grad + lambda_reg*be_grad;
        // Preconditioner
        //auto preconditioner = arma::sp_fmat(arma::size(total_hess));
        //arma::fvec total_hess_diag(total_hess.diag());
        //total_hess_diag.save("test_BasicOptimisation_data/total_hess_diag",arma::raw_ascii);
        //total_hess_diag.for_each( [](arma::fvec::elem_type& val) { val = 1.0f/val; } );  // NOTE: the '&' is crucial!
        //preconditioner.diag() = total_hess_diag;
        //preconditioner.save("test_BasicOptimisation_data/preconditioner",arma::coord_ascii);
        //preconditioner.for_each( [](arma::sp_fmat::elem_type& val) { val = 1.0f/val; } );  // NOTE: the '&' is crucial!
        // Solve preconditioned system
        //arma::fvec update = arma::spsolve(total_hess*preconditioner, -total_grad, "superlu", solver_opts);
        arma::fvec update = arma::spsolve(total_hess, total_grad);//, "superlu", solver_opts);
        update.save("test_BasicOptimisation_data/update",arma::raw_ascii);
        auto updated_parameters = add_parameters(parameters,update);
        my_cost_fxn.set_parameters(updated_parameters);
        my_regulariser.set_parameters(updated_parameters);
        cf_cost_next = my_cost_fxn.cost();
        be_cost_next = my_regulariser.cost();
        total_cost_next = cf_cost_next + lambda_reg*be_cost_next;
        std::cout << "last_cost = " << total_cost_last << std::endl;
        std::cout << "next_cost = " << total_cost_next << std::endl;
        if (total_cost_next > total_cost_last){
          lambda_lm *= 10.0f;
          std::cout << "Increasing lambda_lm to " << lambda_lm << std::endl;
        }
        else{
          parameters = updated_parameters;
          lambda_lm /= 10.0f;
          std::cout << "Decreasing lambda_lm to " << lambda_lm << std::endl;
          warp_field_.set_parameters(parameters);
          auto warped_points = warp_field_.apply_warp(sample_points_);
          MMORF::save_as_nifti(
              vol_mov_mo_,
              warped_points,
              std::vector<int>{
                  vol_mov_ni_.xsize(),
                  vol_mov_ni_.ysize(),
                  vol_mov_ni_.zsize()},
              vol_mov_ni_,
              std::string("test_BasicOptimisation_data/result_iteration_" + std::to_string(i)));
        }
      }
      while (total_cost_next > total_cost_last);
    }
    // Save parameters
    //save_parameters(parameters,"lm_result");
  } // levenberg_marquardt
} // namespace
