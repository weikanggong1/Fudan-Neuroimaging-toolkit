// Unit Test File

#include "MMORFio.h"
#include "CostFxnLogJacobian.cuh"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
//#include <memory>
//#include <iostream>
//#include <algorithm>
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

  class CFBE_consistancy : public ::testing::Test
  {
    public:
      CFBE_consistancy()
        : filename_ref_("./data/MNI152_T1_2mm_smooth_10mmFWHM.nii.gz")
        , filename_warp_params_("./data/actual_warp_params")
        , vol_ref_mo_(filename_ref_)
        , affine_{
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0},
            {0,0,0,1}}
        , warp_field_(std::make_shared<MMORF::WarpFieldBSpline>(vol_ref_mo_.get_extents(),knot_spacing_))
      {
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
        srand(24);
        //auto params = arma::fmat();
        //params.load(filename_warp_params_);
        //for (auto col = 0; col < 3; ++col){
        //  actual_warp_params_.push_back(arma::conv_to<std::vector<float> >::from(params.col(col)));
        //}

        actual_warp_params_ = warp_field_->get_parameters();
        for (auto& param_vec : actual_warp_params_){
          for(auto& param : param_vec){
            param = 40.0f*(static_cast<float>(rand())/static_cast<float>(RAND_MAX) - 0.5);
          }
        }

      }

    protected:
      const float knot_spacing_ = 20.0f;
      const int sampling_frequency_ = 10;
      const std::string filename_ref_;
      const std::string filename_warp_params_;
      MMORF::VolumeBSpline vol_ref_mo_;
      arma::Mat<float> affine_;
      std::shared_ptr<MMORF::WarpFieldBSpline> warp_field_;
      NEWIMAGE::volume<float> vol_ref_ni_;
      std::vector<std::vector<float> > sample_points_;
      std::vector<std::vector<float> > actual_warp_params_;
      const int sample_start_x = 90;
      const int sample_end_x = -90;
      const int sample_start_y = -126;
      const int sample_end_y = 90;
      const int sample_start_z = -72;
      const int sample_end_z = 108;
      const float lambda_reg_ = 1e8f;

  }; // CFBE_consistancy
  TEST_F(CFBE_consistancy, func_cost_consistancy)
  {
    warp_field_->set_parameters(actual_warp_params_);
    auto extents_original = warp_field_->get_extents();
    std::cout << "Original_extents: ";
    for (const auto& point : extents_original.first){
      std::cout << point << " ";
    }
    std::cout << "   :   ";
    for (const auto& point : extents_original.second){
      std::cout << point << " ";
    }
    std::cout << std::endl;
    for (auto i = 2; i <= 5 ; ++i){
      auto my_regulariser = MMORF::CostFxnLogJacobian(
          warp_field_,
          i);
      std::cout << my_regulariser.cost() << std::endl;
    }
    std::cout << "REPARAMETERISE\n";
    *warp_field_ = warp_field_->reparameterise(2);
    auto extents_reparam = warp_field_->get_extents();
    std::cout << "Reparameterised_extents: ";
    for (const auto& point : extents_reparam.first){
      std::cout << point << " ";
    }
    std::cout << "   :   ";
    for (const auto& point : extents_reparam.second){
      std::cout << point << " ";
    }
    std::cout << std::endl;
    for (auto i = 2; i <= 5 ; ++i){
      auto my_regulariser = MMORF::CostFxnLogJacobian(
          warp_field_,
          i);
      std::cout << my_regulariser.cost() << std::endl;
    }
    std::cout << "CROP\n";
    *warp_field_ = warp_field_->crop(vol_ref_mo_.get_extents());
    auto extents_crop = warp_field_->get_extents();
    std::cout << "Cropped_extents: ";
    for (const auto& point : extents_crop.first){
      std::cout << point << " ";
    }
    std::cout << "   :   ";
    for (const auto& point : extents_crop.second){
      std::cout << point << " ";
    }
    std::cout << std::endl;
    for (auto i = 2; i <= 5 ; ++i){
      auto my_regulariser = MMORF::CostFxnLogJacobian(
          warp_field_,
          i);
      std::cout << my_regulariser.cost() << std::endl;
    }
  } // func_cost_consistancy

} // namespace
