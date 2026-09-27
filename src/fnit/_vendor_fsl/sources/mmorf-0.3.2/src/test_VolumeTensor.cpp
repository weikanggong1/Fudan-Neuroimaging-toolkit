// Testing VolumeTensor class functionality

#include "VolumeTensor.cuh"
#include "MMORFio.h"

#include "newimage/newimageall.h"

#include "gtest/gtest.h"

#include <armadillo>

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <utility>

namespace
{
  class VolumeTensorTest : public ::testing::Test
  {
    public:
      VolumeTensorTest()
        : filename_("./data/HCP/100307_diffusion/T1w/Diffusion/dti_tensor.nii.gz")
      {
        NEWIMAGE::read_volume4D(test_vol_,filename_);
        test_vol_.setextrapolationmethod(NEWIMAGE::mirror);
        std::vector<float> x_points;
        std::vector<float> y_points;
        std::vector<float> z_points;
        for (auto z = sample_start_z; z <= sample_end_z; ++z, ++z){
          for (auto y = sample_start_y; y <= sample_end_y; ++y, ++y){
            for (auto x = sample_start_x; x >= sample_end_x; --x, --x){
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
    protected:
      const std::string filename_;
      NEWIMAGE::volume4D<float> test_vol_;
      std::vector<std::vector<float> > sample_points_;
      const int sample_start_x = 90;
      const int sample_end_x = -90;
      const int sample_start_y = -126;
      const int sample_end_y = 90;
      const int sample_start_z = -72;
      const int sample_end_z = 108;
  };

  TEST_F(VolumeTensorTest, ctor_newimage_volume_func_sample)
  {
    MMORF::VolumeTensor my_vol(test_vol_);
    auto samples = my_vol.sample(sample_points_);
    EXPECT_EQ(samples.size(), 9);
    EXPECT_EQ(samples[0].size(), sample_points_[0].size());
  }

  TEST_F(VolumeTensorTest, ctor_string_func_sample)
  {
    MMORF::VolumeTensor my_vol(filename_);
    auto samples = my_vol.sample(sample_points_);
    EXPECT_EQ(samples.size(), 9);
    EXPECT_EQ(samples[0].size(), sample_points_[0].size());
  }

  TEST_F(VolumeTensorTest, ctor_newimage_volume_smoothed_func_sample)
  {
    MMORF::VolumeTensor my_vol(test_vol_, 5.0f);
    auto samples = my_vol.sample(sample_points_);
    EXPECT_EQ(samples.size(), 9);
    EXPECT_EQ(samples[0].size(), sample_points_[0].size());
  }

  TEST_F(VolumeTensorTest, ctor_string_smoothed_func_sample)
  {
    MMORF::VolumeTensor my_vol(filename_, 5.0f);
    auto samples = my_vol.sample(sample_points_);
    EXPECT_EQ(samples.size(), 9);
    EXPECT_EQ(samples[0].size(), sample_points_[0].size());
  }

  TEST_F(VolumeTensorTest, ctor_string_func_sample_derivative)
  {
    MMORF::VolumeTensor my_vol(filename_);
    for (auto i = 0; i < 3; ++i){
      auto sampled_vol = my_vol.sample_derivative(sample_points_,i);
    }
  }

  TEST_F(VolumeTensorTest, ctor_string_smoothed_func_sample_derivative)
  {
    MMORF::VolumeTensor my_vol(filename_, 5.0f);
    for (auto i = 0; i < 3; ++i){
      auto sampled_vol = my_vol.sample_derivative(sample_points_,i);
    }
  }
/*
  TEST_F(VolumeBSplineTest, multiple_vols)
  {
    MMORF::VolumeBSpline my_vol_1(filename_);
    MMORF::VolumeBSpline my_vol_2(filename_);
    auto sampled_vol_1 = my_vol_1.sample(sample_points_);
    auto sampled_vol_2 = my_vol_2.sample(sample_points_);
    sampled_vol_1 = my_vol_1.sample(sample_points_);
    sampled_vol_2 = my_vol_2.sample(sample_points_);
    EXPECT_EQ(sampled_vol_1,sampled_vol_2);
  }

  TEST_F(VolumeBSplineTest, ctor_string_func_set_coefficients)
  {
    auto my_vol = MMORF::VolumeBSpline(filename_);
    auto dims = my_vol.get_dimensions();
    auto sampled_vol_1 = my_vol.sample(sample_points_);
    auto coef_sz = 1;
    for (const auto& dim : dims){
      coef_sz *= dim;
    }
    auto coeffs_new = std::vector<float>(coef_sz, 1.0);
    my_vol.set_coefficients(coeffs_new);
    auto sampled_vol_2 = my_vol.sample(sample_points_);
    EXPECT_NE(sampled_vol_1,sampled_vol_2);
  }

  TEST_F(VolumeBSplineTest, ctor_string_func_get_extents)
  {
    auto my_vol = MMORF::VolumeBSpline(filename_);
    auto extents = my_vol.get_extents();
    EXPECT_EQ(extents.first,(std::vector<float>{-90,-126,-72}));
    EXPECT_EQ(extents.second,(std::vector<float>{90,90,108}));
  }

  TEST_F(VolumeBSplineTest, ctor_vecs_func_get_extents)
  {
    auto dims_in = std::vector<int>{5,5,5};
    auto res_in = std::vector<float>{10,10,10};
    auto origin_in = std::vector<float>{2,2,2};
    auto my_vol = MMORF::VolumeBSpline(dims_in,res_in,origin_in);
    auto extents = my_vol.get_extents();
    for (auto i = 0; i< 3; ++i)
    {
      EXPECT_FLOAT_EQ(extents.first[i],-20.0f);
      EXPECT_FLOAT_EQ(extents.second[i],20.0f);
    }
  }

  TEST_F(VolumeBSplineTest, ctor_string_func_reparameterise)
  {
    MMORF::VolumeBSpline my_vol(filename_);
    auto reparameterised_vol = my_vol.reparameterise(2);
    auto sampled_vol_1 = my_vol.sample(sample_points_);
    auto sampled_vol_2 = reparameterised_vol.sample(sample_points_);
    MMORF::save_as_nifti(
        my_vol,
        sample_points_,
        my_vol.get_dimensions(),
        test_vol_,
        "data/pre_parameterised_vol");
    MMORF::save_as_nifti(
        reparameterised_vol,
        sample_points_,
        my_vol.get_dimensions(),
        test_vol_,
        "data/reparameterised_vol");
    for (auto i = 0; i < sampled_vol_1.size(); ++i){
      EXPECT_NEAR(sampled_vol_1[i],sampled_vol_2[i], 5);
    }
  }

  TEST_F(VolumeBSplineTest, ctor_vecs_func_get_resolution)
  {
    auto dims_in = std::vector<int>{
      test_vol_.xsize(),
      test_vol_.ysize(),
      test_vol_.zsize()};
    auto res_in = std::vector<float>{
      test_vol_.xdim(),
      test_vol_.ydim(),
      test_vol_.zdim()};
    auto origin_in = std::vector<float>{0, 0, 0};
    auto my_vol = MMORF::VolumeBSpline(dims_in,res_in,origin_in);
    auto res_out = my_vol.get_resolution();
    EXPECT_EQ(test_vol_.xdim(),res_out.at(0));
    EXPECT_EQ(test_vol_.ydim(),res_out.at(1));
    EXPECT_EQ(test_vol_.zdim(),res_out.at(2));
  }

  TEST_F(VolumeBSplineTest, ctor_string_func_get_original_sample_positions)
  {
    auto my_vol = MMORF::VolumeBSpline(filename_);
    auto test_points = my_vol.get_original_sample_positions();
    auto true_samples = my_vol.sample(sample_points_);
    auto test_samples = my_vol.sample(test_points);
    EXPECT_EQ(test_samples, true_samples);
  }

  TEST_F(VolumeBSplineTest, ctor_string_func_crop)
  {
    auto my_vol = MMORF::VolumeBSpline(filename_);
    auto old_dimensions = my_vol.get_dimensions();
    auto extents_big = std::make_pair(
        std::vector<float>({-300.0, -300.0, -300.0}),
        std::vector<float>({300.0, 300.0, 300.0}));
    auto cropped_vol = my_vol.crop(extents_big);
    auto new_dimensions = cropped_vol.get_dimensions();
    EXPECT_EQ(old_dimensions, new_dimensions);

    auto extents_exact = std::make_pair(
        std::vector<float>({-90.0, -126.0, -72.0}),
        std::vector<float>({90.0, 90.0, 108.0}));
    cropped_vol = my_vol.crop(extents_exact);
    new_dimensions = cropped_vol.get_dimensions();
    EXPECT_EQ(old_dimensions, new_dimensions);

    auto extents_less = std::make_pair(
        std::vector<float>({0.0, -16.01, 18.00}),
        std::vector<float>({90.0, 90.0, 108.00}));
    my_vol.crop(extents_less);
    auto expected_dimensions = old_dimensions;
    for (auto& dimension : expected_dimensions) dimension = dimension/2 + 2;
    cropped_vol = my_vol.crop(extents_less);
    new_dimensions = cropped_vol.get_dimensions();
    EXPECT_EQ(expected_dimensions, new_dimensions);
  }
  */

} // namespace
