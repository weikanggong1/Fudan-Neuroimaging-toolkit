// Unit Test File

#include "MMORFio.h"
#include "WarpFieldBSpline.cuh"
#include "gtest/gtest.h"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <memory>
#include <iostream>
#include <utility>

namespace
{
  class WarpFieldTest : public ::testing::Test
  {
    public:
      WarpFieldTest()
        : filename_("./data/MNI152_T1_2mm.nii.gz")
      {
        // Read in a NEWIMAGE::volume
        NEWIMAGE::read_volume(test_vol_,filename_);
        test_vol_.setextrapolationmethod(NEWIMAGE::mirror);
        // Calculate dimensions and resolution of volume
        dimensions_ = std::vector<int>{
          test_vol_.xsize(),
          test_vol_.ysize(),
          test_vol_.zsize()};
        resolution_ = std::vector<float>{
          test_vol_.xdim(),
          test_vol_.ydim(),
          test_vol_.zdim()};
        origin_ = std::vector<float>{45.0f, 63.0f, 36.0f};
        // Calculate parameter size of MMORF::VolumeBSpline
        parameter_sz_ = dimensions_[0]*dimensions_[1]*dimensions_[2];
        // Generate a set of points for sampling MMORF::VolumeBSpline
        std::vector<float> x_points;
        std::vector<float> y_points;
        std::vector<float> z_points;
        for (auto z = sample_start_z; z <= sample_start_z + 108; ++z){
          for (auto y = sample_start_y; y <= sample_start_y + 90; ++y){
            for (auto x = sample_start_x; x <= sample_start_x + 90; ++x){
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
      NEWIMAGE::volume<float> test_vol_;
      std::vector<int> dimensions_;
      std::vector<float> resolution_;
      std::vector<float> origin_;
      int parameter_sz_;
      std::vector<std::vector<float> > sample_points_;
      const int sample_start_z = -72;
      const int sample_start_y = -126;
      const int sample_start_x = -90;
  }; // WarpFieldTest

  TEST_F(WarpFieldTest, func_get_parameters)
  {
    auto my_warp = MMORF::WarpFieldBSpline(dimensions_,resolution_,origin_);
    auto parameters = my_warp.get_parameters();
    auto zeros = std::vector<std::vector<float> >(
        3, std::vector<float>(parameter_sz_, 0.0));
    EXPECT_EQ(parameters,zeros);
  } //func_get_parameters

  TEST_F(WarpFieldTest, func_set_parameters)
  {
    auto my_warp = MMORF::WarpFieldBSpline(dimensions_,resolution_,origin_);
    auto parameters = my_warp.get_parameters();
    auto zeros = std::vector<std::vector<float> >(
        3, std::vector<float>(parameter_sz_, 0.0));
    EXPECT_EQ(parameters,zeros);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(parameter_sz_, 1.0));
    my_warp.set_parameters(ones);
    parameters = my_warp.get_parameters();
    EXPECT_EQ(parameters,ones);
  } //func_set_parameters

  TEST_F(WarpFieldTest, func_apply_warp)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto warped_points = my_warp.apply_warp(sample_points_);
    EXPECT_EQ(warped_points,sample_points_);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(my_warp.get_parameters()[0].size(), 1.0));
    my_warp.set_parameters(ones);
    warped_points = my_warp.apply_warp(sample_points_);
    EXPECT_NE(warped_points,sample_points_);
  } //func_apply_warp

  TEST_F(WarpFieldTest, ctor_corners_func_get_extents)
  {
    auto bottom_corner = std::vector<float>{-20.0f,-20.0f,-20.0f};
    auto top_corner = std::vector<float>{20.0f,20.0f,20.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto new_extents = my_warp.get_extents();
    auto bottom_expected = std::vector<float>{-30.0f,-30.0f,-30.0f};
    auto top_expected = std::vector<float>{30.0f,30.0f,30.0f};

    for (auto i = 0; i < 3; ++i){
      EXPECT_FLOAT_EQ(new_extents.first[i],bottom_expected[i]);
      EXPECT_FLOAT_EQ(new_extents.second[i],top_expected[i]);
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_get_parameters)
  {
    auto bottom_corner = std::vector<float>{-10.0f,-10.0f,-10.0f};
    auto top_corner = std::vector<float>{99.0f,100.0f,101.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto parameters = my_warp.get_parameters();

    EXPECT_EQ(parameters[0].size(),14*14*15);
  }

  TEST_F(WarpFieldTest, ctor_corners_func_get_dimensions)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto dimensions = my_warp.get_dimensions();

    EXPECT_EQ(dimensions,(std::vector<int>{14,14,15}));
  }

  TEST_F(WarpFieldTest, ctor_corners_func_get_sub_warp)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto dimensions = my_warp.get_dimensions();
    auto sub_warp_0 = my_warp.get_sub_warp(0);
    auto sub_warp_1 = my_warp.get_sub_warp(1);
    auto sub_warp_2 = my_warp.get_sub_warp(2);

    EXPECT_EQ(dimensions,(sub_warp_0.get_dimensions()));
    EXPECT_EQ(dimensions,(sub_warp_1.get_dimensions()));
    EXPECT_EQ(dimensions,(sub_warp_2.get_dimensions()));
  }

  TEST_F(WarpFieldTest, ctor_corners_func_jacobian_determinant)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(my_warp.get_parameters()[0].size(), 1.0));
    auto jacobian_determinants = my_warp.jacobian_determinants(sample_points_);

    for (auto i = 0; i < jacobian_determinants.size(); ++i){
      EXPECT_FLOAT_EQ(1.0f, jacobian_determinants[i]);
    }

    my_warp.set_parameters(ones);
    jacobian_determinants = my_warp.jacobian_determinants(sample_points_);

    for (auto i = 0; i < jacobian_determinants.size(); ++i){
      EXPECT_FLOAT_EQ(1.0f, jacobian_determinants[i]);
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_get_jacobian_elements)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(my_warp.get_parameters()[0].size(), 1.0));
    auto jacobian_elements = my_warp.get_jacobian_elements(sample_points_);

    for (auto i = 0; i < jacobian_elements.size(); ++i){
      for (auto j = 0; j < jacobian_elements[i].size(); ++j){
        if (i == 0 || i == 4 || i == 8){
          EXPECT_FLOAT_EQ(1.0f, jacobian_elements[i][j]);
        }
        else{
          EXPECT_FLOAT_EQ(0.0f, jacobian_elements[i][j]);
        }
      }
    }

    my_warp.set_parameters(ones);
    jacobian_elements = my_warp.get_jacobian_elements(sample_points_);

    for (auto i = 0; i < jacobian_elements.size(); ++i){
      if (i == 0 || i == 4 || i == 8){
        auto ones_float = std::vector<float>(jacobian_elements[i].size(),1.0f);
        EXPECT_EQ(ones_float, jacobian_elements[i]);
      }
      else{
        auto zeros_float = std::vector<float>(jacobian_elements[i].size(),1.0e-6f);
        EXPECT_GT(zeros_float, jacobian_elements[i]);
      }
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_sample_warp)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(my_warp.get_parameters()[0].size(), 1.0));
    my_warp.set_parameters(ones);
    auto samples = my_warp.sample_warp(sample_points_);
    MMORF::save_as_nifti(
        samples,
        std::vector<int>{
            test_vol_.xsize(),
            test_vol_.ysize(),
            test_vol_.zsize()},
        test_vol_,
        std::string("data_test_WarpFieldBSpline/sampled_warp"));
  }

  TEST_F(WarpFieldTest, ctor_corners_func_reparameterise_get_extents_NIREP)
  {
    auto bottom_corner = std::vector<float>{0.0f,0.0f,0.0f};
    auto top_corner = std::vector<float>{212.0f,272.0f,200.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 40.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    my_warp = my_warp.reparameterise(2);
    auto new_extents = my_warp.get_extents();
    auto bottom_expected = std::vector<float>{-60.0f,-60.0f,-60.0f};
    auto top_expected = std::vector<float>{300.0f,340.0f,260.0f};

    for (auto i = 0; i < 3; ++i){
      EXPECT_FLOAT_EQ(new_extents.first[i],bottom_expected[i]);
      EXPECT_FLOAT_EQ(new_extents.second[i],top_expected[i]);
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_reparameterise_get_extents)
  {
    auto bottom_corner = std::vector<float>{-20.0f,-20.0f,-20.0f};
    auto top_corner = std::vector<float>{20.0f,20.0f,20.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    my_warp = my_warp.reparameterise(2);
    my_warp = my_warp.reparameterise(2);
    auto new_extents = my_warp.get_extents();
    auto bottom_expected = std::vector<float>{-37.5f,-37.5f,-37.5f};
    auto top_expected = std::vector<float>{37.5f,37.5f,37.5f};

    for (auto i = 0; i < 3; ++i){
      EXPECT_FLOAT_EQ(new_extents.first[i],bottom_expected[i]);
      EXPECT_FLOAT_EQ(new_extents.second[i],top_expected[i]);
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_reparameterise_jacobian_determinant)
  {
    auto bottom_corner = std::vector<float>{-10,-10,-10};
    auto top_corner = std::vector<float>{99,100,101};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    my_warp = my_warp.reparameterise(2);
    auto ones = std::vector<std::vector<float> >(
        3, std::vector<float>(my_warp.get_parameters()[0].size(), 1.0));
    auto jacobian_determinants = my_warp.jacobian_determinants(sample_points_);

    for (auto i = 0; i < jacobian_determinants.size(); ++i){
      EXPECT_FLOAT_EQ(1.0f, jacobian_determinants[i]);
    }

    my_warp.set_parameters(ones);
    jacobian_determinants = my_warp.jacobian_determinants(sample_points_);

    for (auto i = 0; i < jacobian_determinants.size(); ++i){
      EXPECT_FLOAT_EQ(1.0f, jacobian_determinants[i]);
    }
  }

  TEST_F(WarpFieldTest, func_get_robust_sample_positions)
  {
    auto bottom_corner = std::vector<float>{-25.0f,-24.0f,-23.0f};
    auto top_corner = std::vector<float>{25.0f,24.0f,23.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 10.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    for (auto i = 1; i <=4; i*=2){
      auto positions = my_warp.get_robust_sample_positions(i);
      auto dimensions = my_warp.get_robust_sample_dimensions(i);
      EXPECT_EQ(positions[0].size(), dimensions[0]*dimensions[1]*dimensions[2]);
    }
  } //func_get_robust_sample_positions

  TEST_F(WarpFieldTest, ctor_corners_func_crop_NIREP)
  {
    auto bottom_corner = std::vector<float>{0.0f,0.0f,0.0f};
    auto top_corner = std::vector<float>{212.0f,272.0f,200.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 40.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto old_extents = my_warp.get_extents();
    my_warp = my_warp.reparameterise(2);
    my_warp = my_warp.crop(extents);
    auto new_extents = my_warp.get_extents();
    auto bottom_expected = std::vector<float>{-20.0f,-20.0f,-20.0f};
    auto top_expected = std::vector<float>{240.0f,300.0f,220.0f};

    for (auto i = 0; i < 3; ++i){
      EXPECT_FLOAT_EQ(new_extents.first[i],bottom_expected[i]);
      EXPECT_FLOAT_EQ(new_extents.second[i],top_expected[i]);
    }
  }

  TEST_F(WarpFieldTest, ctor_corners_func_crop)
  {
    auto bottom_corner = std::vector<float>{-100.0f,-100.0f,-100.0f};
    auto top_corner = std::vector<float>{100.0f,100.0f,100.0f};
    auto extents = std::make_pair(bottom_corner,top_corner);
    auto knot_spacing = 40.0f;
    auto my_warp = MMORF::WarpFieldBSpline(extents,knot_spacing);
    auto old_extents = my_warp.get_extents();
    my_warp = my_warp.reparameterise(2);
    my_warp = my_warp.crop(extents);
    auto new_extents = my_warp.get_extents();
    for (auto i = 0; i < 3; ++i){
      EXPECT_FLOAT_EQ(new_extents.first[i],old_extents.first[i] + knot_spacing/2);
      EXPECT_FLOAT_EQ(new_extents.second[i],old_extents.second[i] - knot_spacing/2);
    }
  }
} // namespace
