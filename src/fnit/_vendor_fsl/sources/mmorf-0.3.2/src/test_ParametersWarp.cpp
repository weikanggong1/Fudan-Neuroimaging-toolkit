// First Unit Test File

#include "MmorfMemory.h"
#include "Parameters.h"
#include "ParametersWarp.h"
#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include <memory>

namespace
{
  class ParametersWarpTest : public ::testing::Test
  {
    public:
      ParametersWarpTest()
        : i_(3)
        , j_(4)
        , k_(5)
        , dims_{i_,j_,k_}
      {
        for (auto n = 0; n < i_*j_*k_; n++){
          vals_.push_back(static_cast<double>(n));
        }
      }
    protected:
      const int i_;
      const int j_;
      const int k_;
      const std::vector<int> dims_;
      std::vector<double> vals_;
  };

  TEST_F(ParametersWarpTest, ctor_dims_func_dimensions)
  {
    MMORF::ParametersWarp my_params(dims_);
    auto dims_out = my_params.dimensions();
    EXPECT_EQ(i_, dims_out[0]);
    EXPECT_EQ(j_, dims_out[1]);
    EXPECT_EQ(k_, dims_out[2]);
  }

  TEST_F(ParametersWarpTest, ctor_dims_subscript_index)
  {
    MMORF::ParametersWarp my_params(dims_);
    for (auto i = 0; i < i_; ++i){
      for (auto j = 0; j < j_; ++j){
        for (auto k = 0; k < k_; ++k){
          std::vector<int> index{i,j,k};
          EXPECT_EQ(0, my_params[index]);
        }
      }
    }
  }

  TEST_F(ParametersWarpTest, ctor_vals_func_dimensions)
  {
    MMORF::ParametersWarp my_params(dims_,vals_);
    auto dims_out = my_params.dimensions();
    EXPECT_EQ(i_, dims_out[0]);
    EXPECT_EQ(j_, dims_out[1]);
    EXPECT_EQ(k_, dims_out[2]);
  }

  TEST_F(ParametersWarpTest, ctor_vals_subscript_index)
  {
    MMORF::ParametersWarp my_params(dims_,vals_);
    for (auto i = 0; i < i_; ++i){
      for (auto j = 0; j < j_; ++j){
        for (auto k = 0; k < k_; ++k){
          std::vector<int> index{i,j,k};
          auto value_out = static_cast<double>(i + j*i_ + k*j_);
          EXPECT_EQ(value_out, my_params[index]);
        }
      }
    }
  }

  TEST_F(ParametersWarpTest, ctor_copy)
  {
    MMORF::ParametersWarp my_params(dims_,vals_);
    MMORF::ParametersWarp my_params_2(my_params);
    for (auto i = 0; i < i_; ++i){
      for (auto j = 0; j < j_; ++j){
        for (auto k = 0; k < k_; ++k){
          std::vector<int> index{i,j,k};
          EXPECT_EQ(my_params[index], my_params_2[index]);
        }
      }
    }
  }

  TEST_F(ParametersWarpTest, op_copy_assign)
  {
    MMORF::ParametersWarp my_params(dims_,vals_);
    MMORF::ParametersWarp my_params_2 = my_params;
    for (auto i = 0; i < i_; ++i){
      for (auto j = 0; j < j_; ++j){
        for (auto k = 0; k < k_; ++k){
          std::vector<int> index{i,j,k};
          EXPECT_EQ(my_params[index], my_params_2[index]);
        }
      }
    }
  }

  TEST_F(ParametersWarpTest, polymorphism)
  {
    auto warp_ptr = static_cast<std::unique_ptr<MMORF::Parameters>>(
                      MMORF::make_unique<MMORF::ParametersWarp>(dims_,vals_));
    for (auto i = 0; i < i_; ++i){
      for (auto j = 0; j < j_; ++j){
        for (auto k = 0; k < k_; ++k){
          std::vector<int> index{i,j,k};
          auto value_out = static_cast<double>(i + j*i_ + k*j_);
          EXPECT_EQ(value_out, (*warp_ptr)[index]);
        }
      }
    }
  }

} // namespace
