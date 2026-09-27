// Unit Test File

#include "MmorfMemory.h"
#include "PositionND.h"
#include "gtest/gtest.h"

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class PositionTest : public ::testing::Test
  {
    public:
      PositionTest()
        : vector_length_3_{0.5,1.5,2.5}
        , vector_length_4_{0.5,1.5,2.5,3.5}
        , position_length_3_(vector_length_3_)
        , position_length_4_(vector_length_4_)
      {
      }
    protected:
      std::vector<float> vector_length_3_;
      std::vector<float> vector_length_4_;
      MMORF::PositionND position_length_3_;
      MMORF::PositionND position_length_4_;
  }; // PositionTest

  TEST_F(PositionTest, func_get_dimensions)
  {
    EXPECT_EQ(position_length_3_.get_dimensions(),vector_length_3_.size());
    EXPECT_EQ(position_length_4_.get_dimensions(),vector_length_4_.size());
  } //func_get_dimensions

  TEST_F(PositionTest, func_get_positions)
  {
    EXPECT_EQ(position_length_3_.get_positions(),vector_length_3_);
    EXPECT_EQ(position_length_4_.get_positions(),vector_length_4_);
  } // func_get_positions

  TEST_F(PositionTest, op_subscript_position)
  {
    for (auto i = 0; i < vector_length_3_.size(); ++i){
      EXPECT_EQ(position_length_3_[i],vector_length_3_[i]);
      position_length_3_[i] = vector_length_3_[i] + 1.0;
      EXPECT_EQ(position_length_3_[i],vector_length_3_[i] + 1.0);
    }
    for (auto i = 0; i < vector_length_4_.size(); ++i){
      EXPECT_EQ(position_length_4_[i],vector_length_4_[i]);
      position_length_4_[i] = vector_length_4_[i] + 1.0;
      EXPECT_EQ(position_length_4_[i],vector_length_4_[i] + 1.0);
    }
  } // op_subscript_position

  TEST_F(PositionTest, func_set_positions)
  {
    auto new_vec_3 = vector_length_3_;
    auto new_vec_4 = vector_length_4_;
    for (auto& val : new_vec_3){
      val += 1.0;
    }
    for (auto& val : new_vec_4){
      val += 1.0;
    }
    position_length_3_.set_positions(new_vec_3);
    position_length_4_.set_positions(new_vec_4);
    for (auto i = 0; i < new_vec_3.size(); ++i){
      EXPECT_EQ(position_length_3_[i],new_vec_3[i]);
    }
    for (auto i = 0; i < new_vec_4.size(); ++i){
      EXPECT_EQ(position_length_4_[i],new_vec_4[i]);
    }
  } // func_set_positions
} // namespace
