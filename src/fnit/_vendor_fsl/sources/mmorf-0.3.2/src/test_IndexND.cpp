// Unit Test File

#include "MmorfMemory.h"
#include "IndexND.h"
#include "gtest/gtest.h"

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class IndexTest : public ::testing::Test
  {
    public:
      IndexTest()
        : vector_length_3_{0,1,2}
        , vector_length_4_{0,1,2,3}
        , index_length_3_(vector_length_3_)
        , index_length_4_(vector_length_4_)
      {
      }
    protected:
      std::vector<int> vector_length_3_;
      std::vector<int> vector_length_4_;
      MMORF::IndexND index_length_3_;
      MMORF::IndexND index_length_4_;
  }; // IndexTest

  TEST_F(IndexTest, func_get_dimensions)
  {
    EXPECT_EQ(index_length_3_.get_dimensions(),vector_length_3_.size());
    EXPECT_EQ(index_length_4_.get_dimensions(),vector_length_4_.size());
  } //func_get_dimensions

  TEST_F(IndexTest, func_get_indices)
  {
    EXPECT_EQ(index_length_3_.get_indices(),vector_length_3_);
    EXPECT_EQ(index_length_4_.get_indices(),vector_length_4_);
  } // func_get_indices

  TEST_F(IndexTest, op_subscript_index)
  {
    for (auto i = 0; i < vector_length_3_.size(); ++i){
      EXPECT_EQ(index_length_3_[i],vector_length_3_[i]);
      index_length_3_[i] = vector_length_3_[i] + 1;
      EXPECT_EQ(index_length_3_[i],vector_length_3_[i] + 1);
    }
    for (auto i = 0; i < vector_length_4_.size(); ++i){
      EXPECT_EQ(index_length_4_[i],vector_length_4_[i]);
      index_length_4_[i] = vector_length_4_[i] + 1;
      EXPECT_EQ(index_length_4_[i],vector_length_4_[i] + 1);
    }
  } // op_subscript_index

  TEST_F(IndexTest, func_set_indices)
  {
    auto new_vec_3 = vector_length_3_;
    auto new_vec_4 = vector_length_4_;
    for (auto& val : new_vec_3){
      val += 1;
    }
    for (auto& val : new_vec_4){
      val += 1;
    }
    index_length_3_.set_indices(new_vec_3);
    index_length_4_.set_indices(new_vec_4);
    for (auto i = 0; i < new_vec_3.size(); ++i){
      EXPECT_EQ(index_length_3_[i],new_vec_3[i]);
    }
    for (auto i = 0; i < new_vec_4.size(); ++i){
      EXPECT_EQ(index_length_4_[i],new_vec_4[i]);
    }
  } // func_set_indices
} // namespace
