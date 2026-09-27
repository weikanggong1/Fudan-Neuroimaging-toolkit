// First Unit Test File

#include "MmorfMemory.h"
#include "Vector.h"
#include "VectorArmadillo.h"
#include "gtest/gtest.h"
#include "armadillo"

#include <vector>
#include <memory>
#include <iostream>

namespace
{
  class VectorArmadilloTest : public ::testing::Test
  {
    public:
      VectorArmadilloTest()
        : size_(5)
        , vals_{0.0,1.0,2.0,3.0,4.0}
      {}
    protected:
      const int size_;
      const std::vector<double> vals_;
  };

  TEST_F(VectorArmadilloTest, ctor_size_func_size)
  {
    MMORF::VectorArmadillo my_vec(size_);
    auto size_out = my_vec.size();
    EXPECT_EQ(size_, size_out);
  }

  TEST_F(VectorArmadilloTest, ctor_size_subscript_index)
  {
    MMORF::VectorArmadillo my_vec(size_);
    for (auto i = 0; i < size_; ++i){
      EXPECT_EQ(0, my_vec[i]);
    }
  }

  TEST_F(VectorArmadilloTest, ctor_vals_func_size)
  {
    MMORF::VectorArmadillo my_vec(vals_);
    auto size_out = my_vec.size();
    EXPECT_EQ(vals_.size(), size_out);
  }

  TEST_F(VectorArmadilloTest, ctor_vals_subscript_index)
  {
    MMORF::VectorArmadillo my_vec(vals_);
    for (std::size_t i = 0; i < vals_.size(); ++i){
      auto val_out = vals_.at(i);
      EXPECT_EQ(val_out, my_vec[i]);
    }
  }

  TEST_F(VectorArmadilloTest, ctor_copy)
  {
    MMORF::VectorArmadillo my_vec(vals_);
    MMORF::VectorArmadillo my_vec_2(my_vec);
    for (auto i = 0; i < my_vec.size(); ++i){
      EXPECT_EQ(my_vec[i], my_vec_2[i]);
    }
  }

  TEST_F(VectorArmadilloTest, op_copy_assign)
  {
    MMORF::VectorArmadillo my_vec(vals_);
    MMORF::VectorArmadillo my_vec_2 = my_vec;
    for (auto i = 0; i < my_vec.size(); ++i){
      EXPECT_EQ(my_vec[i], my_vec_2[i]);
    }
  }

  TEST_F(VectorArmadilloTest, polymorphism)
  {
    auto vector_ptr = static_cast<std::unique_ptr<MMORF::Vector>>(
                      MMORF::make_unique<MMORF::VectorArmadillo>(vals_));
    for (auto i = 0; i < vector_ptr->size(); ++i){
      auto val_out = vals_.at(i);
      EXPECT_EQ(val_out, (*vector_ptr)[i]);
    }
  }

} // namespace
