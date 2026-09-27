
// First Unit Test File

#include "MmorfMemory.h"
#include "EddySplineField.cuh"
#include "gtest/gtest.h"

#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <string>

namespace
{

  // Read in a file containing a JtJ matrix in CSR format, i.e. assume 3 columns where:
  // Column1: Column index
  // Column2: Row index
  // Column3: Value
  std::vector<std::vector<float> > read_JtJ(const std::string& filename)
  {
    auto r_vec = std::vector<std::vector<float> >(3);
    // Open file
    auto file = std::ifstream(filename);
    auto line = std::string();
    // Read file
    if (file){
      while (getline(file, line)){
        auto c = 0.0f;
        auto r = 0.0f;
        auto v = 0.0f;
        auto ss = std::stringstream(line);
        ss >> c >> r >> v;
        r_vec.at(0).push_back(c);
        r_vec.at(1).push_back(r);
        r_vec.at(2).push_back(v);
      }
    }
    return r_vec;
  }

  std::vector<std::vector<float> > difference_JtJ(
      const std::vector<std::vector<float> >& JtJ1,
      const std::vector<std::vector<float> >& JtJ2)
  {
    auto r_vec = std::vector<std::vector<float> >(3, std::vector<float>(JtJ1.at(0).size()));
    for (auto i = 0; i < 2; ++i){
      std::transform(JtJ1.at(i).begin(), JtJ1.at(i).end(), JtJ2.at(i).begin(),
          r_vec.at(i).begin(), std::minus<int>());
    }
    return r_vec;
  }

  class EddySplineFieldTest : public ::testing::Test
  {
    public:
      EddySplineFieldTest()
        : ksp_(4)
        , filename_("./data/test_volume.nii.gz")
        , image_size_(3)
        , voxel_size_(3)
        , knot_spacing_(3)
        , deriv_x_ {0,0,1}
        , deriv_y_ {0,1,0}
        , deriv_z_ {1,0,0}
      {
        NEWIMAGE::read_volume(test_vol_,filename_);
        test_vol_.setextrapolationmethod(NEWIMAGE::mirror);
        image_size_.at(0) = test_vol_.xsize();
        image_size_.at(1) = test_vol_.ysize();
        image_size_.at(2) = test_vol_.zsize();
        voxel_size_.at(0) = test_vol_.xdim();
        voxel_size_.at(1) = test_vol_.ydim();
        voxel_size_.at(2) = test_vol_.zdim();
        knot_spacing_.at(0) = static_cast<unsigned int>((ksp_ / voxel_size_.at(0)) + 0.5);
        knot_spacing_.at(1) = static_cast<unsigned int>((ksp_ / voxel_size_.at(0)) + 0.5);
        knot_spacing_.at(2) = static_cast<unsigned int>((ksp_ / voxel_size_.at(0)) + 0.5);
        bf_spline_field_ptr_ = MMORF::make_unique<BASISFIELD::splinefield>(
            image_size_, voxel_size_, knot_spacing_);
        mm_spline_field_ptr_ = MMORF::make_unique<MMORF::EddySplineField>(
            image_size_, voxel_size_, knot_spacing_);
      }
    protected:
      const unsigned int ksp_;
      const std::string filename_;
      NEWIMAGE::volume<float> test_vol_;
      std::vector<unsigned int> image_size_;
      std::vector<double> voxel_size_;
      std::vector<unsigned int> knot_spacing_;
      std::vector<unsigned int> deriv_x_;
      std::vector<unsigned int> deriv_y_;
      std::vector<unsigned int> deriv_z_;
      std::unique_ptr<BASISFIELD::splinefield> bf_spline_field_ptr_;
      std::unique_ptr<MMORF::EddySplineField> mm_spline_field_ptr_;
  };

  TEST_F(EddySplineFieldTest, func_CoefSz_)
  {
    EXPECT_EQ(bf_spline_field_ptr_->CoefSz_x(),mm_spline_field_ptr_->CoefSz_x());
    EXPECT_EQ(bf_spline_field_ptr_->CoefSz_y(),mm_spline_field_ptr_->CoefSz_y());
    EXPECT_EQ(bf_spline_field_ptr_->CoefSz_z(),mm_spline_field_ptr_->CoefSz_z());
  }

  TEST_F(EddySplineFieldTest, func_AsVolume)
  {
    auto bf_spline_vol = test_vol_;
    auto mm_spline_vol = test_vol_;
    bf_spline_field_ptr_->AsVolume(bf_spline_vol);
    mm_spline_field_ptr_->AsVolume(mm_spline_vol);
    EXPECT_EQ(bf_spline_vol,mm_spline_vol);
  }

  TEST_F(EddySplineFieldTest, func_SetCoef)
  {
    auto my_coefs = NEWMAT::ColumnVector(mm_spline_field_ptr_->CoefSz_x() *
        mm_spline_field_ptr_->CoefSz_y() *
        mm_spline_field_ptr_->CoefSz_z());
    my_coefs = 2.0;

    auto bf_spline_vol = test_vol_;
    auto mm_spline_vol = test_vol_;
    bf_spline_field_ptr_->AsVolume(bf_spline_vol);
    mm_spline_field_ptr_->AsVolume(mm_spline_vol);
    EXPECT_EQ(bf_spline_vol,mm_spline_vol);

    bf_spline_field_ptr_->SetCoef(my_coefs);
    bf_spline_field_ptr_->AsVolume(bf_spline_vol);
    mm_spline_field_ptr_->AsVolume(mm_spline_vol);
    EXPECT_NE(bf_spline_vol,mm_spline_vol);

    mm_spline_field_ptr_->SetCoef(my_coefs);
    bf_spline_field_ptr_->AsVolume(bf_spline_vol);
    mm_spline_field_ptr_->AsVolume(mm_spline_vol);
    EXPECT_EQ(bf_spline_vol,mm_spline_vol);
  }

  TEST_F(EddySplineFieldTest, func_BendEnergy)
  {
    auto my_coefs = NEWMAT::ColumnVector(mm_spline_field_ptr_->CoefSz_x() *
        mm_spline_field_ptr_->CoefSz_y() *
        mm_spline_field_ptr_->CoefSz_z());
    my_coefs = 2.0;
    EXPECT_EQ(bf_spline_field_ptr_->BendEnergy(),mm_spline_field_ptr_->BendEnergy());

    bf_spline_field_ptr_->SetCoef(my_coefs);
    EXPECT_NE(bf_spline_field_ptr_->BendEnergy(),mm_spline_field_ptr_->BendEnergy());

    mm_spline_field_ptr_->SetCoef(my_coefs);
    EXPECT_EQ(bf_spline_field_ptr_->BendEnergy(),mm_spline_field_ptr_->BendEnergy());
  }

  TEST_F(EddySplineFieldTest, func_BendEnergyGrad)
  {
    auto my_coefs = NEWMAT::ColumnVector(mm_spline_field_ptr_->CoefSz_x() *
        mm_spline_field_ptr_->CoefSz_y() *
        mm_spline_field_ptr_->CoefSz_z());
    my_coefs = 2.0;

    NEWMAT::ColumnVector bf_be_grad = bf_spline_field_ptr_->BendEnergyGrad();
    NEWMAT::ColumnVector mm_be_grad = mm_spline_field_ptr_->BendEnergyGrad();
    EXPECT_EQ(bf_be_grad,mm_be_grad);

    bf_spline_field_ptr_->SetCoef(my_coefs);
    bf_be_grad = bf_spline_field_ptr_->BendEnergyGrad();
    mm_be_grad = mm_spline_field_ptr_->BendEnergyGrad();
    EXPECT_NE(bf_be_grad,mm_be_grad);

    mm_spline_field_ptr_->SetCoef(my_coefs);
    bf_be_grad = bf_spline_field_ptr_->BendEnergyGrad();
    mm_be_grad = mm_spline_field_ptr_->BendEnergyGrad();
    EXPECT_EQ(bf_be_grad,mm_be_grad);
  }
/*
  TEST_F(EddySplineFieldTest, func_BendEnergyHess)
  {
    // Random coefficient generation
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);
    auto coef_sz = mm_spline_field_ptr_->CoefSz_x() *
        mm_spline_field_ptr_->CoefSz_y() *
        mm_spline_field_ptr_->CoefSz_z();
    auto my_coefs = NEWMAT::ColumnVector(coef_sz);
    for (auto i = 1; i <= coef_sz; ++i){
      my_coefs(i) = distribution(generator);
      if (i/100 == 0){
      }
    }

    auto bf_be_hess = bf_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    auto mm_be_hess = mm_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    EXPECT_EQ((*bf_be_hess)(100,102),(*mm_be_hess)(100,102));

    bf_spline_field_ptr_->SetCoef(my_coefs);
    bf_be_hess = bf_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    mm_be_hess = mm_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    EXPECT_EQ((*bf_be_hess)(100,102),(*mm_be_hess)(100,102));

    mm_spline_field_ptr_->SetCoef(my_coefs);
    bf_be_hess = bf_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    mm_be_hess = mm_spline_field_ptr_->BendEnergyHess(MISCMATHS::BFMatrixFloatPrecision);
    EXPECT_EQ((*bf_be_hess)(100,102),(*mm_be_hess)(100,102));

  }
*/

  TEST_F(EddySplineFieldTest, func_JtJ_symetrical)
  {
    // Setup parameters
    auto bf_JtJ_filename = std::string("./data/bf_JtJ_symmetrical");
    auto mm_JtJ_filename = std::string("./data/mm_JtJ_symmetrical");
    auto bf_JtJ_asymm_filename = std::string("./data/bf_JtJ_non_symmetrical");
    auto mm_JtJ_asymm_filename = std::string("./data/mm_JtJ_non_symmetrical");
    auto deriv = std::vector<unsigned int>(3,0);
    auto ima1 = test_vol_;
    auto ima2 = test_vol_;
    auto mask = NEWIMAGE::volume<char>(test_vol_.xsize(),test_vol_.ysize(),test_vol_.zsize());
    auto prec = MISCMATHS::BFMatrixFloatPrecision;

    mask = 0x01;
    // No differentiation
    //auto bf_JtJ = bf_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    auto mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //auto bf_JtJ_asymm = bf_spline_field_ptr_->JtJ(deriv,ima1,deriv,ima2,&mask,prec);
    auto mm_JtJ_asymm = mm_spline_field_ptr_->JtJ(deriv,ima1,deriv,ima2,&mask,prec);
    mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    mm_JtJ_asymm = mm_spline_field_ptr_->JtJ(deriv,ima1,deriv,ima2,&mask,prec);
    mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    mm_JtJ_asymm = mm_spline_field_ptr_->JtJ(deriv,ima1,deriv,ima2,&mask,prec);
    //mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //mm_JtJ_asymm = mm_spline_field_ptr_->JtJ(deriv,ima1,deriv,ima2,&mask,prec);
    //bf_JtJ->Print(bf_JtJ_filename);
    //mm_JtJ->Print(mm_JtJ_filename);
    //mm_JtJ_asymm->Print(mm_JtJ_asymm_filename);
    // Differentiate x
    //deriv.at(0) = 1;
    //bf_JtJ = bf_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //bf_JtJ->Print(bf_JtJ_filename + "_dx");
    //mm_JtJ->Print(mm_JtJ_filename + "_dx");
    // Differentiate y
    //deriv.at(0) = 0;
    //deriv.at(1) = 1;
    //bf_JtJ = bf_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //bf_JtJ->Print(bf_JtJ_filename + "_dy");
    //mm_JtJ->Print(mm_JtJ_filename + "_dy");
    // Differentiate z
    //deriv.at(1) = 0;
    //deriv.at(2) = 1;
    //bf_JtJ = bf_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //mm_JtJ = mm_spline_field_ptr_->JtJ(deriv,ima1,ima2,&mask,prec);
    //bf_JtJ->Print(bf_JtJ_filename + "_dz");
    //mm_JtJ->Print(mm_JtJ_filename + "_dz");
    // Differentiate x for spl1 and y for spl2
    //bf_JtJ_asymm = bf_spline_field_ptr_->JtJ(deriv_x_,ima1,deriv_y_,ima2,&mask,prec);
    //mm_JtJ_asymm = mm_spline_field_ptr_->JtJ(deriv_x_,ima1,deriv_y_,ima2,&mask,prec);
    //bf_JtJ_asymm->Print(bf_JtJ_asymm_filename + "_dx_dy");
    //mm_JtJ_asymm->Print(mm_JtJ_asymm_filename + "_dx_dy");
/*
    // Calculate error
    //auto error_JtJ = difference_JtJ(read_JtJ(bf_JtJ_filename),read_JtJ(mm_JtJ_filename));
    auto bf_JtJ_data = read_JtJ(bf_JtJ_filename);
    auto mm_JtJ_data = read_JtJ(mm_JtJ_filename);
    //std::cout << error_JtJ.at(0).size() << std::endl;
    //std::cout << error_JtJ.at(1).size() << std::endl;
    //std::cout << error_JtJ.at(2).size() << std::endl;

    // Test Hessians are the same
    for (std::vector<std::vector<float> >::iterator bf_i = bf_JtJ_data.begin(), mm_i = mm_JtJ_data.begin()
        ; bf_i != bf_JtJ_data.end() && mm_i != mm_JtJ_data.end()
        ; ++bf_i, ++mm_i){
      for (std::vector<float>::iterator bf_j = bf_i->begin(), mm_j = mm_i->begin()
          ; bf_j != bf_i->end() && mm_j != mm_i->end()
          ; ++bf_j, ++mm_j){
        EXPECT_NEAR(*bf_j, *mm_j, 1e-5 * abs(*bf_j));
      }
    }
*/
    /*
    for (const auto& i : error_JtJ){
      for (const auto& j : i){
        EXPECT_NEAR(0.0f, j, 0.1f);
      }
    }
    */
  }
} // namespace
