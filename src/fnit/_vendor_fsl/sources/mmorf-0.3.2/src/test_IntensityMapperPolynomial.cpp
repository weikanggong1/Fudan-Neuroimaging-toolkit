// Unit Test File

#include "MmorfMemory.h"
#include "IntensityMapperPolynomial.cuh"
#include "VolumeBSpline.cuh"

#include "gtest/gtest.h"

#include <string>
#include <memory>
#include <vector>
#include <iostream>

namespace
{
  class IntensityMapperPolynomialTest : public ::testing::Test
  {
    public:
      IntensityMapperPolynomialTest()
        : filename_domain_("./data/HCP_T1_2_flirt.nii.gz")
        , filename_range_("./data/HCP_T1_2_flirt.nii.gz")
        , filename_mask_("./data/HCP_T1_brain_mask.nii.gz")
        , vol_domain_(filename_domain_)
        , vol_range_(filename_range_)
        , vol_mask_(filename_mask_)
        , sample_positions_(vol_domain_.get_original_sample_positions())
        , samples_domain_(vol_domain_.sample(sample_positions_))
        , samples_range_(vol_range_.sample(sample_positions_))
        , samples_mask_(vol_mask_.sample(sample_positions_))
      {
        for (auto i = 0; i < samples_mask_.size(); ++i){
          if (samples_mask_[i] > 0.1f){
            samples_domain_masked_.push_back(samples_domain_[i]);
            samples_range_masked_.push_back(samples_range_[i]);
          }
        }
      }

    protected:
      const std::string           filename_domain_;
      const std::string           filename_range_;
      const std::string           filename_mask_;
      MMORF::VolumeBSpline        vol_domain_;
      MMORF::VolumeBSpline        vol_range_;
      MMORF::VolumeBSpline        vol_mask_;
      std::vector<vector<float> > sample_positions_;
      std::vector<float>          samples_domain_;
      std::vector<float>          samples_range_;
      std::vector<float>          samples_mask_;
      std::vector<float>          samples_domain_masked_;
      std::vector<float>          samples_range_masked_;
  }; // IntensityMapperPolynomialTest


  TEST_F(IntensityMapperPolynomialTest, ctor)
  {
    auto mapper = MMORF::IntensityMapperPolynomial(
        3,
        samples_domain_masked_,
        samples_range_masked_);
  } // ctor

  TEST_F(IntensityMapperPolynomialTest, func_map_intensities)
  {
    auto mapper = MMORF::IntensityMapperPolynomial(
        3,
        samples_domain_masked_,
        samples_range_masked_);
    std::cout << "Start" << std::endl;
    auto mapped_vals = mapper.map_intensities(samples_domain_);
    std::cout << "End" << std::endl;
  } // func_map_intensities

} // namespace
