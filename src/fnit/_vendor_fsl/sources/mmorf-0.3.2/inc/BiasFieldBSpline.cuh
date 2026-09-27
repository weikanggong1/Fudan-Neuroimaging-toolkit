//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Multiplicative bias field for correcting bias corrupted images
/// \details Based on MMORF::VolumeBSpline objects, allowing efficient sampling using the GPU
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef BIAS_FIELD_B_SPLINE_CUH
#define BIAS_FIELD_B_SPLINE_CUH

#include "BiasField.h"
#include "VolumeBSpline.cuh"

#include <armadillo>

#include <vector>
#include <memory>
#include <utility>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class BiasFieldBSpline : public BiasField
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~BiasFieldBSpline();
      /// Move ctor
      BiasFieldBSpline(BiasFieldBSpline&& rhs);
      /// Move assignment operator
      BiasFieldBSpline& operator=(BiasFieldBSpline&& rhs);
      /// Copy ctor
      BiasFieldBSpline(const BiasFieldBSpline& rhs);
      /// Copy assignment operator
      BiasFieldBSpline& operator=(const BiasFieldBSpline& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct empty field of given dimensions and resolution
//      BiasFieldBSpline(
//          const std::vector<int>& dimensions,
//          const std::vector<float>& resolution,
//          const std::vector<float>& origin);
      // Construct empty field of given dimensions and resolution
      /// \param extents Pair of positions representing min and max vertices of the field
      /// \param knot_spacing Defines the offset between each B-spline
      BiasFieldBSpline(
          const std::pair<std::vector<float>,std::vector<float> >& extents,
          float knot_spacing);
      /// Sample the bias field at specified positions
      std::vector<float> sample_bias(
          const std::vector<std::vector<float> >& positions) const override;
      /// Given a set of positions and intensities, return the biased intensities
      std::vector<float> apply_bias(
          const std::vector<std::vector<float> >& positions,
          const std::vector<float>&               intensities) const override;
      /// Get the parameters defining the bias
      /// \details Returns a vector of vectors containing the b-spline parameters for each
      ///          direction of the bias
      std::vector<std::vector<float> >get_parameters() const override;
      /// Set the parameters defining the bias
      /// \details Accepts a vector of vectors containing the b-spline parameters for each
      ///          direction of the bias
      void set_parameters(
          const std::vector<std::vector<float> >& parameters) ;
      /// Get the dimensions of the bias field
      std::vector<int> get_dimensions() const override;
      /// Get the extents of the bias field
      std::pair<std::vector<float>,std::vector<float> > get_extents() const override;
      /// Get the number of parameters in bias field
      std::pair<int,int> get_parameter_size() const;
      /// Get the volume representing the bias field
      MMORF::VolumeBSpline get_bias_vol() const;
      /// Get the knot-spacing of the bias field in mm
      float get_knot_spacing() const;
      /// Reparameterise the bias field
      /// \details This increases the number of parameters used in the bias field by an
      ///          integer factor. In doing so this maintains a one-to-one relationship
      ///          between the bias fields at each level
      BiasFieldBSpline reparameterise(const int scaling_factor) const;
      /// Crop the bias field
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      BiasFieldBSpline crop(
          const std::pair<std::vector<float>, std::vector<float> > extents) const;
      /// Return the set of regularly positioned sample positions which all have the maximum
      /// number of splines with support in that region for a given sampling frequency (as in
      /// frequency per knot-spacing)
      std::vector<std::vector<float> > get_robust_sample_positions(
          const int sampling_frequency) const;
      /// Return the dimensions of the set of regularly spaced sample positions which all have
      /// the maximum number of splines with support in that region for a given sampling
      /// frequency (as in frequency per knot-spacing)
      std::vector<int> get_robust_sample_dimensions(
          const int sampling_frequency) const;

    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // BiasFieldBSpline
} // MMORF
#endif // BIAS_FIELD_B_SPLINE_CUH
