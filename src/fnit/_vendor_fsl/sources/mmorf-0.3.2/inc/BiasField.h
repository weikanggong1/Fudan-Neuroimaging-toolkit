//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Multiplicative bias field interface
/// \details Provides the interface towhich all bias fields must conform
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef BIAS_FIELD_H
#define BIAS_FIELD_H

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Vector field describing a deformation.
  class BiasField
  {
    public:
      /// Default destructor
      virtual ~BiasField() {}
      /// Sample the bias field at specified positions
      virtual std::vector<float> sample_bias(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Given a set of positions, return the biased positions
      virtual std::vector<float> apply_bias(
          const std::vector<std::vector<float> >& positions,
          const std::vector<float>&               intensities) const = 0;
      /// Get the parameters defining the bias
      virtual std::vector<std::vector<float> > get_parameters() const = 0;
      /// Set the parameters defining the bias
      virtual void set_parameters(
          const std::vector<std::vector<float> >& parameters) = 0;
      /// Get the dimensions of the bias field
      virtual std::vector<int> get_dimensions() const = 0;
      /// Get the extents of the bias field
      virtual std::pair<std::vector<float>,std::vector<float> > get_extents() const = 0;
  }; // BiasField
} // MMORF
#endif // BIAS_FIELD_H
