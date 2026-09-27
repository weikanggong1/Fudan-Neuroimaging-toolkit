///
/// \file
/// \brief Warp Field interface.
/// \details Provides the interface to which all Warp Fields must conform
/// \author Frederik Lange
/// \date December 2017
/// \copyright FMRIB Copyright Licence

#ifndef WARP_FIELD_H
#define WARP_FIELD_H

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Vector field describing a deformation.
  class WarpField
  {
    public:
      /// Default destructor
      virtual ~WarpField() {}
      /// Sample the warp field at specified positions
      virtual std::vector<std::vector<float> > sample_warp(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Given a set of positions, return the warped positions
      virtual std::vector<std::vector<float> > apply_warp(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Get the parameters defining the warp
      virtual std::vector<std::vector<float> > get_parameters() const = 0;
      /// Set the parameters defining the warp
      virtual void set_parameters(
          const std::vector<std::vector<float> >& parameters) = 0;
      /// Get the dimensions of the subwarps
      virtual std::vector<int> get_dimensions() const = 0;
      /// Get the extents of the warp field
      virtual std::pair<std::vector<float>,std::vector<float> > get_extents() const = 0;
      /// Given a set of positions, return the Jacobian determinant at those positions
      virtual std::vector<float> jacobian_determinants(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Given a set of positions, return the elements of the local Jacobian matrix at those
      /// positions
      virtual std::vector<std::vector<float> > get_jacobian_elements(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Scale warp field parameters by constant value
      virtual void scale_parameters(
          const float scaling) = 0;
  }; // WarpField
} // MMORF
#endif // WARP_FIELD_H
