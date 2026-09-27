//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Warp field for transforming sets of coordinates
/// \details Based on MMORF::VolumeBSpline objects, allowing efficient sampling using the GPU
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef WARP_FIELD_B_SPLINE_CUH
#define WARP_FIELD_B_SPLINE_CUH

#include "WarpField.h"
#include "VolumeBSpline.cuh"

#include <armadillo>

#include <vector>
#include <memory>
#include <utility>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class WarpFieldBSpline : public WarpField
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~WarpFieldBSpline();
      /// Move ctor
      WarpFieldBSpline(WarpFieldBSpline&& rhs);
      /// Move assignment operator
      WarpFieldBSpline& operator=(WarpFieldBSpline&& rhs);
      /// Copy ctor
      WarpFieldBSpline(const WarpFieldBSpline& rhs);
      /// Copy assignment operator
      WarpFieldBSpline& operator=(const WarpFieldBSpline& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct empty field of given dimensions and resolution
      WarpFieldBSpline(
          const std::vector<int>& dimensions,
          const std::vector<float>& resolution,
          const std::vector<float>& origin);
      /// Construct empty field of given dimensions and resolution
      /// \param extents Pair of positions representing min and max vertices of the field
      /// \param corner_bottom Point with lowest value in all dimensions (mm)
      /// \param corner_top Point with highest value in all dimensions (mm)
      /// \param knot_spacing Defines the offset between each B-spline
      WarpFieldBSpline(
          const std::pair<std::vector<float>,std::vector<float> >& extents,
          float knot_spacing);
      /// Sample the warp field at specified positions
      std::vector<std::vector<float> > sample_warp(
          const std::vector<std::vector<float> >& positions) const override;
      /// Given a set of positions, return the warped positions
      std::vector<std::vector<float> > apply_warp(
          const std::vector<std::vector<float> >& positions) const override;
      /// Get the parameters defining the warp
      /// \details Returns a vector of vectors containing the b-spline parameters for each
      ///          direction of the warp
      std::vector<std::vector<float> > get_parameters() const override;
      /// Set the parameters defining the warp
      /// \details Accepts a vector of vectors containing the b-spline parameters for each
      ///          direction of the warp
      void set_parameters(
          const std::vector<std::vector<float> >& parameters) override;
      /// Get the dimensions of the subwarps
      std::vector<int> get_dimensions() const override;
      /// Get the extents of the warp field
      std::pair<std::vector<float>,std::vector<float> > get_extents() const override;
      /// Given a set of positions, return the Jacobian determinant at those positions
      std::vector<float> jacobian_determinants(
          const std::vector<std::vector<float> >& positions) const override;
      /// Given a set of positions, return the elements of the local Jacobian matrix at those
      /// positions
      std::vector<std::vector<float> > get_jacobian_elements(
          const std::vector<std::vector<float> >& positions) const override;
      /// Scale warp field parameters by constant value
      void scale_parameters(
          const float scaling) override;
      /// Get the number of parameters in warp
      /// \details pair.first  = number of subwarps
      ///          pair.second = number of parameters per subwarp
      std::pair<int,int> get_parameter_size() const;
      /// Get the volume representing the warp field in the specified direction
      MMORF::VolumeBSpline get_sub_warp(int warp_index) const;
      /// Get the knot-spacing of the warp field in mm
      float get_knot_spacing() const;
      /// Reparameterise the warp field
      /// \details This increases the number of parameters used in the warp field by an
      ///          integer factor. In doing so this maintains a one-to-one relationship
      ///          between the warps at each level
      void reparameterise(const int scaling_factor) const;
      /// Crop the warp field
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      void crop(const std::pair<std::vector<float>, std::vector<float> > extents) const;
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
      /// Apply both an affine transform and a warp to a given set of positions
      /// \details This is necessary when the affine transform is acting only on the original
      ///          sample positions, and independent of the warp field
      std::vector<std::vector<float> > apply_affine_then_warp(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const;
      /// Warp the given positions and then apply the affine transform.
      /// \details This is necessary in a number of situations, such as when finding the
      ///          derivative of a volume in a space reached via the given affine transform
      std::vector<std::vector<float> > apply_warp_then_affine(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const;

    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // WarpFieldBSpline
} // MMORF
#endif // WARP_FIELD_B_SPLINE_CUH
