///
/// \file
/// \brief Implementation of MMORF::Volume based on the CubicInterpolationCUDA code by Danny
///        Ruijters (see: https://github.com/DannyRuijters/CubicInterpolationCUDA)
/// \details This volume resides exclusively on the GPU and allows for rapid cubic
///          interpolation of values within the volume. It uses regular cubic B-splines, and
///          stores the spline coefficients as a 3D device texture, but this shouldn't affect
///          the interface.
/// \author Frederik Lange
/// \date April 2018
/// \copyright FMRIB Copyright Licence

#ifndef VOLUME_B_SPLINE_CUH
#define VOLUME_B_SPLINE_CUH

#include "Volume.h"

#include "newimage/newimageall.h"

#include <thrust/host_vector.h>

#include <armadillo>

#include <vector>
#include <memory>
#include <utility>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class VolumeBSpline : public Volume
  {
    public:
      /// Empty volume of given dimensions and resolution
      /// \param dimensions Dimensions of volume in VOXELS
      /// \param resolution Resolution of a voxel in MM
      /// \param origin Position of origin in VOXELS
      VolumeBSpline(
          const std::vector<int>&   dimensions,
          const std::vector<float>& resolution,
          const std::vector<float>& origin,
          const bool                clamp_border = false);
      /// Construct from NEWIMAGE::volume
      VolumeBSpline(
          const NEWIMAGE::volume<float>&  vol,
          const bool                      clamp_border = false);
      /// Construct from nifti image file
      VolumeBSpline(
          std::string vol_name,
          const bool  clamp_border = false);
      /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
      VolumeBSpline(
          const NEWIMAGE::volume<float>&  vol,
          const float                     smooth_fwhm_mm,
          const bool                      implicit_mask = false,
          const bool                      clamp_border = false,
          const bool                      debug = false);
      /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
      VolumeBSpline(
          std::string       vol_name,
          const float       smooth_fwhm_mm,
          const bool        implicit_mask = false,
          const bool        clamp_border = false,
          const bool        debug = false);
      /// Default dtor
      ~VolumeBSpline();
      /// Move ctor
      VolumeBSpline(VolumeBSpline&& rhs);
      /// Move assignment operator
      VolumeBSpline& operator=(VolumeBSpline&& rhs);
      /// Copy ctor
      VolumeBSpline(const VolumeBSpline& rhs);
      /// Copy assignment operator
      VolumeBSpline& operator=(const VolumeBSpline& rhs);
      /// Return dimensions of volume
      virtual std::vector<int> get_dimensions() const override;
      /// Return values at all specified positions
      virtual std::vector<float> sample(
          const std::vector<std::vector<float> >& positions) const override;
      /// Return value of volume at all specified positions, differentiated in all dimensions
      virtual std::vector<thrust::host_vector<float> >sample_gradient(
          const std::vector<std::vector<float> >& positions) const override;
      /// Get original sample points in mm
      virtual std::vector<std::vector<float> > get_original_sample_positions() const override;
      /// Set b-spline coefficients
      void set_coefficients(const std::vector<float>& coefficients);
      /// Get b-spline coefficients
      std::vector<float> get_coefficients() const;
      /// Get valid extents of volume in real coordinates (mm)
      /// \details Each vector in the pair represents one vertex of the volume, and the
      ///          difference between them gives the valid range for each volume. NOTE: These
      ///          values ARE ordered, i.e. it is guaranteed that:
      ///                             pair.first =< pair.second
      std::pair<std::vector<float>, std::vector<float> > get_extents() const;
      /// Get a reparameterised copy of this volume
      /// \details This increases the number of parameters used in the b-spline representation
      ///          by an integer factor in all directions. This should maintain a one-to-one
      ///          mapping between interpolated intensities in both parameterisations
      VolumeBSpline reparameterise(const int scaling_factor) const;
      /// Crop the volume to the defined extents
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      VolumeBSpline crop(
          const std::pair<std::vector<float>, std::vector<float> > extents) const;
      /// Get the resolution of the underlying parameterisation
      std::vector<float> get_resolution() const;
      /// Get the mm_to_vox rotation
      arma::fmat33 get_mm_to_vox_rotation() const;
    private:
      /// Forward declaration
      class                 Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
      /// Global class identifier allocator
      static int            next_id_;
  }; // VolumeBSpline
} // MMORF
#endif // VOLUME_B_SPLINE_CUH
