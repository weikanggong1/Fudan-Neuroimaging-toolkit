//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Volume object storing a tensor field
/// \details The main function of this class is for facilitating DTI registration, and as
///          such the functionality is geared towards that, rather than just a general 4D
///          volume object. The intention is for this to be used in a CostFxnDTI type object.
/// \author Frederik Lange
/// \date July 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef VOLUME_TENSOR_CUH
#define VOLUME_TENSOR_CUH

#include "newimage/newimageall.h"

#include <thrust/host_vector.h>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class VolumeTensor
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~VolumeTensor();
      /// Move ctor
      VolumeTensor(VolumeTensor&& rhs);
      /// Move assignment operator
      VolumeTensor& operator=(VolumeTensor&& rhs);
      /// Copy ctor
      VolumeTensor(const VolumeTensor& rhs);
      /// Copy assignment operator
      VolumeTensor& operator=(const VolumeTensor& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct from NEWIMAGE::volume
      VolumeTensor(
          const NEWIMAGE::volume4D<float>&  vol,
          const bool                        apply_log = false);
      /// Construct from nifti image file
      VolumeTensor(
          const std::string vol_name,
          const bool        apply_log = false);
      /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
      VolumeTensor(
          const NEWIMAGE::volume4D<float>&  vol,
          float                             smooth_fwhm_mm,
          const bool                        apply_log = false);
      /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
      VolumeTensor(
          const std::string vol_name,
          const float       smooth_fwhm_mm,
          const bool        apply_log = false);
      /// Sample tensor at specified coordinates
      /// \details Each returned vector represents an elements of the tensor at each specified
      ///          sample position. Note that even though the tensor is symmetric, all 9
      ///          elements are returned. Elements are returned in row-major ordering, i.e.
      ///          for n samples:
      ///                                                     [00_0 00_1 ... 00_n]
      ///                                                     [01_0 01_1 ... 01_n]
      ///                                                     [02_0 02_1 ... 02_n]
      ///                         [00_n 01_n 02_n]            [10_0 10_1 ... 02_n]
      ///                   D_n = [10_n 11_n 12_n] => D_vec = [11_0 11_1 ... 02_n]
      ///                         [20_n 21_n 22_n]            [12_0 12_1 ... 02_n]
      ///                                                     [20_0 20_1 ... 20_n]
      ///                                                     [21_0 21_1 ... 21_n]
      ///                                                     [22_0 22_1 ... 22_n]
      std::vector<std::vector<float> > sample(
          const std::vector<std::vector<float> >& positions) const;
      /// Sample the spatial derivative of the elements of the tensor at specified positions
      /// in all directions
      std::vector<std::vector<thrust::host_vector<float> > > sample_gradient(
          const std::vector<std::vector<float> >& positions) const;
      /// Get the resolution of the underlying parameterisation
      std::vector<float> get_resolution() const;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // VolumeTensor
} // MMORF
#endif // VOLUME_TENSOR_CUH
