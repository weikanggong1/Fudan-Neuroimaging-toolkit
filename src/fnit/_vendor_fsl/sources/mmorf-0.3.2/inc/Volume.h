///
/// \file
/// \brief Interface for a Volume object
/// \details The Volume is a critical abstraction for dealing with medical imaging data. This
///          interface sets out the contract that I would like all Volume type objects to
///          adhere to. In particular, this will enforce expectations such as that values
///          within a volume are always accessed using mm coordinates, referenced to the same
///          point in the volume. Note that there may be problems with speed of accessing
///          data etc, but this will only be looked into if it is found to significantly
///          affect performance. In general then, clarity and reusability will be given
///          preference over performance.
/// \author Frederik Lange
/// \date January 2018
/// \copyright FMRIB Copyright Licence


#ifndef VOLUME_H
#define VOLUME_H

#include <thrust/host_vector.h>

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class Volume
  {
    public:
      /// Default destructor
      virtual ~Volume() {}
      /// Return dimensions of volume
      virtual std::vector<int> get_dimensions() const = 0;
      /// Return values at all specified positions
      virtual std::vector<float> sample(
        const std::vector<std::vector<float> >& positions) const = 0;
      /// Return value of volume at all specified positions, differentiated in all dimensions
      virtual std::vector<thrust::host_vector<float> > sample_gradient(
          const std::vector<std::vector<float> >& positions) const = 0;
      /// Get original sample points in mm
      virtual std::vector<std::vector<float> > get_original_sample_positions() const = 0;

  }; // Volume
} // MMORF
#endif // VOLUME_H
