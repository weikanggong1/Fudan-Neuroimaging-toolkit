//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Map intensities between modalities
/// \details Provides the interface for mapping from one set of intensity values to another.
//           this makes no assumptions about linear/nonlinear mapping, but merely provides a
//           consistant interface.
/// \author Frederik Lange
/// \date September 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef INTENSITY_MAPPER_H
#define INTENSITY_MAPPER_H

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class IntensityMapper
  {
    public:
      /// Default destructor
      virtual ~IntensityMapper() {}
      /// Map intensities from the domain to the range
      /// \param intensities_domain Intensity values in the domain to be mapped to the range
      virtual std::vector<float> map_intensities(
          const std::vector<float>& intensities_domain) const = 0;
  }; // IntensityMapper
} // MMORF
#endif // INTENSITY_MAPPER_H
