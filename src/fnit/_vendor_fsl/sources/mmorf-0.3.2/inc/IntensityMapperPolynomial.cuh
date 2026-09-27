//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Map intensities between volumes using a polynomian function
/// \details Takes a set of intensities in the domain and maps them to intensities in the
///          range using a polynomial mapping. Is capable of calculating a best fit polynomial
///          of a given degree when supplied with set of "experimental" or "measured" pairs
///          of values.
/// \author Frederik Lange
/// \date September 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef INTENSITY_MAPPER_POLYNOMIAL
#define INTENSITY_MAPPER_POLYNOMIAL

#include "IntensityMapper.h"

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class IntensityMapperPolynomial : public IntensityMapper
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~IntensityMapperPolynomial();
      /// Move ctor
      IntensityMapperPolynomial(IntensityMapperPolynomial&& rhs);
      /// Move assignment operator
      IntensityMapperPolynomial& operator=(IntensityMapperPolynomial&& rhs);
      /// Copy ctor
      IntensityMapperPolynomial(const IntensityMapperPolynomial& rhs);
      /// Copy assignment operator
      IntensityMapperPolynomial& operator=(const IntensityMapperPolynomial& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Default ctor
      IntensityMapperPolynomial();
      /// Sampled data constructor
      /// \details Supply a set of sampled intensities in the domain and the range and specify
      ///          the degree of the polynomial to fit. Note that the number of samples must
      ///          be greater than the degree of the polynomial being fitted.
      /// \param polynomial_degree Degree of polynomial to fit to sampled intensities
      /// \param intensities_domain Intensities in the domain image (i.e. intensities we map
      ///        FROM)
      /// \param intensities_range Intensities in the range image (i.e. intensities we map TO)
      IntensityMapperPolynomial(
          const int                 polynomial_degree,
          const std::vector<float>& intensities_domain,
          const std::vector<float>& intensities_range,
          const bool                debug = false);
      /// Map intensities from the domain to the range
      /// \param intensities_domain Intensity values in the domain to be mapped to the range
      virtual std::vector<float> map_intensities(
          const std::vector<float>& intensities_domain) const override;
    private:
      /// Forward declaration
      class                 Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
      /// Global class identifier allocator
      static int            next_id_;
  }; // IntensityMapperPolynomial
} // MMORF
#endif // INTENSITY_MAPPER_POLYNOMIAL
