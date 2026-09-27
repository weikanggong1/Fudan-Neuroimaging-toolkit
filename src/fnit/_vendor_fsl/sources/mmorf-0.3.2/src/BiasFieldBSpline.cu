//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Multiplicative bias field for correcting bias corrupted images
/// \details Based on MMORF::VolumeBSpline objects, allowing efficient sampling using the GPU
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "BiasFieldBSpline.cuh"
#include "MmorfMemory.h"
#include "VolumeBSpline.cuh"

#include <armadillo>

#include <vector>
#include <utility>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class BiasFieldBSpline::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct empty field of given dimensions and resolution
//      Impl(
//          const std::vector<int>& dimensions,
//          const std::vector<float>& resolution,
//          const std::vector<float>& origin)
//        : sub_biass_(3,MMORF::VolumeBSpline(dimensions, resolution, origin))
//        , knot_spacing_(static_cast<float>(dimensions.at(0))/resolution.at(0))
//      {}
      /// Construct empty field of given dimensions and resolution
      Impl(
          const std::pair<std::vector<float>,std::vector<float> >& extents,
          float knot_spacing)
        : knot_spacing_(knot_spacing)
        , bias_field_(std::vector<int>(3,1), std::vector<float>(3, 1.0f), std::vector<float>(3,0))
      {
        /// \todo Check for correct dimensionality, and top > bottom
        auto dimensions = std::vector<int>(extents.first.size());
        auto resolution = std::vector<float>(extents.first.size(),knot_spacing);
        auto origin = std::vector<float>(extents.first.size());
        // Might we need to bias the corners in order to define the nonlinear field in terms
        // of the affine transformed space?
        // For now, just calculate where the knots go.
        for (auto i = 0; i < extents.first.size(); ++i){
          // Find extent of dimension in mm
          auto extent_i = extents.second[i] - extents.first[i];
          // Calculate number of knots to completely cover the space, and have one knot either
          // side of the original extent. We therefore add 3: 1 on each side + 1 for fencepost
          // problem
          auto n_knots_i = static_cast<int>(std::ceil(extent_i/knot_spacing)) + 3;
          // Calculate at/between which knots the origin is positioned. Note the "+1" to
          // account for the extra spline we add at the beginning of each dimension
          auto origin_i = -extents.first[i]/knot_spacing + 1.0f ;
          // Update dimensions and origin vectors
          dimensions[i] = n_knots_i;
          origin[i] = origin_i;
        }
        bias_field_ = MMORF::VolumeBSpline(
            dimensions, resolution, origin, true);
        // Default bias field parameters to 1
        auto parameters = bias_field_.get_coefficients();
        for (auto& param : parameters){
          param = 1.0f;
        }
        bias_field_.set_coefficients(parameters);
      }

      /// Sample the bias field at specified positions
      std::vector<float> sample_bias(
          const std::vector<std::vector<float> >& positions) const
      {
        auto samples = bias_field_.sample(positions);
        return samples;
      }

      /// Given a set of positions and intensities, return the biased samples
      std::vector<float> apply_bias(
          const std::vector<std::vector<float> >& positions,
          const std::vector<float>& intensities) const
      {
        /// \todo Exception testing
        assert(positions[0].size() == intensities.size());
        auto intensities_out = intensities;
        // Sample bias field
        auto samples = bias_field_.sample(positions);
        // Multiply intensities by bias field
        std::transform(
            intensities.begin(),
            intensities.end(),
            samples.begin(),
            intensities_out.begin(),
            std::multiplies<float>());
        return intensities_out;
      }

      /// Get the parameters defining the bias
      /// \details Returns a vector containing the b-spline parameters for the bias field
      std::vector<std::vector<float> >get_parameters() const
      {
        auto parameters_out = std::vector<std::vector<float> >(
            1, bias_field_.get_coefficients());
        return parameters_out;
      }

      /// Set the parameters defining the bias
      /// \details Accepts a vector containing the b-spline parameters for the bias field
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        /// \todo Exception testing
        bias_field_.set_coefficients(parameters[0]);
      }

      /// Get the dimensions of the bias field
      std::vector<int> get_dimensions() const
      {
        return bias_field_.get_dimensions();
      }

      /// Get the extents of the bias field
      std::pair<std::vector<float>,std::vector<float> > get_extents() const
      {
        return bias_field_.get_extents();
      }

      /// Get the number of parameters in bias
      /// \details pair.first  =  1 as we only have one underlying volume
      ///          pair.second = number of parameters per subbias
      std::pair<int,int> get_parameter_size() const
      {
        auto parameters_size = 1;
        for (const auto& dim_parameters : bias_field_.get_dimensions()){
          parameters_size *= dim_parameters;
        }
        return std::make_pair(1,parameters_size);
      }

      /// Get the volume representing the bias field in the specified direction
      MMORF::VolumeBSpline get_bias_vol() const
      {
        return bias_field_;
      }

      /// Get the knot-spacing of the bias field in mm
      float get_knot_spacing() const
      {
        return knot_spacing_;
      }

      /// Scale the knot spacing - i.e. change the resolution of our bias field
      void reparameterise(const int scaling_factor)
      {
        bias_field_ = bias_field_.reparameterise(scaling_factor);
        knot_spacing_ /= static_cast<float>(scaling_factor);
      }

      /// Crop the bias field
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      void crop(
          const std::pair<std::vector<float>, std::vector<float> > extents)
      {
        bias_field_ = bias_field_.crop(extents);
      }

      /// Return the set of regularly positioned sample positions which all have the maximum
      /// number of splines with support in that region for a given sampling frequency (as in
      /// frequency per knot-spacing)
      std::vector<std::vector<float> > get_robust_sample_positions(
          const int sampling_frequency) const
      {
        // Get extents
        auto extents_full = bias_field_.get_extents();
        // Get robust sample dimensions
        auto sample_dimensions = get_robust_sample_dimensions(sampling_frequency);
        // Get bias resolution
        auto resolution = bias_field_.get_resolution();
        // Calculate sample steps
        auto sample_step = std::vector<float>();
        for (const auto& res : resolution){
          sample_step.push_back(res/static_cast<float>(sampling_frequency));
        }
        // Convert to robust samples
        // This involves extracting the range one sample in from each edge, hence the slighly
        // odd range of the for loops
        auto sample_positions = std::vector<std::vector<float> >(sample_dimensions.size());
        for (auto z = 0; z < sample_dimensions[2]; ++z){
          for (auto y = 0; y < sample_dimensions[1]; ++y){
            for (auto x = 0; x < sample_dimensions[0]; ++x){
              sample_positions[0].push_back(
                  extents_full.first[0]
                  + resolution[0]
                  + (static_cast<float>(x) * sample_step[0])
                  );
              sample_positions[1].push_back(
                  extents_full.first[1]
                  + resolution[1]
                  + (static_cast<float>(y) * sample_step[1])
                  );
              sample_positions[2].push_back(
                  extents_full.first[2]
                  + resolution[2]
                  + (static_cast<float>(z) * sample_step[2])
                  );
            }
          }
        }
        return sample_positions;
      }

      /// Return the dimensions of the set of regularly spaced sample positions which all have
      /// the maximum number of splines with support in that region for a given sampling
      /// frequency (as in frequency per knot-spacing)
      std::vector<int> get_robust_sample_dimensions(
          const int sampling_frequency) const
      {
        auto sample_dimensions = get_dimensions();
        for (auto& dim : sample_dimensions){
          /// \todo Replace assert with exception
          assert(dim > 3);
          dim = ((dim - 3) * sampling_frequency) + 1;
        }
        return sample_dimensions;
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////

      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      MMORF::VolumeBSpline bias_field_;
      float knot_spacing_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  BiasFieldBSpline::~BiasFieldBSpline() = default;
  /// Move ctor
  BiasFieldBSpline::BiasFieldBSpline(BiasFieldBSpline&& rhs) = default;
  /// Move assignment operator
  BiasFieldBSpline& BiasFieldBSpline::operator=(BiasFieldBSpline&& rhs) = default;
  /// Copy ctor
  BiasFieldBSpline::BiasFieldBSpline(const BiasFieldBSpline& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  BiasFieldBSpline& BiasFieldBSpline::operator=(const BiasFieldBSpline& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
    else{
      *pimpl_ = *rhs.pimpl_;
    }
    return *this;
  }
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
  /// Construct empty field of given dimensions and resolution
//  BiasFieldBSpline::BiasFieldBSpline(
//      const std::vector<int>& dimensions,
//      const std::vector<float>& resolution,
//      const std::vector<float>& origin)
//    : pimpl_(MMORF::make_unique<Impl>(dimensions,resolution,origin))
//  {}
  /// Construct empty field of given dimensions and resolution
  BiasFieldBSpline::BiasFieldBSpline(
      const std::pair<std::vector<float>,std::vector<float> >& extents,
      float knot_spacing)
    : pimpl_(MMORF::make_unique<Impl>(extents,knot_spacing))
  {}
  /// Sample the bias field at specified positions
  std::vector<float> BiasFieldBSpline::sample_bias(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->sample_bias(positions);
  }
  /// Given a set of positions and intensities, return the biased intensities
  std::vector<float> BiasFieldBSpline::apply_bias(
      const std::vector<std::vector<float> >& positions,
      const std::vector<float>& samples) const
  {
    return pimpl_->apply_bias(positions, samples);
  }
  /// Get the parameters defining the bias
  /// \details Returns a vector containing the b-spline parameters for the bias field
  std::vector<std::vector<float> >BiasFieldBSpline::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the parameters defining the bias
  /// \details Accepts a vector containing the b-spline parameters for the bias field
  void BiasFieldBSpline::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get the dimensions of the subbiass
  std::vector<int> BiasFieldBSpline::get_dimensions() const
  {
    return pimpl_->get_dimensions();
  }
  /// Get the extents of the bias field
  std::pair<std::vector<float>,std::vector<float> > BiasFieldBSpline::get_extents() const
  {
    return pimpl_->get_extents();
  }
  /// Get the number of parameters in bias
  /// \details pair.first  = number of subbiass
  ///          pair.second = number of parameters per subbias
  std::pair<int,int> BiasFieldBSpline::get_parameter_size() const
  {
    return pimpl_->get_parameter_size();
  }
  /// Get the volume representing the bias field
  MMORF::VolumeBSpline BiasFieldBSpline::get_bias_vol() const
  {
    return pimpl_->get_bias_vol();
  }
  /// Get the knot-spacing of the bias field in mm
  float BiasFieldBSpline::get_knot_spacing() const
  {
    return pimpl_->get_knot_spacing();
  }
  /// Reparameterise the bias field
  /// \details This increases the number of parameters used in the bias field by an
  ///          integer factor. In doing so this maintains a one-to-one relationship
  ///          between the biass at each level
  BiasFieldBSpline BiasFieldBSpline::reparameterise(const int scaling_factor) const
  {
    /// \todo Replace assert with exception
    assert (scaling_factor > 0);
    auto return_bias_field = (*this);
    return_bias_field.pimpl_->reparameterise(scaling_factor);
    return return_bias_field;
  }
  /// Crop the bias field
  /// \details The underlying samples will be reduced such that the given extents still
  ///          have the maximum number of splines with support at that position
  BiasFieldBSpline BiasFieldBSpline::crop(
      const std::pair<std::vector<float>, std::vector<float> > extents) const
  {
    auto return_bias_field = (*this);
    return_bias_field.pimpl_->crop(extents);
    return return_bias_field;
  }
  /// Return the set of regularly positioned sample positions which all have the maximum
  /// number of splines with support in that region for a given sampling frequency (as in
  /// frequency per knot-spacing)
  std::vector<std::vector<float> > BiasFieldBSpline::get_robust_sample_positions(
      const int sampling_frequency) const
  {
    return pimpl_->get_robust_sample_positions(sampling_frequency);
  }
  /// Return the dimensions of the set of regularly spaced sample positions which all have
  /// the maximum number of splines with support in that region for a given sampling
  /// frequency (as in frequency per knot-spacing)
  std::vector<int> BiasFieldBSpline::get_robust_sample_dimensions(
      const int sampling_frequency) const
  {
    return pimpl_->get_robust_sample_dimensions(sampling_frequency);
  }
} // MMORF
