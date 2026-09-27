//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Warp field for transforming sets of coordinates
/// \details Based on MMORF::VolumeBSpline objects, allowing efficient sampling using the GPU
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#undef NDEBUG
#include "WarpFieldBSpline.cuh"
#include "MmorfMemory.h"
#include "VolumeBSpline.cuh"
#include "CostFxnKernels.cuh"
#include "helper_cuda.h"

#include <armadillo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>
#include <utility>
#include <random>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class WarpFieldBSpline::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct empty field of given dimensions and resolution
      Impl(
          const std::vector<int>& dimensions,
          const std::vector<float>& resolution,
          const std::vector<float>& origin)
        : sub_warps_(3,MMORF::VolumeBSpline(dimensions, resolution, origin, true))
        , knot_spacing_(static_cast<float>(dimensions.at(0))/resolution.at(0))
      {}
      /// Construct empty field of given dimensions and resolution
      Impl(
          const std::pair<std::vector<float>,std::vector<float> >& extents,
          float knot_spacing)
        : knot_spacing_(knot_spacing)
      {
        /// \todo Check for correct dimensionality, and top > bottom
        auto dimensions = std::vector<int>(extents.first.size());
        auto resolution = std::vector<float>(extents.first.size(),knot_spacing);
        auto origin = std::vector<float>(extents.first.size());
        // Might we need to warp the corners in order to define the nonlinear field in terms
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
        sub_warps_ = std::vector<MMORF::VolumeBSpline>(
            extents.first.size(),
            MMORF::VolumeBSpline(dimensions, resolution, origin, true));
        //random_initialise_warp_();
      }
      /// Sample the warp field at specified positions
      std::vector<std::vector<float> > sample_warp(
          const std::vector<std::vector<float> >& positions) const
      {
        auto samples = std::vector<std::vector<float> >();
        for (const auto& sub_warp : sub_warps_){
          samples.push_back(sub_warp.sample(positions));
        }
        return samples;
      }
      /// Given a set of positions, return the warped positions
      std::vector<std::vector<float> > apply_warp(
          const std::vector<std::vector<float> >& positions) const
      {
        /// \todo Exception testing
        assert(positions.size() == sub_warps_.size());
        auto positions_out = positions;
        // Sample warps
        for (auto i_dim = size_t(0), i_dim_end = positions_out.size();
            i_dim < i_dim_end;
            ++i_dim){
          auto warp_sample = sub_warps_[i_dim].sample(positions);
          for (auto sample = warp_sample.begin(),
                position = positions_out[i_dim].begin();
              sample < warp_sample.end();
              ++sample, ++position)
          {
            *position += *sample;
          }
        }
        return positions_out;
      }
      /// Get the parameters defining the warp
      /// \details Returns a vector of vectors containing the b-spline parameters for each
      ///          direction of the warp
      std::vector<std::vector<float> > get_parameters() const
      {
        auto parameters_out = std::vector<std::vector<float> >(sub_warps_.size());
        auto i_dim = 0;
        for (const auto& sub_warp : sub_warps_){
          parameters_out[i_dim] = sub_warp.get_coefficients();
          ++i_dim;
        }
        return parameters_out;
      }
      /// Set the parameters defining the warp
      /// \details Accepts a vector of vectors containing the b-spline parameters for each
      ///          direction of the warp
      void set_parameters(
          const std::vector<std::vector<float> >& parameters)
      {
        /// \todo Exception testing
        assert(parameters.size() == sub_warps_.size());
        auto i_dim =0;
        for (const auto& sub_param : parameters){
          sub_warps_[i_dim].set_coefficients(sub_param);
          ++i_dim;
        }
      }
      /// Get the dimensions of the subwarps
      std::vector<int> get_dimensions() const
      {
        return sub_warps_[0].get_dimensions();
      }
      /// Get the extents of the warp field
      std::pair<std::vector<float>,std::vector<float> > get_extents() const
      {
        return sub_warps_[0].get_extents();
      }
      /// Given a set of positions, return the Jacobian determinant at those positions
      /// \todo Make this non-specific for 3D
      std::vector<float> jacobian_determinants(
          const std::vector<std::vector<float> >& positions) const
      {
        auto jac_det_sz = positions[0].size();
        auto x_warp_derivatives = sub_warps_[0].sample_gradient(positions);
        auto y_warp_derivatives = sub_warps_[1].sample_gradient(positions);
        auto z_warp_derivatives = sub_warps_[2].sample_gradient(positions);
        auto jac_det = std::vector<float>(jac_det_sz,0.0f);
        // Make everything a device vector using thrust::
        auto j11_dev = thrust::device_vector<float>(x_warp_derivatives[0]);
        auto j12_dev = thrust::device_vector<float>(x_warp_derivatives[1]);
        auto j13_dev = thrust::device_vector<float>(x_warp_derivatives[2]);
        auto j21_dev = thrust::device_vector<float>(y_warp_derivatives[0]);
        auto j22_dev = thrust::device_vector<float>(y_warp_derivatives[1]);
        auto j23_dev = thrust::device_vector<float>(y_warp_derivatives[2]);
        auto j31_dev = thrust::device_vector<float>(z_warp_derivatives[0]);
        auto j32_dev = thrust::device_vector<float>(z_warp_derivatives[1]);
        auto j33_dev = thrust::device_vector<float>(z_warp_derivatives[2]);
        auto jac_det_dev = thrust::device_vector<float>(jac_det);
        // Get all the raw pointers ready for the Kernel
        float *j11_raw = thrust::raw_pointer_cast(j11_dev.data());
        float *j12_raw = thrust::raw_pointer_cast(j12_dev.data());
        float *j13_raw = thrust::raw_pointer_cast(j13_dev.data());
        float *j21_raw = thrust::raw_pointer_cast(j21_dev.data());
        float *j22_raw = thrust::raw_pointer_cast(j22_dev.data());
        float *j23_raw = thrust::raw_pointer_cast(j23_dev.data());
        float *j31_raw = thrust::raw_pointer_cast(j31_dev.data());
        float *j32_raw = thrust::raw_pointer_cast(j32_dev.data());
        float *j33_raw = thrust::raw_pointer_cast(j33_dev.data());
        float *jac_det_raw = thrust::raw_pointer_cast(jac_det_dev.data());
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            MMORF::kernel_jac_det,
            0,
            0);
        unsigned int blocks = static_cast<unsigned int>(
            std::ceil(float(jac_det_sz)/float(threads)));
        dim3 blocks_1d(blocks);
        // Call CUDA Kernel
        MMORF::kernel_jac_det<<<blocks_1d,threads>>>(
          // Input
          jac_det_sz,
          j11_raw,
          j12_raw,
          j13_raw,
          j21_raw,
          j22_raw,
          j23_raw,
          j31_raw,
          j32_raw,
          j33_raw,
          // Output
          jac_det_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Copy jac_det back to the host
        auto jac_det_host = thrust::host_vector<float>(jac_det_dev);
        thrust::copy(
            jac_det_host.begin(),
            jac_det_host.end(),
            jac_det.begin());
        return jac_det;
      }
      /// Given a set of positions, return the elements of the local Jacobian matrix at those
      /// positions
      std::vector<std::vector<float> > get_jacobian_elements(
          const std::vector<std::vector<float> >& positions) const
      {
        auto jacobian_dim = sub_warps_.size();
        auto samples_dim = positions[0].size();
        // Create an empty return vector of the correct size.
        // Note: Assumes that the number of subwarps matches the number of dimensions
        auto jacobian_elements = std::vector<std::vector<float> >(
            jacobian_dim * jacobian_dim,
            std::vector<float>(samples_dim));
        // Loop through all derivatives of all dimensions
        for (auto row = 0; row < jacobian_dim; ++row){
          auto gradients = sub_warps_[row].sample_gradient(positions);
#pragma omp parallel for
          for (auto col = 0; col < jacobian_dim; ++col){
            auto jacobian_entry = (row * jacobian_dim) + col;
            for (auto samp = 0; samp < samples_dim; ++samp){
              if (row != col){
                jacobian_elements[jacobian_entry][samp] = gradients[col][samp];
              }
              else{
                jacobian_elements[jacobian_entry][samp] = gradients[col][samp] + 1.0f;
              }
            }
          }
        }
        return jacobian_elements;
      }
      /// Scale warp field parameters by constant value
      void scale_parameters(
          const float scaling)
      {
        for (auto& warp : sub_warps_){
          auto params = warp.get_coefficients();
          for (auto& param : params){
            param *= scaling;
          }
          warp.set_coefficients(params);
        }
      }
      /// Get the number of parameters in warp
      /// \details pair.first  = number of subwarps
      ///          pair.second = number of parameters per subwarp
      std::pair<int,int> get_parameter_size() const
      {
        auto parameters_per_subwarp = 1;
        for (const auto& dim_parameters : sub_warps_[0].get_dimensions()){
          parameters_per_subwarp *= dim_parameters;
        }
        return std::make_pair(sub_warps_.size(),parameters_per_subwarp);
      }
      /// Get the volume representing the warp field in the specified direction
      MMORF::VolumeBSpline get_sub_warp(int warp_index) const
      {
        /// \todo Exception testing
        assert(warp_index >= 0 && warp_index < sub_warps_.size());
        return sub_warps_[warp_index];
      }
      /// Get the knot-spacing of the warp field in mm
      float get_knot_spacing() const
      {
        return knot_spacing_;
      }
      /// Reparameterise the warp field
      /// \details This increases the number of parameters used in the warp field by an
      ///          integer factor. In doing so this maintains a one-to-one relationship
      ///          between the warps at each level
      void reparameterise(const int scaling_factor)
      {
        for (auto& warp : sub_warps_){
          warp = warp.reparameterise(scaling_factor);
        }
        knot_spacing_ /= static_cast<float>(scaling_factor);
      }
      /// Crop the warp field
      /// \details The underlying samples will be reduced such that the given extents still
      ///          have the maximum number of splines with support at that position
      void crop(
          const std::pair<std::vector<float>, std::vector<float> > extents)
      {
        for (auto& warp : sub_warps_){
          warp = warp.crop(extents);
        }
      }
      /// Return the set of regularly positioned sample positions which all have the maximum
      /// number of splines with support in that region for a given sampling frequency (as in
      /// frequency per knot-spacing)
      std::vector<std::vector<float> > get_robust_sample_positions(
          const int sampling_frequency) const
      {
        // Get extents
        auto extents_full = sub_warps_[0].get_extents();
        // Get robust sample dimensions
        auto sample_dimensions = get_robust_sample_dimensions(sampling_frequency);
        // Get warp resolution
        auto resolution = sub_warps_[0].get_resolution();
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
      /// Apply both an affine transform and a warp to a given set of positions
      /// \details This is necessary when the affine transform is acting only on the original
      ///          sample positions, and independent of the warp field
      std::vector<std::vector<float> > apply_affine_then_warp(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        auto affine_positions = affine_transform_(positions, affine_mat);
        auto warp_offsets = sample_warp(positions);
        auto final_positions = add_positions_(affine_positions, warp_offsets);
        return final_positions;
      }
      /// Warp the given positions and then apply the affine transform.
      /// \details This is necessary in a number of situations, such as when finding the
      ///          derivative of a volume in a space reached via the given affine transform
      std::vector<std::vector<float> > apply_warp_then_affine(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        auto warped_positions = apply_warp(positions);
        auto affine_positions = affine_transform_(warped_positions, affine_mat);
        return affine_positions;
      }
    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Intitialise warp with (small) random values
      void random_initialise_warp_()
      {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0e-3f, 1.0e-3f);
        auto params = this->get_parameters();
        for (auto& param : params){
          std::generate(
              param.begin(),
              param.end(),
              [&dis, &gen](){return dis(gen);}
              );
        }
        this->set_parameters(params);
      }
      /// Affine transform a given set of positions
      std::vector<std::vector<float> > affine_transform_(
          const std::vector<std::vector<float> >& positions,
          const arma::fmat& affine_mat) const
      {
        /// \todo Exception testing
        assert(positions.size() == sub_warps_.size());
        auto positions_out = std::vector<std::vector<float> >(positions.size());
        // Apply affine transform
        // NB!!! Because armadillo is column major, we use a column vector for each vector of
        // positions for a particular dimension, and therefore the transpose of the affine_
        // matrix
        arma::fmat augmented_positions(positions[0].size(), positions.size() + 1);
        { // Scope limiting
          auto vec_dim = 0;
          for (const auto& vec : positions){
            augmented_positions.col(vec_dim) = arma::fvec(vec);
            ++vec_dim;
          }
          augmented_positions.col(vec_dim) = arma::fvec(
              positions[0].size(),
              arma::fill::ones);
          augmented_positions = augmented_positions * (affine_mat.t());
        } // Scope limiting
        { // Scope limiting
          auto vec_dim = 0;
          for (auto& vec : positions_out){
            vec = arma::conv_to<std::vector<float> >::from(augmented_positions.col(vec_dim));
            ++vec_dim;
          }
        } // Scope limiting
        return positions_out;
      }
      /// Add two positions together
      std::vector<std::vector<float> > add_positions_(
          std::vector<std::vector<float> > positions_first,
          std::vector<std::vector<float> > positions_second) const
      {
        // Check positions have the same dimensions
        assert(positions_first.size() == positions_second.size()
            && positions_first[0].size() == positions_second[0].size());
        auto positions_out = positions_first;
        for (auto i = 0; i < positions_out.size(); ++i){
          for (auto j = 0; j < positions_out[0].size(); ++j){
            positions_out[i][j] += positions_second[i][j];
          }
        }
        return positions_out;
      }
      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      std::vector<MMORF::VolumeBSpline> sub_warps_;
      float knot_spacing_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  WarpFieldBSpline::~WarpFieldBSpline() = default;
  /// Move ctor
  WarpFieldBSpline::WarpFieldBSpline(WarpFieldBSpline&& rhs) = default;
  /// Move assignment operator
  WarpFieldBSpline& WarpFieldBSpline::operator=(WarpFieldBSpline&& rhs) = default;
  /// Copy ctor
  WarpFieldBSpline::WarpFieldBSpline(const WarpFieldBSpline& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  WarpFieldBSpline& WarpFieldBSpline::operator=(const WarpFieldBSpline& rhs)
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
  WarpFieldBSpline::WarpFieldBSpline(
      const std::vector<int>& dimensions,
      const std::vector<float>& resolution,
      const std::vector<float>& origin)
    : pimpl_(MMORF::make_unique<Impl>(dimensions,resolution,origin))
  {}
  /// Construct empty field of given dimensions and resolution
  WarpFieldBSpline::WarpFieldBSpline(
      const std::pair<std::vector<float>,std::vector<float> >& extents,
      float knot_spacing)
    : pimpl_(MMORF::make_unique<Impl>(extents,knot_spacing))
  {}
  /// Sample the warp field at specified positions
  std::vector<std::vector<float> > WarpFieldBSpline::sample_warp(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->sample_warp(positions);
  }
  /// Given a set of positions, return the warped positions
  std::vector<std::vector<float> > WarpFieldBSpline::apply_warp(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->apply_warp(positions);
  }
  /// Get the parameters defining the warp
  /// \details Returns a vector of vectors containing the b-spline parameters for each
  ///          direction of the warp
  std::vector<std::vector<float> > WarpFieldBSpline::get_parameters() const
  {
    return pimpl_->get_parameters();
  }
  /// Set the parameters defining the warp
  /// \details Accepts a vector of vectors containing the b-spline parameters for each
  ///          direction of the warp
  void WarpFieldBSpline::set_parameters(
      const std::vector<std::vector<float> >& parameters)
  {
    pimpl_->set_parameters(parameters);
  }
  /// Get the dimensions of the subwarps
  std::vector<int> WarpFieldBSpline::get_dimensions() const
  {
    return pimpl_->get_dimensions();
  }
  /// Get the extents of the warp field
  std::pair<std::vector<float>,std::vector<float> > WarpFieldBSpline::get_extents() const
  {
    return pimpl_->get_extents();
  }
  /// Given a set of positions, return the Jacobian determinant at those positions
  std::vector<float> WarpFieldBSpline::jacobian_determinants(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->jacobian_determinants(positions);
  }
  /// Given a set of positions, return the elements of the local Jacobian matrix at those
  /// positions
  std::vector<std::vector<float> > WarpFieldBSpline::get_jacobian_elements(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->get_jacobian_elements(positions);
  }
  /// Scale warp field parameters by constant value
  void WarpFieldBSpline::scale_parameters(
      const float scaling)
  {
    pimpl_->scale_parameters(scaling);
  }
  /// Get the number of parameters in warp
  /// \details pair.first  = number of subwarps
  ///          pair.second = number of parameters per subwarp
  std::pair<int,int> WarpFieldBSpline::get_parameter_size() const
  {
    return pimpl_->get_parameter_size();
  }
  /// Get the volume representing the warp field in the specified direction
  MMORF::VolumeBSpline WarpFieldBSpline::get_sub_warp(int warp_index) const
  {
    return pimpl_->get_sub_warp(warp_index);
  }
  /// Get the knot-spacing of the warp field in mm
  float WarpFieldBSpline::get_knot_spacing() const
  {
    return pimpl_->get_knot_spacing();
  }
  /// Reparameterise the warp field
  /// \details This increases the number of parameters used in the warp field by an
  ///          integer factor. In doing so this maintains a one-to-one relationship
  ///          between the warps at each level
  void WarpFieldBSpline::reparameterise(const int scaling_factor) const
  {
    /// \todo Replace assert with exception
    assert (scaling_factor > 0);
    pimpl_->reparameterise(scaling_factor);
  }
  /// Crop the warp field
  /// \details The underlying samples will be reduced such that the given extents still
  ///          have the maximum number of splines with support at that position
  void WarpFieldBSpline::crop(
      const std::pair<std::vector<float>, std::vector<float> > extents) const
  {
    pimpl_->crop(extents);
  }
  /// Return the set of regularly positioned sample positions which all have the maximum
  /// number of splines with support in that region for a given sampling frequency (as in
  /// frequency per knot-spacing)
  std::vector<std::vector<float> > WarpFieldBSpline::get_robust_sample_positions(
      const int sampling_frequency) const
  {
    return pimpl_->get_robust_sample_positions(sampling_frequency);
  }
  /// Return the dimensions of the set of regularly spaced sample positions which all have
  /// the maximum number of splines with support in that region for a given sampling
  /// frequency (as in frequency per knot-spacing)
  std::vector<int> WarpFieldBSpline::get_robust_sample_dimensions(
      const int sampling_frequency) const
  {
    return pimpl_->get_robust_sample_dimensions(sampling_frequency);
  }
  /// Apply both an affine transform and a warp to a given set of positions
  /// \details This is necessary when the affine transform is acting only on the original
  ///          sample positions, and independent of the warp field
  std::vector<std::vector<float> > WarpFieldBSpline::apply_affine_then_warp(
      const std::vector<std::vector<float> >& positions,
      const arma::fmat& affine_mat) const
  {
    return pimpl_->apply_affine_then_warp(positions, affine_mat);
  }
  /// Warp the given positions and then apply the affine transform.
  /// \details This is necessary in a number of situations, such as when finding the
  ///          derivative of a volume in a space reached via the given affine transform
  std::vector<std::vector<float> > WarpFieldBSpline::apply_warp_then_affine(
      const std::vector<std::vector<float> >& positions,
      const arma::fmat& affine_mat) const
  {
    return pimpl_->apply_warp_then_affine(positions, affine_mat);
  }
} // MMORF
