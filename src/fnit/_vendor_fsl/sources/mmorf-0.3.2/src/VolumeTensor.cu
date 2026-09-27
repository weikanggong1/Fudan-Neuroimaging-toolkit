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
#include "MmorfMemory.h"
#include "VolumeTensor.cuh"
#include "Volume.h"
#include "VolumeBSpline.cuh"

#include "newimage/newimageall.h"

#include <armadillo>

#include <memory>
#include <vector>
#include <string>
#include <cmath>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class VolumeTensor::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Construct from NEWIMAGE::volume
      Impl(
          const NEWIMAGE::volume4D<float>&  vol,
          const bool                        apply_log)
        : tensor_(6)
      {
        // Split original volume
        auto split_vols = std::vector<NEWIMAGE::volume<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          split_vols[i] = split_4D_volume_(vol, i);
        }
        // Extract NEWIMAGE values
        auto tensor_vals = std::vector<std::vector<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_vals[i] = extract_newimage_data_(split_vols[i]);
        }
        // Rotate tensors
        auto rotation_mat = extract_newimage_rotation_(vol);
        tensor_vals = rotate_tensor_(tensor_vals, rotation_mat);
        // Log-transform tensors
        if (apply_log){
          tensor_vals = log_transform_tensor_(tensor_vals);
        }
        // Update NEWIMAGE values
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          update_newimage_data_(split_vols[i], tensor_vals[i]);
        }
        // Loop over elements of the tensor and make a 3D volume for each
        for (int i = 0; i < 6; ++i){
          tensor_[i] = std::make_shared<MMORF::VolumeBSpline>(split_vols[i]);
        }
      }
      /// Construct from nifti image file
      Impl(
          const std::string vol_name,
          const bool        apply_log)
        : tensor_(6)
      {
        // Read in 4D volume
        NEWIMAGE::volume4D<float> vol;
        NEWIMAGE::read_volume4D(vol, vol_name);
        // Split original volume
        auto split_vols = std::vector<NEWIMAGE::volume<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          split_vols[i] = split_4D_volume_(vol, i);
        }
        // Extract NEWIMAGE values
        auto tensor_vals = std::vector<std::vector<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_vals[i] = extract_newimage_data_(split_vols[i]);
        }
        // Rotate tensors
        auto rotation_mat = extract_newimage_rotation_(vol);
        tensor_vals = rotate_tensor_(tensor_vals, rotation_mat);
        // Log-transform tensors
        if (apply_log){
          tensor_vals = log_transform_tensor_(tensor_vals);
        }
        // Update NEWIMAGE values
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          update_newimage_data_(split_vols[i], tensor_vals[i]);
        }
        // Loop over elements of the tensor and make a 3D volume for each
        for (int i = 0; i < 6; ++i){
          tensor_[i] = std::make_shared<MMORF::VolumeBSpline>(split_vols[i]);
        }
      }
      /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
      Impl(
          const NEWIMAGE::volume4D<float>&  vol,
          const float                       smooth_fwhm_mm,
          const bool                        apply_log)
        : tensor_(6)
      {
        // Split original volume
        auto split_vols = std::vector<NEWIMAGE::volume<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          split_vols[i] = split_4D_volume_(vol, i);
        }
        // Extract NEWIMAGE values
        auto tensor_vals = std::vector<std::vector<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_vals[i] = extract_newimage_data_(split_vols[i]);
        }
        // Rotate tensors
        auto rotation_mat = extract_newimage_rotation_(vol);
        tensor_vals = rotate_tensor_(tensor_vals, rotation_mat);
        // Log-transform tensors
        if (apply_log){
          tensor_vals = log_transform_tensor_(tensor_vals);
        }
        // Update NEWIMAGE values
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          update_newimage_data_(split_vols[i], tensor_vals[i]);
        }
        // Loop over elements of the tensor and make a 3D volume for each
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_[i] = std::make_shared<MMORF::VolumeBSpline>(split_vols[i], smooth_fwhm_mm);
        }
      }
      /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
      Impl(
          const std::string vol_name,
          const float       smooth_fwhm_mm,
          const bool        apply_log)
        : tensor_(6)
      {
        // Read in 4D volume
        NEWIMAGE::volume4D<float> vol;
        NEWIMAGE::read_volume4D(vol, vol_name);
        // Split original volume
        auto split_vols = std::vector<NEWIMAGE::volume<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          split_vols[i] = split_4D_volume_(vol, i);
        }
        // Extract NEWIMAGE values
        auto tensor_vals = std::vector<std::vector<float> >(6);
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_vals[i] = extract_newimage_data_(split_vols[i]);
        }
        // Rotate tensors
        auto rotation_mat = extract_newimage_rotation_(vol);
        tensor_vals = rotate_tensor_(tensor_vals, rotation_mat);
        // Log-transform tensors
        if (apply_log){
          tensor_vals = log_transform_tensor_(tensor_vals);
        }
        // Update NEWIMAGE values
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          update_newimage_data_(split_vols[i], tensor_vals[i]);
        }
        // Loop over elements of the tensor and make a 3D volume for each
#pragma omp parallel for
        for (int i = 0; i < 6; ++i){
          tensor_[i] = std::make_shared<MMORF::VolumeBSpline>(split_vols[i], smooth_fwhm_mm);
        }
      }
      /// Sample tensor at specified coordinates
      std::vector<std::vector<float> > sample(
          const std::vector<std::vector<float> >& positions) const
      {
        auto samples = std::vector<std::vector<float> >(9);
        samples[0] =              tensor_[0]->sample(positions);
        samples[1] = samples[3] = tensor_[1]->sample(positions);
        samples[2] = samples[6] = tensor_[2]->sample(positions);
        samples[4] =              tensor_[3]->sample(positions);
        samples[5] = samples[7] = tensor_[4]->sample(positions);
        samples[8] =              tensor_[5]->sample(positions);
        return samples;
      }
      /// Sample the spatial derivative of the elements of the tensor at specified positions
      /// in all directions
      std::vector<std::vector<thrust::host_vector<float> > > sample_gradient(
          const std::vector<std::vector<float> >& positions) const
      {
        auto samples = std::vector<std::vector<thrust::host_vector<float> > >(9);
        samples[0] =              tensor_[0]->sample_gradient(positions);
        samples[1] = samples[3] = tensor_[1]->sample_gradient(positions);
        samples[2] = samples[6] = tensor_[2]->sample_gradient(positions);
        samples[4] =              tensor_[3]->sample_gradient(positions);
        samples[5] = samples[7] = tensor_[4]->sample_gradient(positions);
        samples[8] =              tensor_[5]->sample_gradient(positions);
        return samples;
      }
      /// Get the resolution of the underlying parameterisation
      std::vector<float> get_resolution() const
      {
        return tensor_[0]->get_resolution();
      }
    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Split Diffusion tensor into individual volumes. Note that we reverse the xy and xy
      /// directions as the "x" direction is Right-Left in FSL DTI images.
      const NEWIMAGE::volume<float> split_4D_volume_(
          const NEWIMAGE::volume4D<float>& vol,
          const int                      i)
      {
        NEWIMAGE::volume<float> vol_3D = vol[i];
        return vol_3D;
      }

      /// Extract data from NEWIMAGE::volume
      std::vector<float> extract_newimage_data_(const NEWIMAGE::volume<float>& vol)
      {
        auto newimage_data = std::vector<float>(vol.xsize()*vol.ysize()*vol.zsize());
        unsigned int i = 0;
        for (auto it = vol.fbegin();
            it != vol.fend();
            ++it, ++i)
        {
          newimage_data[i] = *it * 1e6f;
        }
        return newimage_data;
      }
      /// Update data in NEWIMAGE::volume
      void update_newimage_data_(
          NEWIMAGE::volume<float>&  vol,
          const std::vector<float>& vals)
      {
        auto i = 0;
        for (auto z = 0; z < vol.zsize(); ++z){
          for (auto y = 0; y < vol.ysize(); ++y){
            for (auto x = 0; x < vol.xsize(); ++x, ++i){
              vol(x, y, z) = vals[i];
            }
          }
        }
      }

      /// Log-transform the diffusion tensor
      std::vector<std::vector<float> > log_transform_tensor_(
          const std::vector<std::vector<float> >& tensor)
      {
        auto log_tensor = std::vector<std::vector<float> >(
            tensor.size(),
            std::vector<float>(tensor[0].size())
            );
        auto tensor_arma = std::vector<arma::mat33>(tensor[0].size());
        for (int i = 0; i < tensor_arma.size(); ++i){
          tensor_arma[i](0,0)                        = tensor[0][i];
          tensor_arma[i](0,1) = tensor_arma[i](1,0)  = tensor[1][i];
          tensor_arma[i](0,2) = tensor_arma[i](2,0)  = tensor[2][i];
          tensor_arma[i](1,1)                        = tensor[3][i];
          tensor_arma[i](1,2) = tensor_arma[i](2,1)  = tensor[4][i];
          tensor_arma[i](2,2)                        = tensor[5][i];
        }
        auto n_success = 0;
        auto n_failure = 0;
        auto failure_positions = std::vector<int>();
        auto average_tensor = arma::mat33();
        average_tensor.zeros();
        std::cout << average_tensor << std::endl;
        for (int i = 0; i < tensor_arma.size(); ++i){
       // for (int i = 0; i < 5000; ++i){
          bool log_success = arma::logmat_sympd(tensor_arma[i],tensor_arma[i]);
          if (log_success && !std::isnan(tensor_arma[i](0,0))){
            n_success++;
            average_tensor = average_tensor + tensor_arma[i];
            log_tensor[0][i] = tensor_arma[i](0,0);
            log_tensor[1][i] = tensor_arma[i](0,1);
            log_tensor[2][i] = tensor_arma[i](0,2);
            log_tensor[3][i] = tensor_arma[i](1,1);
            log_tensor[4][i] = tensor_arma[i](1,2);
            log_tensor[5][i] = tensor_arma[i](2,2);
          }
          else{
            n_failure++;
            failure_positions.push_back(i);
            log_tensor[0][i] = 0;
            log_tensor[1][i] = 0;
            log_tensor[2][i] = 0;
            log_tensor[3][i] = 0;
            log_tensor[4][i] = 0;
            log_tensor[5][i] = 0;
            //log_tensor[0][i] = std::nanf("");
            //log_tensor[1][i] = std::nanf("");
            //log_tensor[2][i] = std::nanf("");
            //log_tensor[3][i] = std::nanf("");
            //log_tensor[4][i] = std::nanf("");
            //log_tensor[5][i] = std::nanf("");
          }
        }
        average_tensor = average_tensor / n_success;
        //for (const auto& i : failure_positions){
        //  log_tensor[0][i] = average_tensor(0,0);
        //  log_tensor[1][i] = average_tensor(0,1);
        //  log_tensor[2][i] = average_tensor(0,2);
        //  log_tensor[3][i] = average_tensor(1,1);
        //  log_tensor[4][i] = average_tensor(1,2);
        //  log_tensor[5][i] = average_tensor(2,2);
        //}
        std::cout << "Successes: " << n_success << std::endl;
        std::cout << "Failures:  " << n_failure << std::endl;
        std::cout << tensor_arma[10] << std::endl;
        std::cout << average_tensor << std::endl;
        return log_tensor;
      }

      /// Extract rotation from NEWIMAGE::volume
      arma::fmat33 extract_newimage_rotation_(const NEWIMAGE::volume4D<float>& vol)
      {
        auto vox_to_mm_newimage = vol.newimagevox2mm_mat();
        //auto vox_to_mm_newimage = vol.sampling_mat();
        auto vox_to_mm_arma = arma::fmat33();
        for (auto j = 0; j < 3; ++j){
          for (auto i = 0; i < 3; ++i){
            vox_to_mm_arma(i,j) = vox_to_mm_newimage.element(i,j);
          }
        }
        auto rotation_mat = arma::fmat33();
        rotation_mat = arma::sqrtmat_sympd(
            (vox_to_mm_arma * (vox_to_mm_arma.t())).i()) * vox_to_mm_arma;
        return rotation_mat;
      }

      /// Apply rotation matrix to tensor data
      std::vector<std::vector<float> > rotate_tensor_(
          const std::vector<std::vector<float> >& tensor,
          const arma::fmat33&                     rotation_mat)
      {
        auto tensor_rotated = std::vector<std::vector<float> >(
            tensor.size(),
            std::vector<float>(tensor[0].size())
            );
        auto tensor_arma = std::vector<arma::fmat33>(tensor[0].size());
        for (int i = 0; i < tensor_arma.size(); ++i){
          // Convert vec to arma
          tensor_arma[i](0,0)                        = tensor[0][i];
          tensor_arma[i](0,1) = tensor_arma[i](1,0)  = tensor[1][i];
          tensor_arma[i](0,2) = tensor_arma[i](2,0)  = tensor[2][i];
          tensor_arma[i](1,1)                        = tensor[3][i];
          tensor_arma[i](1,2) = tensor_arma[i](2,1)  = tensor[4][i];
          tensor_arma[i](2,2)                        = tensor[5][i];
          // Rotate tensor
          tensor_arma[i] = rotation_mat * tensor_arma[i] * (rotation_mat.t());
          // Convert arma to vec
          tensor_rotated[0][i] = tensor_arma[i](0,0);
          tensor_rotated[1][i] = tensor_arma[i](0,1);
          tensor_rotated[2][i] = tensor_arma[i](0,2);
          tensor_rotated[3][i] = tensor_arma[i](1,1);
          tensor_rotated[4][i] = tensor_arma[i](1,2);
          tensor_rotated[5][i] = tensor_arma[i](2,2);
        }
        return tensor_rotated;
      }

      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      std::vector<std::shared_ptr<MMORF::VolumeBSpline> > tensor_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  VolumeTensor::~VolumeTensor() = default;
  /// Move ctor
  VolumeTensor::VolumeTensor(VolumeTensor&& rhs) = default;
  /// Move assignment operator
  VolumeTensor& VolumeTensor::operator=(VolumeTensor&& rhs) = default;
  /// Copy ctor
  VolumeTensor::VolumeTensor(const VolumeTensor& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  VolumeTensor& VolumeTensor::operator=(const VolumeTensor& rhs)
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
  /// Construct from NEWIMAGE::volume
  VolumeTensor::VolumeTensor(
      const NEWIMAGE::volume4D<float>&  vol,
      const bool                        apply_log)
    : pimpl_(MMORF::make_unique<Impl>(vol, apply_log))
  {}
  /// Construct from nifti image file
  VolumeTensor::VolumeTensor(
      const std::string vol_name,
      const bool        apply_log)
    : pimpl_(MMORF::make_unique<Impl>(vol_name, apply_log))
  {}
  /// Construct from NEWIMAGE::volume and smooth with Gaussian (defined by FWHM mm)
  VolumeTensor::VolumeTensor(
      const NEWIMAGE::volume4D<float>&  vol,
      const float                       smooth_fwhm_mm,
      const bool                        apply_log)
    : pimpl_(MMORF::make_unique<Impl>(vol, smooth_fwhm_mm, apply_log))
  {}
  /// Construct from nifti image file and smooth with Gaussian (defined by FWHM mm)
  VolumeTensor::VolumeTensor(
      const std::string vol_name,
      const float       smooth_fwhm_mm,
      const bool        apply_log)
    : pimpl_(MMORF::make_unique<Impl>(vol_name, smooth_fwhm_mm, apply_log))
  {}
  /// Sample tensor at specified coordinates
  std::vector<std::vector<float> > VolumeTensor::sample(
          const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->sample(positions);
  }
  /// Sample the spatial derivative of the elements of the tensor at specified positions
  /// in all directions
  std::vector<std::vector<thrust::host_vector<float> > > VolumeTensor::sample_gradient(
      const std::vector<std::vector<float> >& positions) const
  {
    return pimpl_->sample_gradient(positions);
  }
  /// Get the resolution of the underlying parameterisation
  std::vector<float> VolumeTensor::get_resolution() const
  {
    return pimpl_->get_resolution();
  }
} // MMORF
