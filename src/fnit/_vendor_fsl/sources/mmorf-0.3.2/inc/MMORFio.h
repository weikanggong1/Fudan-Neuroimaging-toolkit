//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Provides input/output functionality for MMORF classes
/// \details Additional utilities when, for example, one wishes to save a MMORF::Volume object
///          to disk as a NIFTI file
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MMORF_IO_H
#define MMORF_IO_H

#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <string>
#include <cmath>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Pass in a warp field and save the entire volume (which may be outside the extents of
  /// the original volume, as warp fields are always truly square in x, y, and z
  /// \param warp WarpField in world space
  /// \param sampling_res Desired resolution of the sampled warp (in mm)
  /// \param filename Path to save sampled warp to
  inline void save_full_warp_as_nifti(
      const MMORF::WarpFieldBSpline&  warp,
      const float                     sampling_res,
      std::string                     filename)
  {
    auto knot_spacing = warp.get_knot_spacing();
    auto sampling_freq = static_cast<int>(
        std::ceil(knot_spacing/sampling_res));
    auto sample_dims = warp.get_robust_sample_dimensions(sampling_freq);
    auto sample_pos = warp.get_robust_sample_positions(sampling_freq);
    auto samples = warp.sample_warp(sample_pos);
    auto samples_concat = std::accumulate(
            samples.begin(), samples.end(),
            std::vector<float>(),
            [](std::vector<float> a, std::vector<float> b)
            {
              a.insert(a.end(), b.begin(), b.end());
              return a;
            });
    NEWIMAGE::volume4D<float> tmp_vol_4D;
    tmp_vol_4D.reinitialize(
        sample_dims[0],
        sample_dims[1],
        sample_dims[2],
        3,
        samples_concat.data(),
        false);
    NEWIMAGE::write_volume4D(tmp_vol_4D, filename);
  }
  /// Pass in a warp field, volume and affine and save the entire volume (which may be
  /// outside the original volume, as warp fields are always truly square in x, y, and z
  /// \param warp WarpField in world space
  /// \param sampling_res Desired resolution of the sampled warp (in mm)
  /// \param filename Path to save sampled warp to
  inline void save_full_vol_as_nifti(
      const MMORF::WarpFieldBSpline&  warp,
      const MMORF::Volume&            vol,
      const arma::fmat                aff,
      const float                     sampling_res,
      std::string                     filename)
  {
    auto knot_spacing = warp.get_knot_spacing();
    auto sampling_freq = static_cast<int>(
        std::ceil(knot_spacing/sampling_res));
    auto sample_dims = warp.get_robust_sample_dimensions(sampling_freq);
    auto sample_pos = warp.apply_warp_then_affine(
        warp.get_robust_sample_positions(sampling_freq),
        aff);
    auto samples = vol.sample(sample_pos);
    NEWIMAGE::volume<float> tmp_vol;
    tmp_vol.reinitialize(
        sample_dims[0],
        sample_dims[1],
        sample_dims[2],
        1,
        samples.data(),
        false);
    NEWIMAGE::write_volume(tmp_vol, filename);
  }
  /// Sample a MMORF::Volume at particular points and save the result as a NIFTI file
  /// \details Uses a NEWIMAGE::volume as the underlying class to perform the storage
  template <typename T>
  inline void save_as_nifti(
      const MMORF::Volume&                mm_volume,
      const std::vector<std::vector<T> >& sample_points,
      const std::vector<int>&             sample_dimensions,
      const NEWIMAGE::volume<T>&          ni_volume,
      std::string                         filename)
  {
    // Ensure dimensions match
    assert(sample_dimensions.size() == 3);
    assert(
        sample_dimensions[0]*sample_dimensions[1]*sample_dimensions[2]
        == sample_points[0].size());
    // Sample volume
    auto samples = mm_volume.sample(sample_points);
    // Setup temporary volume
    auto temp_vol = NEWIMAGE::volume<T>();
    temp_vol.reinitialize(
        sample_dimensions[0],
        sample_dimensions[1],
        sample_dimensions[2],
        1,
        samples.data(),
        false);
    temp_vol.copyproperties(ni_volume);
    // Save volume
    NEWIMAGE::write_volume(temp_vol,filename);
  }

  /// Sample derivative of a MMORF::Volume at particular points and save the result as a
  /// NIFTI file
  /// \details Uses a NEWIMAGE::volume as the underlying class to perform the storage
  template <typename T>
  inline void save_as_nifti(
      const MMORF::Volume&                mm_volume,
      const std::vector<std::vector<T> >& sample_points,
      const std::vector<int>&             sample_dimensions,
      const int                           deriv_direction,
      const NEWIMAGE::volume<T>&          ni_volume,
      std::string filename)
  {
    // Ensure dimensions match
    assert(sample_dimensions.size() == 3);
    assert(deriv_direction < 3);
    assert(
        sample_dimensions[0]*sample_dimensions[1]*sample_dimensions[2]
        == sample_points[0].size());
    // Sample volume
    auto samples = mm_volume.sample_gradient (sample_points);
    // Setup temporary volume
    auto temp_vol = NEWIMAGE::volume<T>();
    temp_vol.reinitialize(
        sample_dimensions[0],
        sample_dimensions[1],
        sample_dimensions[2],
        samples[deriv_direction].data(),
        false);
    temp_vol.copyproperties(ni_volume);
    // Save volume
    NEWIMAGE::write_volume(temp_vol,filename);
  }

  /// Save sampled values as a NIFTI file
  /// \details Uses a NEWIMAGE::volume as the underlying class to perform the storage
  template <typename T>
  inline void save_as_nifti(
      std::vector<T>&             samples,
      const std::vector<int>&     sample_dimensions,
      const NEWIMAGE::volume<T>&  ni_volume,
      std::string                 filename)
  {
    // Ensure dimensions match
    assert(sample_dimensions.size() == 3);
    assert(
        sample_dimensions[0]*sample_dimensions[1]*sample_dimensions[2]
        == samples.size());
    // Setup temporary volume
    auto temp_vol = NEWIMAGE::volume<T>();
    temp_vol.reinitialize(
        sample_dimensions[0],
        sample_dimensions[1],
        sample_dimensions[2],
        1,
        samples.data(),
        false);
    temp_vol.copyproperties(ni_volume);
    // Save volume
    NEWIMAGE::write_volume(temp_vol,filename);
  }

  /// Save multiple sampled values as a 4D NIFTI file
  template <typename T>
  inline void save_as_nifti(
      std::vector<std::vector<T> >& samples,
      const std::vector<int>&       sample_dimensions,
      const NEWIMAGE::volume<T>&    ni_volume,
      std::string                   filename)
  {
    // Ensure dimensions match
    assert(sample_dimensions.size() == 3);
    assert(
        sample_dimensions[0]*sample_dimensions[1]*sample_dimensions[2]
        == samples[0].size());
    // Setup temporary volume
    auto temp_vol_4D = NEWIMAGE::volume4D<T>();
    temp_vol_4D.copyproperties(ni_volume);
    for (auto i = 0; i < samples.size(); ++i){
      auto temp_vol = NEWIMAGE::volume<T>();
      temp_vol.reinitialize(
          sample_dimensions[0],
          sample_dimensions[1],
          sample_dimensions[2],
          1,
          samples[i].data(),
          false);
      temp_vol.copyproperties(ni_volume);
      temp_vol_4D.addvolume(temp_vol);
    }
    // Save volume
    NEWIMAGE::write_volume4D(temp_vol_4D,filename);
  }
} // MMORF
#endif // MMORF_IO_H
