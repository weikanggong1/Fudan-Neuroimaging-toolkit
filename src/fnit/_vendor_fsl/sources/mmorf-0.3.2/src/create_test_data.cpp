
#include "MMORFio.h"
#include "WarpFieldBSpline.cuh"
#include "VolumeBSpline.cuh"

#include "newimage/newimageall.h"

#include <armadillo>

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>

std::vector<std::vector<float> > generate_sample_points(
    const std::pair<std::vector<float>, std::vector<float> >& extents,
    float resolution);
std::vector<std::vector<float> > generate_warp_parameters(
    const MMORF::WarpFieldBSpline& warp_field);
void save_parameters(const std::vector<std::vector<float> >& parameters);

int main()
{
  // Read in data and make preliminary objects
  auto filename_ref = std::string("./data/MNI152_T1_2mm");
  auto filename_mov = std::string("./data/MNI152_T1_2mm_warped");
  auto filename_warp = std::string("./data/actual_warp_field");
  auto vol_ref_ni = NEWIMAGE::volume<float>();
  NEWIMAGE::read_volume(vol_ref_ni,filename_ref);
  auto vol_mov_ni = NEWIMAGE::volume<float>();
  auto vol_ref_mo = MMORF::VolumeBSpline(filename_ref);
  auto knot_spacing = 20.0f;
  auto sampling_frequency = 10;
  auto affine_mat = arma::Mat<float>{
      {1,0,0,0},
      {0,1,0,0},
      {0,0,1,0},
      {0,0,0,1}};
  auto warp_field = MMORF::WarpFieldBSpline(vol_ref_mo.get_extents(),knot_spacing,affine_mat);
  auto sample_points = generate_sample_points(vol_ref_mo.get_extents(),2);
  auto warp_parameters = generate_warp_parameters(warp_field);

  // Generate warped samples
  warp_field.set_parameters(warp_parameters);
  auto sample_points_warped = warp_field.apply_warp(sample_points);

  // Save warp parameters
  save_parameters(warp_parameters);

  // Save warp field
  std::vector<std::vector<float> > warp_samples = warp_field.sample_warp(sample_points);
  MMORF::save_as_nifti(
      warp_samples,
      std::vector<int>{
          vol_ref_ni.xsize(),
          vol_ref_ni.ysize(),
          vol_ref_ni.zsize()},
      vol_ref_ni,
      filename_warp);

  // Save warped volume
  MMORF::save_as_nifti(
      vol_ref_mo,
      sample_points_warped,
      std::vector<int>{
          vol_ref_ni.xsize(),
          vol_ref_ni.ysize(),
          vol_ref_ni.zsize()},
      vol_ref_ni,
      filename_mov);

  // Create MMORF::Volume from saved volume
  NEWIMAGE::read_volume(vol_mov_ni,filename_mov);
  auto vol_mov_mo = MMORF::VolumeBSpline(filename_mov);

  // Save derivarives of original volume
  for (auto i = 0; i < 3; ++i){
    MMORF::save_as_nifti(
        vol_ref_mo,
        sample_points,
        std::vector<int>{
            vol_ref_ni.xsize(),
            vol_ref_ni.ysize(),
            vol_ref_ni.zsize()},
        i,
        vol_ref_ni,
        filename_ref + "_deriv_" + std::to_string(i));
  }

  // Save derivatives of warped volume
  for (auto i = 0; i < 3; ++i){
    MMORF::save_as_nifti(
        vol_ref_mo,
        sample_points_warped,
        std::vector<int>{
            vol_ref_ni.xsize(),
            vol_ref_ni.ysize(),
            vol_ref_ni.zsize()},
        i,
        vol_ref_ni,
        filename_mov + "_deriv_" + std::to_string(i));
  }

  return 0;
}

std::vector<std::vector<float> > generate_sample_points(
    const std::pair<std::vector<float>, std::vector<float> >& extents,
    float resolution)
{
  auto points = std::vector<std::vector<float> >(3);
  // Calculate sample points in reference space. NOTE: this is explicitly implemented
  // for the 3D case and should be revised. Alsom the use of "push_back" is convenient,
  // but potentially slow.
  /// \todo Revise explicit 3D implementation and use of push_back
  for (auto k = extents.first[2];
      k <= extents.second[2];
      k += resolution){
    for (auto j = extents.first[1];
        j <= extents.second[1];
        j += resolution){
      for (auto i = extents.second[0]; // NB Backwards because of MNI storage of x dimension
          i >= extents.first[0];
          i -= resolution){
        points[0].push_back(i);
        points[1].push_back(j);
        points[2].push_back(k);
      }
    }
  }
  return points;
}

std::vector<std::vector<float> > generate_warp_parameters(
    const MMORF::WarpFieldBSpline& warp_field)
{
  auto warp_params = std::vector<std::vector<float> >();
  std::srand(10);
  for (auto i = 0; i < 3; ++i){
    auto params = std::vector<float>(warp_field.get_parameter_size().second);
    std::generate(params.begin(),params.end(),std::rand);
    for (auto& param : params){
      param = 20*(param/RAND_MAX - 0.5);
    }
    warp_params.push_back(params);
  }
  return warp_params;
}

void save_parameters(const std::vector<std::vector<float> >& parameters)
{
  auto all_params = arma::fmat(parameters[0].size(),parameters.size());
  auto col = 0;
  for (const auto& params : parameters){
    all_params.col(col) = arma::fvec(params);
    ++col;
  }
  all_params.save("data/actual_warp_params",arma::raw_ascii);
}
