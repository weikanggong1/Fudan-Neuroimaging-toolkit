//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Functions which operate on tensor volumes
/// \details Facilitate averaging, rotating, interpolating etc. of DTI volumes
/// \author Frederik Lange
/// \date January 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "tensor_fxns.h"
#include "newimage/newimageall.h"
#include <armadillo>
#include <vector>
#include <string>

/// Return the average of an arbitrary number of tensor volumes
NEWIMAGE::volume4D<double> MMORF::tensor_average(
    const std::vector<std::string>& vols_name)
{
  auto average_tensor = NEWIMAGE::volume4D<double> ();
  NEWIMAGE::read_volume4D(average_tensor, vols_name[0]);
  auto n_vols = vols_name.size();
  auto n_voxels = average_tensor.xsize() * average_tensor.ysize() * average_tensor.zsize();
  auto data_average_arma = std::vector<arma::mat33>(
      n_voxels,
      arma::zeros<arma::mat>(3,3));
  auto inf_vols_count = std::vector<int>(n_voxels,0);
  for (auto i = 0; i < n_vols; ++i){
    auto vol_i = NEWIMAGE::volume4D<double>();
    NEWIMAGE::read_volume4D(vol_i, vols_name[i]);
    auto data_i = sample_newimage_volume4D(vol_i);
    auto data_arma = dti_vector_to_arma(data_i);
    auto data_arma_log = log_transform_tensors(data_arma);
    for (auto j = 0; j < n_voxels; ++j){
      if (data_arma_log[j].is_finite()){
        data_average_arma[j] += data_arma_log[j];
      }
      else ++inf_vols_count[j];
    }
  }
  for (auto j = 0; j < n_voxels; ++j){
    if (inf_vols_count[j] >= n_vols){
      data_average_arma[j].zeros();
    }
    else{
      data_average_arma[j] /= (n_vols - inf_vols_count[j]);
      bool exp_success = arma::expmat_sym(
          data_average_arma[j],
          data_average_arma[j]);
      if (!exp_success){
        exp_success = arma::expmat_sym(
            data_average_arma[j],
            arma::symmatu(data_average_arma[j]));
        if (!exp_success){
          data_average_arma[j].zeros();
        }
      }
      if (inf_vols_count[j] > 0){
        data_average_arma[j] *= 
          (static_cast<float>(n_vols - inf_vols_count[j]) / static_cast<float>(n_vols));
      }
    }
  }
  auto data_average = dti_arma_to_vector(data_average_arma);
  auto idx = 0;
  for (int z = 0; z < average_tensor.zsize(); ++z){
    for (int y = 0; y < average_tensor.ysize(); ++y){
      for (int x = 0; x < average_tensor.xsize(); ++x){
        average_tensor(x,y,z,0) = data_average[0][idx];
        average_tensor(x,y,z,1) = data_average[1][idx];
        average_tensor(x,y,z,2) = data_average[2][idx];
        average_tensor(x,y,z,3) = data_average[3][idx];
        average_tensor(x,y,z,4) = data_average[4][idx];
        average_tensor(x,y,z,5) = data_average[5][idx];
        ++idx;
      }
    }
  }
  return average_tensor;
}

/// Extract data from NEWIMAGE::volume
std::vector<std::vector<double> > MMORF::sample_newimage_volume4D(
    NEWIMAGE::volume4D<double>& vol)
{
  auto newimage_data = std::vector<std::vector<double> >(vol.tsize());
  for (int i = 0; i < vol.tsize(); ++i){
    newimage_data[i].reserve(vol.xsize()*vol.ysize()*vol.zsize());
    auto tmp_vol = vol[i];
    for (NEWIMAGE::volume<double>::fast_const_iterator it = tmp_vol.fbegin();
         it != tmp_vol.fend();
         ++it){
      newimage_data[i].push_back(*it);
    }
  }
  return newimage_data;
}

/// Convert raw DTI vector data into armadillo matrix. Note this assumes the FSL tensor
/// convention has been followed
std::vector<arma::mat33> MMORF::dti_vector_to_arma(
    const std::vector<std::vector<double> >& dti_vec)
{
  auto arma_vec = std::vector<arma::mat33>(dti_vec[0].size());
  for (int i = 0; i < dti_vec[0].size(); ++i){
    arma_vec[i](0,0)                     = dti_vec[0][i];
    arma_vec[i](0,1) = arma_vec[i](1,0)  = dti_vec[1][i];
    arma_vec[i](0,2) = arma_vec[i](2,0)  = dti_vec[2][i];
    arma_vec[i](1,1)                     = dti_vec[3][i];
    arma_vec[i](1,2) = arma_vec[i](2,1)  = dti_vec[4][i];
    arma_vec[i](2,2)                     = dti_vec[5][i];
  }
  return arma_vec;
}

/// Convert armadillo matrix into raw DTI vector data. Note this assumes the FSL tensor
/// convention has been followed
std::vector<std::vector<double> > MMORF::dti_arma_to_vector(
    const std::vector<arma::mat33>& dti_arma)
{
  auto dti_vec = std::vector<std::vector<double> >(
      6,
      std::vector<double>(dti_arma.size()));
  for (int i = 0; i < dti_arma.size(); ++i){
    dti_vec[0][i] = dti_arma[i](0,0);
    dti_vec[1][i] = dti_arma[i](0,1);
    dti_vec[2][i] = dti_arma[i](0,2);
    dti_vec[3][i] = dti_arma[i](1,1);
    dti_vec[4][i] = dti_arma[i](1,2);
    dti_vec[5][i] = dti_arma[i](2,2);
  }
  return dti_vec;
}

/// Calculate the matrix logarithm of the diffusion tensors. If the transform fails, the
/// matrix is set to infinity.
std::vector<arma::mat33> MMORF::log_transform_tensors(
    const std::vector<arma::mat33>& tensors)
{
  auto log_tensors = tensors;
  for (int i = 0; i < tensors.size(); ++i){
    bool log_success = arma::logmat_sympd(log_tensors[i], tensors[i]);
    if (!log_success){
      log_tensors[i].fill(arma::datum::inf);
    }
  }
  return log_tensors;
}

/// Calculate the matrix exponential of the diffusion tensors. If the transform fails, the
/// matrix is set to infinity.
std::vector<arma::mat33> MMORF::exp_transform_tensors(
    const std::vector<arma::mat33>& tensors)
{
  auto exp_tensors = tensors;
  for (int i = 0; i < tensors.size(); ++i){
    bool exp_success = arma::expmat_sym(exp_tensors[i], tensors[i]);
    if (!exp_success){
      exp_success = arma::expmat_sym(exp_tensors[i], arma::symmatu(tensors[i]));
      if (!exp_success){
        exp_tensors[i].zeros();
      }
    }
  }
  return exp_tensors;
}
