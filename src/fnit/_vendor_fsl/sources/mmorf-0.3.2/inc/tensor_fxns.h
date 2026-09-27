//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Functions which operate on tensor volumes
/// \details Facilitate averaging, rotating, interpolating etc. of DTI volumes
/// \author Frederik Lange
/// \date January 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "newimage/newimageall.h"
#include <armadillo>
#include <vector>
#include <string>

namespace MMORF
{
  /// Return the average of an arbitrary number of tensor volumes
  NEWIMAGE::volume4D<double> tensor_average(
      const std::vector<std::string>& vols_name);

  /// Extract data from NEWIMAGE::volume
  std::vector<std::vector<double> > sample_newimage_volume4D(
      NEWIMAGE::volume4D<double>& vol);

  /// Convert raw DTI vector data into armadillo matrix. Note this assumes the FSL tensor
  /// convention has been followed
  std::vector<arma::mat33> dti_vector_to_arma(
      const std::vector<std::vector<double> >& dti_vec);

  /// Convert armadillo matrix into raw DTI vector data. Note this assumes the FSL tensor
  /// convention has been followed
  std::vector<std::vector<double> > dti_arma_to_vector(
      const std::vector<arma::mat33>& dti_arma);

  /// Calculate the matrix logarithm of the diffusion tensors. If the transform fails, the
  /// matrix is set to infinity.
  std::vector<arma::mat33> log_transform_tensors(
      const std::vector<arma::mat33>& tensors);

  /// Calculate the matrix exponential of the diffusion tensors. If the transform fails, the
  /// matrix is set to infinity.
  std::vector<arma::mat33> exp_transform_tensors(
      const std::vector<arma::mat33>& tensors);
} // namespace MMORF
