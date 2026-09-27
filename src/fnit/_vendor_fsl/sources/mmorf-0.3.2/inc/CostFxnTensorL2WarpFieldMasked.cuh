//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the L2 norm of a masked tensor volume
/// \details Requires a B-spline parametrised warp field
/// \author Frederik Lange
/// \date August 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_TENSOR_L2_WARP_FIELD_MASKED_CUH
#define COST_FXN_TENSOR_L2_WARP_FIELD_MASKED_CUH

#include "CostFxn.h"
#include "VolumeTensor.cuh"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnTensorL2WarpFieldMasked : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnTensorL2WarpFieldMasked();
      /// Move ctor
      CostFxnTensorL2WarpFieldMasked(CostFxnTensorL2WarpFieldMasked&& rhs);
      /// Move assignment operator
      CostFxnTensorL2WarpFieldMasked& operator=(CostFxnTensorL2WarpFieldMasked&& rhs);
      /// Copy ctor
      CostFxnTensorL2WarpFieldMasked(const CostFxnTensorL2WarpFieldMasked& rhs);
      /// Copy assignment operator
      CostFxnTensorL2WarpFieldMasked& operator=(const CostFxnTensorL2WarpFieldMasked& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, and defining warp field
      /// characteristics
      /// \param vol_ref Reference (stationary) volume
      /// \param vol_mov Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param mask_ref 3D volume used to mask the reference volume
      /// \param knot_spacing Warp field B-spline knot spacing (mm) in reference space
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        warp field will be sampled every 2mm
      CostFxnTensorL2WarpFieldMasked(
          std::shared_ptr<MMORF::VolumeTensor>     vol_ref,
          std::shared_ptr<MMORF::VolumeTensor>     vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency
          );
      /// Get the current value of the parameters
      std::vector<std::vector<float> > get_parameters() const override;
      /// Set the current value of the parameters
      void set_parameters(
          const std::vector<std::vector<float> >& parameters) override;
      /// Get cost under current parameterisation
      float cost() const override;
      /// Get Jte under current parameterisation
      arma::fvec grad() const override;
      /// Get JtJ under current parameterisation
      MMORF::SparseDiagonalMatrixTiled hess() const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // CostFxnTensorL2WarpFieldMasked
} // MMORF
#endif // COST_FXN_TENSOR_L2_WARP_FIELD_MASKED_CUH
