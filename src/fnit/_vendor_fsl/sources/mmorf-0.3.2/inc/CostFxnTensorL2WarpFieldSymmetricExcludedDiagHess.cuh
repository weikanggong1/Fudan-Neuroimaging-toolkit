//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the L2 norm of a masked tensor volume
/// \details Requires a B-spline parametrised warp field, Note that the Hessian calculation
///          uses a Majorise-Minimise diagonalisation strategy.
///          This is now extended to include a greedy approximation to symmetrisation of the
///          problem by multiplying the cost by (1 + |J|).
/// \author Frederik Lange
/// \date March 2021
/// \copyright Copyright (C) 2021 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_TENSOR_L2_WARP_FIELD_SYMMETRIC_EXCLUDED_DIAG_HESS_CUH
#define COST_FXN_TENSOR_L2_WARP_FIELD_SYMMETRIC_EXCLUDED_DIAG_HESS_CUH

#include "CostFxn.h"
#include "VolumeTensor.cuh"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess();
      /// Move ctor
      CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess(
          CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess&& rhs);
      /// Move assignment operator
      CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess& operator=(
          CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess&& rhs);
      /// Copy ctor
      CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess(
          const CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess& rhs);
      /// Copy assignment operator
      CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess& operator=(
          const CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess& rhs);
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
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param knot_spacing Warp field B-spline knot spacing (mm) in reference space
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        warp field will be sampled every 2mm
      CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess(
          std::shared_ptr<MMORF::VolumeTensor>     vol_ref,
          std::shared_ptr<MMORF::VolumeTensor>     vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_mov,
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
  }; // CostFxnTensorL2WarpFieldSymmetricExcludedDiagHess
} // MMORF
#endif // COST_FXN_TENSOR_L2_WARP_FIELD_SYMMETRIC_EXCLUDED_DIAG_HESS_CUH
