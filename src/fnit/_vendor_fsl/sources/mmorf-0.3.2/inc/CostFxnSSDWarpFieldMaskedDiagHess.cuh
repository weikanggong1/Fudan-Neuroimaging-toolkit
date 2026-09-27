//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped. Additionally, a mask in the reference image domain is used to
///        constrain the area of interest during optimisaton.
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
///          Note that the Hessian calculation uses an approximation whereby it is represented
///          by a single main diagonal made up of the sum of absolute values of each
///          row/column (which is the same thing as H is symmetrical).
/// \author Frederik Lange
/// \date July 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_SSD_WARP_FIELD_MASKED_DIAG_HESS_CUH
#define COST_FXN_SSD_WARP_FIELD_MASKED_DIAG_HESS_CUH

#include "CostFxn.h"
#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnSSDWarpFieldMaskedDiagHess : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnSSDWarpFieldMaskedDiagHess();
      /// Move ctor
      CostFxnSSDWarpFieldMaskedDiagHess(CostFxnSSDWarpFieldMaskedDiagHess&& rhs);
      /// Move assignment operator
      CostFxnSSDWarpFieldMaskedDiagHess& operator=(CostFxnSSDWarpFieldMaskedDiagHess&& rhs);
      /// Copy ctor
      CostFxnSSDWarpFieldMaskedDiagHess(const CostFxnSSDWarpFieldMaskedDiagHess& rhs);
      /// Copy assignment operator
      CostFxnSSDWarpFieldMaskedDiagHess& operator=(
          const CostFxnSSDWarpFieldMaskedDiagHess& rhs);
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
      CostFxnSSDWarpFieldMaskedDiagHess(
          std::shared_ptr<MMORF::Volume>           vol_ref,
          std::shared_ptr<MMORF::Volume>           vol_mov,
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
  }; // CostFxnSSDWarpFieldMaskedDiagHess
} // MMORF
#endif // COST_FXN_SSD_WARP_FIELD_MASKED_DIAG_HESS_CUH
