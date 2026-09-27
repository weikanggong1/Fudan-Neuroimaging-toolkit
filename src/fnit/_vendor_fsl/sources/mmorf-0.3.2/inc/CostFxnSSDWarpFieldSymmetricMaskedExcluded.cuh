//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped. Additionally, a mask in the ref & mov image domain is used to
///        constrain the area of interest during optimisaton. This is now extended to include
///        a greedy approximation to symmetrisation of the problem by multiplying the cost
///        by (1 + |J|).
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
/// \author Frederik Lange
/// \date March 2021
/// \copyright Copyright (C) 2021 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_SSD_WARP_FIELD_SYMMETRIC_MASKED_EXCLUDED_CUH
#define COST_FXN_SSD_WARP_FIELD_SYMMETRIC_MASKED_EXCLUDED_CUH

#include "CostFxn.h"
#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "BiasFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnSSDWarpFieldSymmetricMaskedExcluded : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnSSDWarpFieldSymmetricMaskedExcluded();
      /// Move ctor
      CostFxnSSDWarpFieldSymmetricMaskedExcluded(CostFxnSSDWarpFieldSymmetricMaskedExcluded&& rhs);
      /// Move assignment operator
      CostFxnSSDWarpFieldSymmetricMaskedExcluded& operator=(CostFxnSSDWarpFieldSymmetricMaskedExcluded&& rhs);
      /// Copy ctor
      CostFxnSSDWarpFieldSymmetricMaskedExcluded(const CostFxnSSDWarpFieldSymmetricMaskedExcluded& rhs);
      /// Copy assignment operator
      CostFxnSSDWarpFieldSymmetricMaskedExcluded& operator=(const CostFxnSSDWarpFieldSymmetricMaskedExcluded& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, warp and (optional) bias field
      /// \param vol_ref Reference (stationary) volume
      /// \param vol_mov Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param mask_ref 3D volume used to mask the reference volume
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param warp_field Shared pointer to fully formed warp field
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///             warp field will be sampled every 2mm
      /// \param bias_field Shared pointer to fully formed bias field
      CostFxnSSDWarpFieldSymmetricMaskedExcluded(
          std::shared_ptr<MMORF::Volume>           vol_ref,
          std::shared_ptr<MMORF::Volume>           vol_mov,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::Volume>           mask_ref,
          std::shared_ptr<MMORF::Volume>           mask_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int                                      sampling_frequency,
          std::shared_ptr<MMORF::BiasFieldBSpline> bias_field = nullptr,
          const bool                               debug = false
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
  }; // CostFxnSSDWarpFieldSymmetricMaskedExcluded
} // MMORF
#endif // COST_FXN_SSD_WARP_FIELD_SYMMETRIC_MASKED_EXCLUDED_CUH
