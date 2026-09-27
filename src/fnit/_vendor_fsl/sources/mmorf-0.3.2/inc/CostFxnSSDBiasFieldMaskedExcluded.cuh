//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes in the
///        presence of a multiplicative bias field.
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume. The bias correction is performed on the reference image in order to
///          simplify the calculation of spatial intensity gradients in the moving image at
///          higher levels of the optimisation process.
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_SSD_BIAS_FIELD_MASKED_EXCLUDED_CUH
#define COST_FXN_SSD_BIAS_FIELD_MASKED_EXCLUDED_CUH

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
  class CostFxnSSDBiasFieldMaskedExcluded : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnSSDBiasFieldMaskedExcluded();
      /// Move ctor
      CostFxnSSDBiasFieldMaskedExcluded(CostFxnSSDBiasFieldMaskedExcluded&& rhs);
      /// Move assignment operator
      CostFxnSSDBiasFieldMaskedExcluded& operator=(CostFxnSSDBiasFieldMaskedExcluded&& rhs);
      /// Copy ctor
      CostFxnSSDBiasFieldMaskedExcluded(const CostFxnSSDBiasFieldMaskedExcluded& rhs);
      /// Copy assignment operator
      CostFxnSSDBiasFieldMaskedExcluded& operator=(const CostFxnSSDBiasFieldMaskedExcluded& rhs);
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
      /// \param mask_mov 3D volume used to mask the moving volume
      /// \param warp_field Warp field to resample vol_mov
      /// \param bias_field Bias field to apply to vol_ref
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        bias field will be sampled every 2mm
      CostFxnSSDBiasFieldMaskedExcluded(
          std::shared_ptr<MMORF::Volume>                 vol_ref,
          std::shared_ptr<MMORF::Volume>                 vol_mov,
          const arma::fmat&                              affine_ref,
          const arma::fmat&                              affine_mov,
          std::shared_ptr<MMORF::Volume>                 mask_ref,
          std::shared_ptr<MMORF::Volume>                 mask_mov,
          const std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          std::shared_ptr<MMORF::BiasFieldBSpline>       bias_field,
          int                                            sampling_frequency,
          const bool                                     debug = false
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
      class                 Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
      /// Global class identifier allocator
      static int            next_id_;
  }; // CostFxnSSDBiasFieldMaskedExcluded
} // MMORF
#endif // COST_FXN_SSD_BIAS_FIELD_MASKED_EXCLUDED_CUH
