//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the sum of squared differences between two volumes, one of
///        which is warped
/// \details The warp is defined by b-spline field coefficients, one set per dimension in the
///          volume
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_SSD_WARP_FIELD
#define COST_FXN_SSD_WARP_FIELD

#include "CostFxn.h"
#include "Volume.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnSSDWarpField : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnSSDWarpField();
      /// Move ctor
      CostFxnSSDWarpField(CostFxnSSDWarpField&& rhs);
      /// Move assignment operator
      CostFxnSSDWarpField& operator=(CostFxnSSDWarpField&& rhs);
      /// Copy ctor
      CostFxnSSDWarpField(const CostFxnSSDWarpField& rhs);
      /// Copy assignment operator
      CostFxnSSDWarpField& operator=(const CostFxnSSDWarpField& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in fully constructed volumes, and defining warp field
      /// characteristics
      /// \param vol_reference Reference (stationary) volume
      /// \param vol_moving Transformed volume
      /// \param affine_ref Pre-calculated affine transform FROM a common space TO the
      ///                   reference space
      /// \param affine_mov Pre-calculated affine transform FROM a common space TO the
      ///                   moving space
      /// \param knot_spacing Warp field B-spline knot spacing (mm) in reference space
      /// \param sampling_frequency How often to sample between B-spline knots. E.g. for:
      ///             knot_spacing        = 10mm
      ///             sampling_frequency  = 5
      ///        warp field will be sampled every 2mm
      CostFxnSSDWarpField(
          std::shared_ptr<MMORF::Volume>           vol_reference,
          std::shared_ptr<MMORF::Volume>           vol_moving,
          const arma::fmat&                        affine_ref,
          const arma::fmat&                        affine_mov,
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          const int                                sampling_frequency
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
  }; // CostFxnSSDWarpField
} // MMORF
#endif // COST_FXN_SSD_WARP_FIELD
