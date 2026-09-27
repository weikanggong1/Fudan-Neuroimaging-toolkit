//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the bending energy of a warp field
/// \details This cost function is designed primarily for use in regularising warps defined
///          by B-splines, and not as a stand-alone cost function
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_BENDING_ENERGY_CUH
#define COST_FXN_BENDING_ENERGY_CUH

#include "CostFxn.h"
#include "WarpFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnBendingEnergy : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnBendingEnergy();
      /// Move ctor
      CostFxnBendingEnergy(CostFxnBendingEnergy&& rhs);
      /// Move assignment operator
      CostFxnBendingEnergy& operator=(CostFxnBendingEnergy&& rhs);
      /// Copy ctor
      CostFxnBendingEnergy(const CostFxnBendingEnergy& rhs);
      /// Copy assignment operator
      CostFxnBendingEnergy& operator=(const CostFxnBendingEnergy& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed warpfield
      CostFxnBendingEnergy(
          std::shared_ptr<MMORF::WarpFieldBSpline> warp_field,
          int sampling_frequency);
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
  }; // CostFxnBendingEnergy
} // MMORF
#endif // COST_FXN_BENDING_ENERGY_CUH
