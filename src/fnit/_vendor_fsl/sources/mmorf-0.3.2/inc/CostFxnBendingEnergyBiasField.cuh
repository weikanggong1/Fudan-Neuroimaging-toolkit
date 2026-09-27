//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function based on the bending energy of a bias field
/// \details This cost function is designed primarily for use in regularising bias field
///          defined by B-splines, and not as a stand-alone cost function
/// \author Frederik Lange
/// \date March 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_BENDING_ENERGY_BIAS_FIELD_CUH
#define COST_FXN_BENDING_ENERGY_BIAS_FIELD_CUH

#include "CostFxn.h"
#include "BiasFieldBSpline.cuh"
#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnBendingEnergyBiasField : public MMORF::CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnBendingEnergyBiasField();
      /// Move ctor
      CostFxnBendingEnergyBiasField(CostFxnBendingEnergyBiasField&& rhs);
      /// Move assignment operator
      CostFxnBendingEnergyBiasField& operator=(CostFxnBendingEnergyBiasField&& rhs);
      /// Copy ctor
      CostFxnBendingEnergyBiasField(const CostFxnBendingEnergyBiasField& rhs);
      /// Copy assignment operator
      CostFxnBendingEnergyBiasField& operator=(const CostFxnBendingEnergyBiasField& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct by passing in a fully constructed bias field
      CostFxnBendingEnergyBiasField(
          std::shared_ptr<MMORF::BiasFieldBSpline> bias_field,
          int                                      sampling_frequency);
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
  }; // CostFxnBendingEnergyBiasField
} // MMORF
#endif // COST_FXN_BENDING_ENERGY_BIAS_FIELD_CUH
