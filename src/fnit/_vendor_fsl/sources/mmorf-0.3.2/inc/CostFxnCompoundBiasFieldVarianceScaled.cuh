//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost function of cost functions, with variance weighting
/// \details This cost function is composed of a number of weighted cost functions and an
///          additional cost function acting as a regulariser. The total cost/grad/hess is
///          a weighted sum of the outputs of all the sub cost functions. Additionally, the
///          non-regularising cost functions are scaled according to the inverse of their
///          variance at the the minimum cost observed up until that point in time.
///          NB!NB!NB!NB! THIS MEANS THAT THE SAME PARAMETERS MAY RETURN DIFFERENT RESULTS
///          FOR COST/GRAD/HESS DEPENDING ON WHEN THEY ARE SET!!!
/// \author Frederik Lange
/// \date January 2023
/// \copyright Copyright (C) 2023 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef COST_FXN_COMPOUND_bias_FIELD_VARIANCE_SCALED_CUH
#define COST_FXN_COMPOUND_bias_FIELD_VARIANCE_SCALED_CUH

#include "CostFxn.h"
#include "SparseDiagonalMatrixTiled.cuh"
#include "BiasField.h"

#include <armadillo>

#include <memory>
#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxnCompoundBiasFieldVarianceScaled : public CostFxn
  {
    public:
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
      /// Default dtor
      ~CostFxnCompoundBiasFieldVarianceScaled();
      /// Move ctor
      CostFxnCompoundBiasFieldVarianceScaled(
          CostFxnCompoundBiasFieldVarianceScaled&& rhs);
      /// Move assignment operator
      CostFxnCompoundBiasFieldVarianceScaled& operator=(
          CostFxnCompoundBiasFieldVarianceScaled&& rhs);
      /// Copy ctor
      CostFxnCompoundBiasFieldVarianceScaled(
          const CostFxnCompoundBiasFieldVarianceScaled& rhs);
      /// Copy assignment operator
      CostFxnCompoundBiasFieldVarianceScaled& operator=(
          const CostFxnCompoundBiasFieldVarianceScaled& rhs);
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
      /// Construct using two pre-made cost functions and a regularisation factor lambda
      /// \details NOTE: Very importantly, the common_bias_field should be shared by ALL
      ///                cost_fxns. This is not explicitly tested for and is the API
      ///                user's responsibility to enforce
      CostFxnCompoundBiasFieldVarianceScaled(
          std::vector<std::shared_ptr<MMORF::CostFxn> >  cost_functions,
          const std::vector<float>&                      lambda_cost_functions,
          std::shared_ptr<MMORF::CostFxn>                regulariser,
          const float                                    lambda_regulariser,
          std::shared_ptr<MMORF::BiasField>              common_bias_field);
      /// Get the current value of the parameters
      virtual std::vector<std::vector<float> > get_parameters() const override;
      /// Set the current value of the parameters
      virtual void set_parameters(
          const std::vector<std::vector<float> >& parameters) override;
      /// Current value of the cost function
      virtual float cost() const override;
      /// Derivitive of cost function at current parameterisation
      virtual arma::fvec grad() const override;
      /// Hessian of cost function at current parameterisation
      virtual MMORF::SparseDiagonalMatrixTiled hess() const override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // CostFxnCompoundBiasFieldVarianceScaled
} // MMORF
#endif // COST_FXN_COMPOUND_bias_FIELD_VARIANCE_SCALED_CUH
