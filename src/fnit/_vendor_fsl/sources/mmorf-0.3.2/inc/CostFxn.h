//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Cost Function basic interface
/// \details Provides the interface to which all derived Cost Functions must conform
/// \author Frederik Lange
/// \date Decenber 2017
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef COST_FXN_H
#define COST_FXN_H

#include "SparseDiagonalMatrixTiled.cuh"

#include <armadillo>

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class CostFxn
  {
    public:
      /// Default destructor
      virtual ~CostFxn() {}
      /// Get the current value of the parameters
      virtual std::vector<std::vector<float> > get_parameters() const = 0;
      /// Set the current value of the parameters
      virtual void set_parameters(
          const std::vector<std::vector<float> >& parameters) = 0;
      /// Current value of the cost function
      virtual float cost() const = 0;
      /// Derivitive of cost function at current parameterisation
      virtual arma::fvec grad() const = 0;
      /// Hessian of cost function at current parameterisation
      virtual MMORF::SparseDiagonalMatrixTiled hess() const = 0;
  }; // CostFxn
} // MMORF
#endif // COST_FXN_H
