//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Nonlinear optimiser basic interface
/// \details Provides the interface to which all derived Optimisers must conform
/// \author Frederik Lange
/// \date May 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OPTIMISER_H
#define OPTIMISER_H

#include "CostFxn.h"
#include "LinearSolver.h"

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class Optimiser
  {
    public:
      /// Default destructor
      virtual ~Optimiser() {}
      /// Perform optimisation
      /// \details Returns the parameters which optimise the function
      /// \param cost_fxn Cost function to be minimised
      /// \param starting_params Initial guess of optimal parameters
      /// \param iterations_max Maximum number of iterations to perform before stopping
      /// \param tolerance_relative If the relative change in cost between iterations is less
      ///                           than this value then the optimisation is considered to
      ///                           have converged
      virtual std::vector<std::vector<float> > optimise(
          MMORF::CostFxn&                         cost_fxn,
          const std::vector<std::vector<float> >& starting_params,
          const int                               iterations_max,
          const float                             tolerance_relative
          ) const = 0;
  }; // Optimiser
} // MMORF
#endif // OPTIMISER_H
