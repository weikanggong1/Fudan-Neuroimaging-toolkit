
/**
 *  \file
 *  \brief Parameters of a warp field
 *  \details Implements the MMORF::Parameters interface
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef PARAMETERS_WARP_H
#define PARAMETERS_WARP_H

#include "Parameters.h"

#include <vector>
#include <memory>

/** Multi-Modal Registration Framework */
namespace MMORF
{
  class ParametersWarp : public Parameters
  {
    public:
      /** Ctor defining only dimensions */
      ParametersWarp(const std::vector<int>& dimensions);
      /** Ctor defing both dimensions and values */
      ParametersWarp(const std::vector<int>& dimensions,
                     const std::vector<double>& values);
      /** Default dtor */
      ~ParametersWarp();
      /** Move ctor */
      ParametersWarp(ParametersWarp&& rhs);
      /** Move assignment operator */
      ParametersWarp& operator=(ParametersWarp&& rhs);
      /** Copy ctor */
      ParametersWarp(const ParametersWarp& rhs);
      /** Copy assignment operator */
      ParametersWarp& operator=(const ParametersWarp& rhs);
      /** \brief Return the value of the parameter at position `index`
       *  \details Should ensure that the `index` is valid
       *  \param index [in] n-dimensional index into Parameters */
      double operator[](const std::vector<int>& index) const override;
      /** Get parameter dimensions */
      std::vector<int> dimensions() const override;
    private:
      /** Forward declaration */
      class Impl;
      /** Pointer to actual implementation object */
      std::unique_ptr<Impl> pimpl_;
    }; // ParametersWarp
} // MMORF
#endif // PARAMETERS_WARP_H
