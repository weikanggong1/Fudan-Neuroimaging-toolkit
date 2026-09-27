
/**
 *  \file
 *  \brief Vector class using Armadillo library
 *  \details Implements the MMORF::Vector interface, using the Armadillo library
 *           as the data store
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef VECTOR_ARMADILLO_H
#define VECTOR_ARMADILLO_H

#include "Vector.h"
#include <vector>
#include <memory>

/** Multi-Modal Registration Framework */
namespace MMORF
{
  class VectorArmadillo : public Vector
  {
    public:
      /** \brief Ctor only specifying size of vector
       *  \details Initialises all values to zero */
      VectorArmadillo(int size);
      /** Ctor specifying all values */
      VectorArmadillo(std::vector<double> vals);
      /** Default dtor */
      ~VectorArmadillo();
      /** Move ctor */
      VectorArmadillo(VectorArmadillo&& rhs);
      /** Move assignment operator */
      VectorArmadillo& operator=(VectorArmadillo&& rhs);
      /** Copy ctor */
      VectorArmadillo(const VectorArmadillo& rhs);
      /** Copy assignment operator */
      VectorArmadillo& operator=(const VectorArmadillo& rhs);
      /** Access element of vector by 0-indexed value */
      double operator[](int index) const override;
      /** Number of elements in Vector */
      int size() const override;
    private:
      /** Forward declaration */
      class Impl;
      /** Pointer to actual implementation object */
      std::unique_ptr<Impl> pimpl_;
  }; // VectorArmadillo
} // MMORF
#endif // VECTOR_ARMADILLO_H
