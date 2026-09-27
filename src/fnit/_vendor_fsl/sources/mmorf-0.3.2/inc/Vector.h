
/**
 *  \file
 *  \brief Vector interface
 *  \details Pure Abstract class defining the minimum necessary interface for a Vector object
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef VECTOR_H
#define VECTOR_H

/** Multi-Modal Registration Framework */
namespace MMORF
{
  class Vector
  {
    public:
      /** Default destructor */
      virtual ~Vector() {}
      /** Access element of vector by 0-indexed value */
      virtual double operator[](int index) const = 0;
      /** Number of elements in Vector */
      virtual int size() const = 0;
  }; // Vector
} // MMORF
#endif // VECTOR_H
