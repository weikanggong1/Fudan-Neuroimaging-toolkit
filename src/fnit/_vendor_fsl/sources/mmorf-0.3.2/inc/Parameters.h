
/**
 *  \file
 *  \brief Parameter interface
 *  \details Pure Abstract class for passing parameter values to function objects
 *  \author Frederik Lange
 *  \date 12/01/2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef PARAMETERS_H
#define PARAMETERS_H 

#include <vector>

/** Multi-Modal Registration Framework */
namespace MMORF
{
  class Parameters
  {
    public:
      /** Default destructor */
      virtual ~Parameters() {}
      /** \brief Return the value of the parameter at position `index`
       *  \details Should ensure that the `index` is valid
       *  \param index [in] n-dimensional index into Parameters */
      virtual double operator[](std::vector<int> const &index) const = 0;
      /** Get parameter dimensions */
      virtual std::vector<int> dimensions() const = 0;
  }; // Parameters
} // MMORF
#endif // PARAMETERS_H
