/**
 *  \file
 *  \brief Useful inline functions for the MMORF framework
 *  \details Prevent code duplication by placing some common simple functions into this
 *           header file.
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
 */

#ifndef MMORF_HELPERS_H
#define MMORF_HELPERS_H

#include <vector>

/** Multi-Modal Registration Framework */
namespace MMORF
{
  /** \brief Convert an subscript index to linear index
   *  \details This is meant to be a fast function, so assumes all checks for validity of
   *           indexing has been performed before calling
   *  \param index [in] subscript index to be converted (0 to n-1)
   *  \param dimensions [in] range of possible indices in each dimension (1 to n)
   */
  inline int index_sub_to_lin(const std::vector<int>& index,
                              const std::vector<int>& dimensions)
  {
    auto lin_index = index.at(0);
    auto no_of_indices = index.size();
    for (std::size_t i = 1; i < no_of_indices; ++i)
    {
      lin_index += index.at(i) * dimensions.at(i-1);
    }
    return lin_index;
  }
}
#endif // MMORF_HELPERS_H
