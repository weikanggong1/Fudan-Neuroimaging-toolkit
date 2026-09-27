//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Simple class for indexing into a range of objects
/// \details This may not be entirely necessary, but it keeps with the general MMORF
///          philosophy of programming to abstract concepts. note that these positions are
///          ALWAYS integers.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef INDEX_H
#define INDEX_H

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class Index
  {
    public:
      /// Default dtor
      virtual ~Index() {}
      /// Get index dimensions
      virtual int get_dimensions() const = 0;
      /// Get index values as std::vector
      virtual std::vector<int> get_indices() const = 0;
      /// Get reference to specific index. This should allow changing of values in index
      virtual int& operator[](int subscript) = 0;
      /// Set all indices
      virtual void set_indices(const std::vector<int>& indices) = 0;
  }; // Index
} // MMORF
#endif // INDEX_H
