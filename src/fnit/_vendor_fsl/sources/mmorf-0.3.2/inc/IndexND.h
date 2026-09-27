//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief N-dimensional index
/// \details Very multi-purpose class for indexing into a variety of objects which are defined
///          in terms of integer indices.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef INDEX_ND_H
#define INDEX_ND_H

#include "Index.h"

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Index into n-dimensional object
  class IndexND : public Index
  {
    public:
      // Rule of 5
      /// Default dtor
      ~IndexND();
      /// Move ctor
      IndexND(IndexND&& rhs);
      /// Move assignment operator
      IndexND& operator=(IndexND&& rhs);
      /// Copy ctor
      IndexND(const IndexND& rhs);
      /// Copy assignment operator
      IndexND& operator=(const IndexND& rhs);

      // Class specific functions
      /// Empty constructor
      IndexND();
      /// Construct using vector of indices
      IndexND(const std::vector<int>& indices);
      /// Get index dimensions
      int get_dimensions() const override;
      /// Get index values as std::vector
      std::vector<int> get_indices() const override;
      /// Get reference to specific index. This should allow changing of values in index
      int& operator[](int subscript) override;
      /// Set all indices
      void set_indices(const std::vector<int>& indices) override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // IndexND
} // MMORF
#endif // INDEX_ND_H
