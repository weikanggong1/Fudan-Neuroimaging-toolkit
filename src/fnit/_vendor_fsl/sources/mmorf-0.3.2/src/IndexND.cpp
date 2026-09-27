//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief N-dimensional index
/// \details Very multi-purpose class for indexing into a variety of objects which are defined
///          in terms of integer indices.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "MmorfMemory.h"
#include "IndexND.h"

#include <vector>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class IndexND::Impl
  {
    public:
      /// Empty constructor
      Impl()
      {}
      /// Construct using vector of indices
      Impl(const std::vector<int>& indices)
        : indices_(indices)
      {}
      /// Get index dimensions
      int get_dimensions() const
      {
        return indices_.size();
      }
      /// Get index values as std::vector
      std::vector<int> get_indices() const
      {
        return indices_;
      }
      /// Get reference to specific index. This should allow changing of values in index
      int& operator[](int subscript)
      {
        return indices_[subscript];
      }
      /// Set all indices
      void set_indices(const std::vector<int>& indices)
      {
        indices_ = indices;
      }
    private:
      std::vector<int> indices_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
  // Rule of 5
  /// Default dtor
  IndexND::~IndexND() = default;
  /// Move ctor
  IndexND::IndexND(IndexND&& rhs) = default;
  /// Move assignment operator
  IndexND& IndexND::operator=(IndexND&& rhs) = default;
  /// Copy ctor
  IndexND::IndexND(const IndexND& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  IndexND& IndexND::operator=(const IndexND& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
    else{
      *pimpl_ = *rhs.pimpl_;
    }
    return *this;
  }

  // Class specific functions
  /// Empty constructor
  IndexND::IndexND()
    : pimpl_(MMORF::make_unique<Impl>())
  {}
  /// Construct using vector of indices
  IndexND::IndexND(const std::vector<int>& indices)
    : pimpl_(MMORF::make_unique<Impl>(indices))
  {}
  /// Get index dimensions
  int IndexND::get_dimensions() const
  {
    return pimpl_->get_dimensions();
  }
  /// Get index values as std::vector
  std::vector<int> IndexND::get_indices() const
  {
    return pimpl_->get_indices();
  }
  /// Get reference to specific index. This should allow changing of values in index
  int& IndexND::operator[](int subscript)
  {
    return (*pimpl_)[subscript];
  }
  /// Set all indices
  void IndexND::set_indices(const std::vector<int>& indices)
  {
    pimpl_->set_indices(indices);
  }
} // MMORF
