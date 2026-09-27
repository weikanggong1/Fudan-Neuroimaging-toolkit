//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief N-dimensional position
/// \details Very multi-purpose class for positioning into a variety of objects which are
///          defined in terms of floating point positions.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "PositionND.h"

#include <vector>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class PositionND::Impl
  {
    public:
      /// Empty constructor
      Impl()
      {}
      /// Construct using vector of indices
      Impl(const std::vector<float>& positions)
        : positions_(positions)
      {}
      /// Get position dimensions
      int get_dimensions() const
      {
        return positions_.size();
      }
      /// Get position values as std::vector
      std::vector<float> get_positions() const
      {
        return positions_;
      }
      /// Get reference to specific position. This should allow changing of values in position
      float& operator[](int subscript)
      {
        return positions_[subscript];
      }
      /// Set all positions
      void set_positions(const std::vector<float>& positions)
      {
        positions_ = positions;
      }
    private:
      std::vector<float> positions_;
  };
////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
  // Rule of 5
  /// Default dtor
  PositionND::~PositionND() = default;
  /// Move ctor
  PositionND::PositionND(PositionND&& rhs) = default;
  /// Move assignment operator
  PositionND& PositionND::operator=(PositionND&& rhs) = default;
  /// Copy ctor
  PositionND::PositionND(const PositionND& rhs)
    : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  PositionND& PositionND::operator=(const PositionND& rhs)
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
  PositionND::PositionND()
    : pimpl_(MMORF::make_unique<Impl>())
  {}
  /// Construct using vector of indices
  PositionND::PositionND(const std::vector<float>& positions)
    : pimpl_(MMORF::make_unique<Impl>(positions))
  {}
  /// Get position dimensions
  int PositionND::get_dimensions() const
  {
    return pimpl_->get_dimensions();
  }
  /// Get position values as std::vector
  std::vector<float> PositionND::get_positions() const
  {
    return pimpl_->get_positions();
  }
  /// Get reference to specific position. This should allow changing of values in position
  float& PositionND::operator[](int subscript)
  {
    return (*pimpl_)[subscript];
  }
  /// Set all positions
  void PositionND::set_positions(const std::vector<float>& positions)
  {
    pimpl_->set_positions(positions);
  }
} // MMORF
