//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief N-dimensional position
/// \details Very multi-purpose class for positioning into a variety of objects which are
///          defined in terms of floating point positions.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef POSITION_ND_H
#define POSITION_ND_H

#include "Position.h"

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  /// Position into n-dimensional object
  class PositionND : public Position
  {
    public:
      // Rule of 5
      /// Default dtor
      ~PositionND();
      /// Move ctor
      PositionND(PositionND&& rhs);
      /// Move assignment operator
      PositionND& operator=(PositionND&& rhs);
      /// Copy ctor
      PositionND(const PositionND& rhs);
      /// Copy assignment operator
      PositionND& operator=(const PositionND& rhs);

      // Class specific functions
      /// Empty constructor
      PositionND();
      /// Construct using vector of indices
      PositionND(const std::vector<float>& positions);
      /// Get position dimensions
      virtual int get_dimensions() const override;
      /// Get position values as std::vector
      virtual std::vector<float> get_positions() const override;
      /// Get reference to specific position. This should allow changing of values in position
      virtual float& operator[](int subscript) override;
      /// Set all positions
      virtual void set_positions(const std::vector<float>& positions) override;
    private:
      /// Forward declaration
      class Impl;
      /// Pointer to actual implementation object
      std::unique_ptr<Impl> pimpl_;
  }; // PositionND
} // MMORF
#endif // POSITION_ND_H
