//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Simple class for specifying a position
/// \details This may not be entirely necessary, but it keeps with the general MMORF
///          philosophy of programming to abstract concepts. note that these positions are
///          ALWAYS floating point values.
/// \author Frederik Lange
/// \date April 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef POSITION_H
#define POSITION_H

#include <vector>

/// Multi-Modal Registration Framework
namespace MMORF
{
  class Position
  {
    public:
      /// Default dtor
      virtual ~Position() {}
      /// Get position dimensions
      virtual int get_dimensions() const = 0;
      /// Get position values as std::vector
      virtual std::vector<float> get_positions() const = 0;
      /// Get reference to specific position. This should allow changing of values in position
      virtual float& operator[](int subscript) = 0;
      /// Set all positions
      virtual void set_positions(const std::vector<float>& positions) = 0;
  }; // Position
} // MMORF
#endif // POSITION_H
