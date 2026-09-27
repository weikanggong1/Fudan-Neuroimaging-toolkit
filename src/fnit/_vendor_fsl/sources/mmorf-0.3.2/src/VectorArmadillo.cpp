/**
 *  \file
 *  \brief Vector class using Armadillo library
 *  \details Implements the MMORF::Vector interface, using the Armadillo library
 *           as the data store
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#include "MmorfMemory.h"
#include "VectorArmadillo.h"
#include "armadillo"
#include <vector>
#include <memory>

/** Multi-Modal Registration Framework */
namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  /** Implementation class for VectorArmadillo */
  class VectorArmadillo::Impl
  {
    public:
      Impl(int size)
      : data_(size,arma::fill::zeros)
      {}
      Impl(std::vector<double> vals)
      : data_(vals)
      {}
      double operator[](int index) const
      {
        return data_(index);
      }
      int size() const
      {
        return data_.n_rows;
      }
    private:
      arma::colvec data_;
  };

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
  /** \brief Ctor only specifying size of vector
   *  \details Initialises all values to zero */
  VectorArmadillo::VectorArmadillo(int size)
  : pimpl_(MMORF::make_unique<Impl>(size))
  {}
  /** Ctor specifying all values */
  VectorArmadillo::VectorArmadillo(std::vector<double> vals)
  : pimpl_(MMORF::make_unique<Impl>(vals))
  {}
  /** Default dtor */
  VectorArmadillo::~VectorArmadillo() = default;
  /** Move ctor */
  VectorArmadillo::VectorArmadillo(VectorArmadillo&& rhs) = default;
  /** Move assignment operator */
  VectorArmadillo& VectorArmadillo::operator=(VectorArmadillo&& rhs) = default;
  /** Copy ctor */
  VectorArmadillo::VectorArmadillo(const VectorArmadillo& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /** Copy assignment operator */
  VectorArmadillo& VectorArmadillo::operator=(const VectorArmadillo& rhs)
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
  double VectorArmadillo::operator[](int index) const
  {
    return (*pimpl_)[index];
  }
  /** Number of elements in Vector */
  int VectorArmadillo::size() const
  {
    return pimpl_->size();
  }
} // MMORF
