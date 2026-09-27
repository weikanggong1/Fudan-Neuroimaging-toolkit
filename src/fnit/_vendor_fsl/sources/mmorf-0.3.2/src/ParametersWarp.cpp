/**
 *  \file
 *  \brief Parameters of a warp field
 *  \details Implements the MMORF::Parameters interface
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#include "MmorfMemory.h"
#include "MmorfHelpers.h"
#include "ParametersWarp.h"

#include <vector>
#include <memory>

/** Multi-Modal Registration Framework */
namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  /** Implementation class for ParametersWarp */
  class ParametersWarp::Impl
  {
    public:
      Impl(const std::vector<int>& dimensions)
      : dimensions_(dimensions)
      {
        auto no_of_values = 1;
        for (const auto v : dimensions)
        {
          no_of_values *= v;
        }
        values_.resize(no_of_values, 0.0);
      }
      Impl(const std::vector<int>& dimensions,
           const std::vector<double>& values)
      : dimensions_(dimensions), values_(values)
      {}
      /** \brief Return the value of the parameter at position `index`
       *  \details Should ensure that the `index` is valid
       *  \param index [in] n-dimensional index into Parameters
       *  \todo I must add check for validity of index */
      double operator[](const std::vector<int>& index) const
      {
        auto lin_index = MMORF::index_sub_to_lin(index, dimensions_);
        auto value = values_.at(lin_index);
        return value;
      }
      /** \todo COMMENT */
      std::vector<int> dimensions() const
      {
        return dimensions_;
      }
    private:
      std::vector<int> dimensions_;
      std::vector<double> values_;
  };

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
  /** Ctor defining only dimensions */
  ParametersWarp::ParametersWarp(const std::vector<int>& dimensions)
  : pimpl_(MMORF::make_unique<Impl>(dimensions))
  {}
  /** Ctor defing both dimensions and values */
  ParametersWarp::ParametersWarp(const std::vector<int>& dimensions,
                                 const std::vector<double>& values)
  : pimpl_(MMORF::make_unique<Impl>(dimensions, values))
  {}
  /** Default dtor */
  ParametersWarp::~ParametersWarp() = default;
  /** Move ctor */
  ParametersWarp::ParametersWarp(ParametersWarp&& rhs) = default;
  /** Move assignment operator */
  ParametersWarp& ParametersWarp::operator=(ParametersWarp&& rhs) = default;
  /** Copy ctor */
  ParametersWarp::ParametersWarp(const ParametersWarp& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /** Copy assignment operator */
  ParametersWarp& ParametersWarp::operator=(const ParametersWarp& rhs)
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
  /** \brief Return the value of the parameter at position `index`
   *  \details Should ensure that the `index` is valid
   *  \param index n-dimensional index into Parameters */
  double ParametersWarp::operator[](const std::vector<int>& index) const
  {
    return (*pimpl_)[index];
  }
  /** \todo COMMENT */
  std::vector<int> ParametersWarp::dimensions() const
  {
    return pimpl_->dimensions();
  }
} // MMORF
