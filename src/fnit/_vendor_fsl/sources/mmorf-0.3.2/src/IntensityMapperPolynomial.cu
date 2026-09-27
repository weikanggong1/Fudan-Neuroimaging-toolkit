//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Map intensities between volumes using a polynomian function
/// \details Takes a set of intensities in the domain and maps them to intensities in the
///          range using a polynomial mapping. Is capable of calculating a best fit polynomial
///          of a given degree when supplied with set of "experimental" or "measured" pairs
///          of values.
/// \author Frederik Lange
/// \date September 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "MmorfMemory.h"
#include "IntensityMapperPolynomial.cuh"

#include <armadillo>

#include <memory>
#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

namespace MMORF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  class IntensityMapperPolynomial::Impl
  {
    public:
      //////////////////////////////////////////////////
      // Public functions
      //////////////////////////////////////////////////
      /// Default ctor
      Impl()
      {}
      /// Sampled data constructor
      Impl(
          const int                 polynomial_degree,
          const std::vector<float>& intensities_domain,
          const std::vector<float>& intensities_range,
          const bool                debug,
          const int                 id)
        : debug_(debug)
        , id_(id)
      {
        if (debug_){
          auto tmp_debug_dir = "mkdir -p " + debug_dir_ + std::to_string(id_);
          auto tmp_sysval = std::system(tmp_debug_dir.c_str());
        }
        assert(polynomial_degree >= 0);
        assert(intensities_domain.size() == intensities_range.size());
        assert(intensities_domain.size() > polynomial_degree);
        degree_ = polynomial_degree;
        calculate_coefficients_(
            polynomial_degree,
            intensities_domain,
            intensities_range);
      }

      /// Map intensities from the domain to the range
      virtual std::vector<float> map_intensities(
          const std::vector<float>& intensities_domain) const
      {
        return intensities_domain;
        //auto intensities_mapped = std::vector<float>(intensities_domain.size(), 0.0f);
        //for (auto i = 0; i < intensities_domain.size(); ++i){
        //  auto intensity_norm = (intensities_domain[i] - mu_x_)/sigma_x_;
        //  for (auto n = degree_; n >= 0; --n){
        //    intensities_mapped[i] += coeffs_[degree_ - n]
        //      * std::pow(intensity_norm, static_cast<float>(n));
        //  }
        //}
        //return intensities_mapped;
      }

    private:
      //////////////////////////////////////////////////
      // Private functions
      //////////////////////////////////////////////////
      /// Calculates polynomial coefficients which fit the intensity mapping in a least
      /// squares sense
      /// \details Solves the problem Xp = y in a least squares sense, where:
      ///                       X = [ 1  x_1 ... x_1^(n-1) x_1^n]
      ///                           |... ... ...    ...     ... |
      ///                           [ 1  x_i ... x_i^(n-1) x_i^n]
      ///                       p = [p_0 p_1 ... p_(n-1) p_n]^T
      ///                       y = [y_1 y_2 ... y_(n-1) y_n]^T
      ///
      ///          The method employed in based on the SVD decomposition of the V matrix.
      ///          Whilst this may be slower than using the "normal equations" it is far more
      ///          stable, which is necessary as X may be very poorly conditioned. This should
      ///          not be a significant problem here, as we would normally fit a relatively
      ///          low order (n = 3) polynomial, and this will not be performed often.
      void calculate_coefficients_(
          const int                 polynomial_degree,
          const std::vector<float>& intensities_domain,
          const std::vector<float>& intensities_range)
      {
        auto vec_x = arma::fvec(intensities_domain);
        auto vec_y = arma::fvec(intensities_range);
        auto mat_U = arma::fmat();
        auto vec_s = arma::fvec();
        auto mat_V = arma::fmat();

        //remove_unlikely_differences_(vec_x, vec_y, 3.0f);

        mu_x_ = arma::mean(vec_x);
        sigma_x_ = arma::stddev(vec_x);
        auto vec_x_norm = arma::fvec((vec_x - mu_x_)/sigma_x_);

        auto mat_X = make_vandermonde_matrix_(
          polynomial_degree,
          vec_x_norm);
        arma::svd_econ(mat_U, vec_s, mat_V, mat_X);
        auto vec_p = svd_to_coefficients_(
            mat_V,
            vec_s,
            mat_U,
            vec_y);
        coeffs_ = arma::conv_to<std::vector<float> >::from(vec_p);

        if (debug_){
          std::cout << "////////////////////////////////////////////////////////////////\n"
                    << "////  In Intensity Mapper Polynomial ID: " << id_ << "\n"
                    << "////  mu_x = " << mu_x_ << "\n"
                    << "////  sigma_x = " << sigma_x_ << "\n"
                    << "////////////////////////////////////////////////////////////////\n"
                    << std::endl;
          vec_x_norm.save(debug_dir_ + std::to_string(id_) + "/vec_x", arma::raw_ascii);
          vec_y.save(debug_dir_ + std::to_string(id_) + "/vec_y", arma::raw_ascii);
          mat_X.save(debug_dir_ + std::to_string(id_) + "/mat_x", arma::raw_ascii);
          mat_U.save(debug_dir_ + std::to_string(id_) + "/mat_U", arma::raw_ascii);
          vec_s.save(debug_dir_ + std::to_string(id_) + "/vec_s", arma::raw_ascii);
          mat_V.save(debug_dir_ + std::to_string(id_) + "/mat_V", arma::raw_ascii);
          vec_p.save(debug_dir_ + std::to_string(id_) + "/vec_p", arma::raw_ascii);
        }
      }

      /// Calculate the mean and standard deviation of the error between two vectors, and
      /// return only those values whose differences fall withing a given range of the mean
      void remove_unlikely_differences_(
          arma::fvec& vec_a,
          arma::fvec& vec_b,
          const float acceptable_deviation)
      {
        auto n_in = arma::size(vec_a);
        // Remove very small (background hopefully) values
        auto vec_a_out = arma::fvec(arma::size(vec_a));
        auto vec_b_out = arma::fvec(arma::size(vec_b));
        auto vec_err = arma::fvec(arma::size(vec_a));
        {
          auto n_out = 0;
          for (auto i = 0; i < vec_err.size(); ++i){
            if (vec_a[i] >= 1 &&
                vec_b[i] >= 1){
              vec_a_out[n_out] = vec_a[i];
              vec_b_out[n_out] = vec_b[i];
              vec_err[n_out] = vec_a[i] - vec_b[i];
              ++n_out;
            }
          }
          vec_a_out = vec_a_out.head(n_out);
          vec_b_out = vec_b_out.head(n_out);
          vec_err = vec_err.head(n_out);
        }

        auto mu = arma::mean(vec_err);
        auto sigma = arma::stddev(vec_err);
        {
          auto n_out = 0;
          for (auto i = 0; i < vec_err.size(); ++i){
            if (std::abs(vec_err[i]) <= mu + sigma*acceptable_deviation){
              vec_a_out[n_out] = vec_a_out[i];
              vec_b_out[n_out] = vec_b_out[i];
              ++n_out;
            }
          }
          vec_a = vec_a_out.head(n_out);
          vec_b = vec_b_out.head(n_out);
        }
      }

      /// Calculate the Vandermonde matrix for the given polynomial degree and domain samples
      arma::fmat make_vandermonde_matrix_(
          const int                 polynomial_degree,
          const arma::fvec&         vec_x) const
      {
        auto mat_X = arma::fmat(
            vec_x.size(),
            polynomial_degree + 1,
            arma::fill::ones);
        for (auto i = polynomial_degree; i > 0; --i){
          mat_X.col(i - 1) = mat_X.col(i) % vec_x;
        }
        return mat_X;
      }

      /// Use the SVD decomposition of X to solve Xp = y for p
      arma::fmat svd_to_coefficients_(
          arma::fmat mat_V,
          arma::fvec vec_s,
          arma::fmat mat_U,
          arma::fvec vec_y)
      {
        // If the singular value is close to zero, do not invert, and set to zero instead
        auto vec_s_inv = vec_s;
        for (auto& s : vec_s_inv){
          if (std::abs(s) > 10.0f*std::numeric_limits<float>::min()){
            s = 1.0f/s;
          }
          else{
            s = 0;
          }
        }
        auto mat_S_inv = arma::diagmat(vec_s_inv);
        arma::fmat vec_p = mat_V * mat_S_inv * mat_U.t() * vec_y;
        return vec_p;
      }

      //////////////////////////////////////////////////
      // Private datamembers
      //////////////////////////////////////////////////
      int                       degree_;
      std::vector<float>        coeffs_;
      float                     mu_x_;
      float                     sigma_x_;
      // Debugging options
      bool                      debug_;
      int                       id_;
      const static std::string  debug_dir_;
  };

  /// Static debug directory initialisation
  const std::string IntensityMapperPolynomial::Impl::debug_dir_ =
    "debug/IntensityMapperPolynomial/";

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Rule of 5
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  IntensityMapperPolynomial::~IntensityMapperPolynomial() = default;
  /// Move ctor
  IntensityMapperPolynomial::IntensityMapperPolynomial(
      IntensityMapperPolynomial&& rhs) = default;
  /// Move assignment operator
  IntensityMapperPolynomial& IntensityMapperPolynomial::operator=(
      IntensityMapperPolynomial&& rhs) = default;
  /// Copy ctor
  IntensityMapperPolynomial::IntensityMapperPolynomial(const IntensityMapperPolynomial& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = MMORF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  IntensityMapperPolynomial& IntensityMapperPolynomial::operator=(
      const IntensityMapperPolynomial& rhs)
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
////////////////////////////////////////////////////////////////////////////////
// Class Specific Functions
////////////////////////////////////////////////////////////////////////////////
  /// Default ctor
  IntensityMapperPolynomial::IntensityMapperPolynomial()
    : pimpl_(MMORF::make_unique<Impl>())
  {}

  /// Sampled data constructor
  IntensityMapperPolynomial::IntensityMapperPolynomial(
      const int                 polynomial_degree,
      const std::vector<float>& intensities_domain,
      const std::vector<float>& intensities_range,
      const bool                debug)
    : pimpl_(MMORF::make_unique<Impl>(
          polynomial_degree,
          intensities_domain,
          intensities_range,
          debug,
          next_id_++))
  {}

  /// Map intensities from the domain to the range
  std::vector<float> IntensityMapperPolynomial::map_intensities(
      const std::vector<float>& intensities_domain) const
  {
    return pimpl_->map_intensities(intensities_domain);
  }

  /// Static datamember initialisation
  int IntensityMapperPolynomial::next_id_ = 0;
} // MMORF
