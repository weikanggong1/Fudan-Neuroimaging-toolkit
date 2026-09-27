/**
 *  \file
 *  \brief Provide an interface matching BASISFIELD::splinefield, specifically for use with
           EDDY::MoveBySuscCF class
 *  \details Essentially this is a wrapper around a BASISFIELD::splinefield object, which
             re-implements some of the more expensive calculations on the GPU, whilst simply
             passing other calculations through to the underlying BASISFIELD::splinefield
             object. As such, expect limited functionality as compared to the BASISFIELD
             version - just enough to get EDDY::MoveBySuscCF running faster. Additionally,
             we are using the pimpl idiom here.
 *  \author Frederik Lange
 *  \date February 2018
 *  \copyright (c) 2018 University of Oxford
*/
#ifndef EDDY_SPLINE_FIELD_CUH
#define EDDY_SPLINE_FIELD_CUH

#include "newimage/newimageall.h"
#include "newmat.h"
#include "miscmaths/bfmatrix.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"

#include <boost/shared_ptr.hpp>

#include <vector>
#include <memory>

/** Multi-Modal Registration Framework */
namespace MMORF
{
  class EddySplineField
  {
    public:
      /** Default dtor */
      ~EddySplineField();
      /** Move ctor */
      EddySplineField(EddySplineField&& rhs);
      /** Move assignment operator */
      EddySplineField& operator=(EddySplineField&& rhs);
      /** Copy ctor */
      EddySplineField(const EddySplineField& rhs);
      /** Copy assignment operator */
      EddySplineField& operator=(const EddySplineField& rhs);

      // Methods to wrap
      /** Standard BASISFIELD::splinefield constructor */
      EddySplineField(const std::vector<unsigned int>& psz, const std::vector<double>& pvxs, 
          const std::vector<unsigned int>& pksp, int porder=3);
      
      unsigned int CoefSz_x() const;
      unsigned int CoefSz_y() const;
      unsigned int CoefSz_z() const;
      /** From base class BASISFIELD::basisfield */
      void AsVolume(NEWIMAGE::volume<float>& vol, BASISFIELD::FieldIndex fi=BASISFIELD::FIELD);
      /** From base class BASISFIELD::basisfield */
      void SetCoef(const NEWMAT::ColumnVector& pcoef);
      /** Calculate Bending Energy for regularisation purposes */
      double BendEnergy() const; 
      /** Calculate gradient of the Bending Energy */
      NEWMAT::ReturnMatrix BendEnergyGrad() const;
      /** Calculate hessian of the Bending Energy */
      boost::shared_ptr<MISCMATHS::BFMatrix> BendEnergyHess(
          MISCMATHS::BFMatrixPrecisionType prec) const;
      /** Jte V1 */
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&  ima1,
                               const NEWIMAGE::volume<float>&  ima2,
                               const NEWIMAGE::volume<char>    *mask)
                               const;
      /** Jte V2 */
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima1,
                               const NEWIMAGE::volume<float>&    ima2,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      /** Jte V3 */
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      /** Jte V4 */
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      /** JtJ V1 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      /** JtJ V2 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      /** JtJ V3 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv, 
                                                 const NEWIMAGE::volume<float>& ima,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      /** JtJ V4 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv, 
                                                 const NEWIMAGE::volume<float>& ima1,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      /** JtJ V5 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv1,
                                                 const NEWIMAGE::volume<float>& ima1,
                                                 const std::vector<unsigned int>& deriv2,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      /** JtJ V6 */
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                                 const BASISFIELD::basisfield& bf2,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;

    private:
      /** Forward declaration */
      class Impl;
      /** Pointer to actual implementation object */
      std::unique_ptr<Impl> pimpl_;
  }; // EddySplineField
} // MMORF
#endif // EDDY_SPLINE_FIELD_CUH
