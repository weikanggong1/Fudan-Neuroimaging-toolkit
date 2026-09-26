/*  Directional Statistics Functions

    Bingham and Watson Distributions and functions to approximate their normalizing constant

    Stam Sotiropoulos  - FMRIB Image Analysis Group

    Copyright (C) 2011 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined (Bingham_Watson_approx_h)
#define Bingham_Watson_approx_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "stdlib.h"

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INV3 0.333333333333333   // 1/3
#define SQRT3 1.732050807568877 //sqrt(3)
#define INV54 0.018518518518519 // 1/54

#define min3(a,b,c)  (a < b ? std::min(a,c) : std::min(b,c) )


//Saddlepoint approximation of confluent hypergeometric function of a matrix argument B
//Vector x has the eigenvalues of B.
float hyp_Sapprox(NEWMAT::ColumnVector& x);


//Saddlepoint approximation of confluent hypergeometric function of a matrix argument, with its eigenvalues being l1,l1,l2 or l1,l2,l2 with l1!=l2.
//Vector x has the three eigenvalues. This function can be also used to approximate a confluent hypergeometric function of a scalar argument k
//by providing x=[k 0 0].
float hyp_Sapprox_twoequal(NEWMAT::ColumnVector& x);


//Saddlepoint approximation of the ratio of two hypergeometric functions, with matrix arguments L and B (3x3). Vectors xL & xB contain the eigenvalues of L and B.
//Used for the ball & Binghams model.
float hyp_SratioB(NEWMAT::ColumnVector& xL,NEWMAT::ColumnVector& xB);


//Saddlepoint aproximation of the ratio ot two hypergeometric functions with matrix arguments L and B in two steps: First denominator, then numerator.
//This allows them to be updated independently, used for the ball & Binghams model to compute the likelihood faster.
//This function returns values used in the denominator approximation. xB containes the two non-zero eigenvalues of matrix B.
NEWMAT::ReturnMatrix approx_denominatorB(NEWMAT::ColumnVector& xB);


//Second step for saddlepoint approximation of the ratio of two hypergeometric functions with matrix arguments L and B (xL has the eigenvalues of L).
//Assume that the denominator has already been approximated by the function above and the parameters are stored in denomvals.
//Here approximate the numerator and return the total ratio approximation.
float hyp_SratioB_knowndenom(NEWMAT::ColumnVector &xL,NEWMAT::ColumnVector& denomvals);


//Saddlepoint approximation of the ratio of two hypergeometric functions, one with matrix argument L and another with scalar argument k. Vector xL contains the eigenvalues of L.
//Used for the ball & Watsons model.
float hyp_SratioW(NEWMAT::ColumnVector& xL,const double k);


//Saddlepoint aproximation of the ratio ot two hypergeometric functions, one with matrix arguments L and the other with scalar argument k in two steps:
//First denominator, then numerator. This allows them to be updated independently, used for the ball & Watsons model to compute the likelihood faster.
//This function returns values used in the denominator approximation.
NEWMAT::ReturnMatrix approx_denominatorW(const double k);


//Second step for saddlepoint approximation of the ratio of two hypergeometric functions, with matrix argument L and scalar argument k (xL has the eigenvalues of L).
//Assume that the denominator has already been approximated by the function above and the parameters are stored in denomvals.
//Here approximate the numerator and return the total ratio approximation.
float hyp_SratioW_knowndenom(NEWMAT::ColumnVector &xL,NEWMAT::ColumnVector& denomvals);


//Using the values of vector x, construct a qubic equation and solve it analytically.
//Solution used for the saddlepoint approximation of the confluent hypergeometric function with matrix argument B (3x3) (See Kume & Wood, 2005)
//Vector x contains the eigenvalues of B.
float find_t(const NEWMAT::ColumnVector& x);

//cubic root
float croot(const float& x);


#endif
