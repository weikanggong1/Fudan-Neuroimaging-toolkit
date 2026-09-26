/*    Copyright (C) 2012 University of Oxford  */

/*  CCOPYRIGHT  */
#include <cmath>
#include <stdlib.h>


#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .sampling_mat()
#endif

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"

NEWMAT::ReturnMatrix rodrigues(const float&,NEWMAT::ColumnVector&);

NEWMAT::ReturnMatrix rodrigues(const float&,const float&,NEWMAT::ColumnVector&);

NEWMAT::ReturnMatrix rodrigues(const NEWMAT::ColumnVector&,const NEWMAT::ColumnVector&);

NEWMAT::ReturnMatrix ppd(const NEWMAT::Matrix&,const NEWMAT::ColumnVector&, const NEWMAT::ColumnVector&);

void vecreg_aff(const NEWIMAGE::volume4D<float>&,NEWIMAGE::volume4D<float>&,const NEWIMAGE::volume<float>&,const NEWMAT::Matrix&,const NEWIMAGE::volume<float>&);

void vecreg_nonlin(const NEWIMAGE::volume4D<float>&,NEWIMAGE::volume4D<float>&,const NEWIMAGE::volume<float>&,NEWIMAGE::volume4D<float>&,const NEWIMAGE::volume<float>&);

void sjgradient(const NEWIMAGE::volume<float>&,NEWIMAGE::volume4D<float>&);
