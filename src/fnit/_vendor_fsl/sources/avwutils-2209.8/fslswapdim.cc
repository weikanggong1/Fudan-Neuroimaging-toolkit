/*  fslswapdim.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2003-2008 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <iostream>
#include <string>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;

void print_usage(const string& progname) {
  cout << "Usage: " << progname << " <input> <a> <b> <c> [output] [--checkLR]" << endl;
  cout << endl;
  cout << "  where a,b,c represent the new x,y,z axes in terms of the" << endl;
  cout << "  old axes.  They can take values of -x,x,y,-y,z,-z" << endl;
  cout << "  --checkLR  is an option that checks if the specified arguments lead to a Left-Right swap or not - it cannot be used with an output name" << endl;
  cout << "  e.g.  " << progname << " invol y x -z outvol" << endl;
}


template <class T>
int fmrib_main(int argc,char *argv[])
{
  string newx=argv[2], newy=argv[3], newz=argv[4];
  string inname=argv[1];
  string outname="";
  bool showmat=false;
  bool checkonly=false;
  if (argc==6) {
    outname=argv[5];
    if (outname=="--checkLR") {
      outname="";
      checkonly=true;
    }
    showmat=false;
  } else {
    showmat=true;
  }

  volume4D<T> invol;
  read_orig_volume(invol,inname);

  Matrix affmat;
  affmat = invol.swapmat(newx,newy,newz);

  if (showmat) {
    cout << affmat << endl;
  }

  if (checkonly) {
    if (affmat.Determinant()<0.0) cout << "LR orientation changed" << endl;
    else cout << "LR orientation preserved" << endl;
    return 0;
  }

  if (affmat.Determinant()<0.0) {
    cout << "WARNING:: Flipping Left/Right orientation (as det < 0)" << endl;
  }

  invol.swapdimensions(newx,newy,newz,true);

  if ( invol.sform_code() == NiftiIO::NIFTI_XFORM_UNKNOWN )
    invol.set_sform(NiftiIO::NIFTI_XFORM_UNKNOWN,arma::zeros(4,4));
  if ( invol.qform_code() == NiftiIO::NIFTI_XFORM_UNKNOWN )
    invol.set_qform(NiftiIO::NIFTI_XFORM_UNKNOWN,arma::zeros(4,4));

  int retval=0;
  if (outname!="") {
    retval = save_orig_volume(invol,outname);
  }
  return retval;
}


int main(int argc,char *argv[])
{
  string progname=argv[0];
  if (argc<5) {
    print_usage(progname);
    return -1;
  }

  string inname = argv[1];
  // call the templated main
  return call_fmrib_main(dtype(inname),argc,argv);
}
