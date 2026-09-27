/*  fslorient.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2003-2012 University of Oxford  */

/*  CCOPYRIGHT  */


#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <iostream>
#include <string>

#include "NewNifti/NewNifti.h"
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"

using namespace std;
using namespace NiftiIO;
using namespace NEWMAT;
using namespace NEWIMAGE;

void print_usage() {
  string progname="fslorient";
  cout << "Usage: " << progname << " <main option> <filename>" << endl;
  cout << endl;
  cout << "  where the main option is one of:" << endl;
  cout << "    -getorient             (prints FSL left-right orientation)" << endl;
  cout << "    -getsform              (prints the 16 elements of the sform matrix)" << endl;
  cout << "    -getqform              (prints the 16 elements of the qform matrix)" << endl;
  cout << "    -setsform <m11 m12 ... m44>  (sets the 16 elements of the sform matrix)" << endl;
  cout << "    -setqform <m11 m12 ... m44>  (sets the 16 elements of the qform matrix)" << endl;
  cout << "    -getsformcode          (prints the sform integer code)" << endl;
  cout << "    -getqformcode          (prints the qform integer code)" << endl;
  cout << "    -setsformcode <code>   (sets sform integer code)" << endl;
  cout << "    -setqformcode <code>   (sets qform integer code)" << endl;
  cout << "    -copysform2qform       (sets the qform equal to the sform - code and matrix)" << endl;
  cout << "    -copyqform2sform       (sets the sform equal to the qform - code and matrix)" << endl;
  cout << "    -deleteorient          (removes orient info from header)" << endl;
  cout << "    -forceradiological     (makes FSL radiological header)" << endl;
  cout << "    -forceneurological     (makes FSL neurological header - not Analyze)" << endl;
  cout << "    -swaporient            (swaps FSL radiological and FSL neurological)" << endl;
  cout << endl;
  cout << "       Note: the stored data order is never changed here - only the header info." << endl;
  cout << "       To change the data storage use fslswapdim." << endl;
  cout << endl;
  cout << "  e.g.  " << progname << " -forceradiological myimage" << endl;
  cout << "        " << progname << " -copysform2qform myimage" << endl;
  cout << "        " << progname << " -setsform -2 0 0 90 0 2 0 -126 0 0 2 -72 0 0 0 1 myimage" << endl;
  cout << endl;
}



template <class T>
void swaporient(volume4D<T>& invol) {
  Matrix swapmat(IdentityMatrix(4));
  swapmat(1,1)=-1;
  swapmat(1,4)=invol.xsize()-1;
  invol.set_sform(invol.sform_code(),invol.sform_mat() * swapmat);
  invol.set_qform(invol.qform_code(),invol.qform_mat() * swapmat);
}


template <class T>
int getorient(const volume4D<T>& invol)
{
  if (invol.left_right_order()==FSL_RADIOLOGICAL) {
    cout << "RADIOLOGICAL" << endl;
  } else {
    cout << "NEUROLOGICAL" << endl;
  }
  return 0;
}

int showmat(const Matrix& mat) {
  for (int a=1; a<=4; a++) {
    for (int b=1; b<=4; b++) {
      cout << mat(a,b) << " ";
    }
  }
  cout << endl;
  return 0;
}


int testminargs(int argnum, int argc) {
  if (argc < (argnum + 1)) {
    print_usage();
    exit(EXIT_FAILURE);
  }
  return 0;
}


template <class T>
int fmrib_main(int argc,char *argv[])
{
  bool modified(false);
  string option(argv[1]), filename(argv[argc-1]);
  volume<T> invol;

  read_orig_volume(invol,filename);

  if ( argc==2 || option=="-getorient") {
    getorient(invol);
  } else if (option=="-getsformcode") {
    cout << invol.sform_code() << endl;
  } else if (option=="-getqformcode") {
    cout << invol.qform_code() << endl;
  } else if (option=="-getqform") {
    showmat(invol.qform_mat());
  } else if (option=="-getsform") {
    showmat(invol.sform_mat());
  } else if (option=="-setsformcode") {
    testminargs(3,argc);
    modified=true;
    int code = (int) atof(argv[2]);
    invol.set_sform(code,invol.sform_mat());
  } else if (option=="-setqformcode") {
    testminargs(3,argc);
    modified=true;
    int code = (int) atof(argv[2]);
    invol.set_qform(code,invol.qform_mat());
  } else if ( (option=="-setqform") || (option=="-setsform") ) {
    testminargs(18,argc);
    modified=true;
    Matrix mat(4,4);
    for (int a=1; a<=4; a++) {
      for (int b=1; b<=4; b++) {
	mat(a,b)=atof(argv[a*4+b-3]);
      }
    }
    // override the last line
    mat(4,1)=0; mat(4,2)=0; mat(4,3)=0; mat(4,4)=1;
    if (option=="-setqform") {
      invol.set_qform(invol.qform_code(),mat);
    }
    if (option=="-setsform") {
      invol.set_sform(invol.sform_code(),mat);
    }
  } else if (option=="-copysform2qform") {
    modified=true;
    invol.set_qform(invol.sform_code(),invol.sform_mat());
  } else if (option=="-copyqform2sform") {
    modified=true;
    invol.set_sform(invol.qform_code(),invol.qform_mat());
  } else if (option=="-swaporient") {
    modified=true;
    swaporient(invol);
  } else if (option=="-deleteorient") {
    modified=true;
    invol.set_sform(NIFTI_XFORM_UNKNOWN,arma::zeros(4,4));
    invol.set_qform(NIFTI_XFORM_UNKNOWN,arma::zeros(4,4));
  } else if (option=="-forceradiological") {
    modified=true;
    if (invol.left_right_order()==FSL_NEUROLOGICAL) {
      swaporient(invol);
    }
  } else if (option=="-forceneurological") {
    modified=true;
    if (invol.left_right_order()==FSL_RADIOLOGICAL) {
      swaporient(invol);
    }
  } else {
    cerr << "Unrecognised option: " << option << endl;
    print_usage();
    return(-1);
  }



  if (modified) {
    int filetype( fslFileType(filename) );
    if ( filetype != FSL_TYPE_ANALYZE ) {
      save_orig_volume(invol,filename,filetype);
    } else {
      cerr << "Cannot modify orientation for Analyze files" << endl;
      cerr << "  All Analyze files are treated as radiological" << endl;
      cerr << "  To change the data storage use fslswapdim" << endl;
      return(-1);
    }
  }
  return 0;
}


int main(int argc,char *argv[])
{
  if (argc<2) {
    print_usage();
    return -1;
  }
  return call_fmrib_main(dtype(string(argv[argc-1])),argc,argv);
}
