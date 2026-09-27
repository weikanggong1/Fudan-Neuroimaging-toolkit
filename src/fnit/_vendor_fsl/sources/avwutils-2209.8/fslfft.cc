/*  fslfft.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2003-2007 University of Oxford  */

/*  CCOPYRIGHT  */


// Calculates and prints the nth cog of an image

#include <iostream>
#include <string>

#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "armawrap/newmat.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

void print_usage(const string& progname) {
  cerr << "Usage: " << progname << " <input volume> <output volume> [-inv]"
       << endl << endl;
  cerr << "e.g.   " << progname << " invol outvol"
       << endl;
}



int main(int argc,char *argv[])
{

  Tracer tr("main");

  string progname=argv[0];
  if (argc<3) {
    print_usage(progname);
    return -1;
  }

  bool inverse=false;
  if (argc>=4) {
    string optarg=argv[3];
    if (optarg=="-inv") {
      inverse=true;
    } else {
      cerr << "Unrecognised option: " << argv[3] << endl;
      return 1;
    }
  }

  string inname=string(argv[1]), outname=string(argv[2]);


  complexvolume vin, vc;
  read_complexvolume(vin,inname);
  vc = vin;
  if (inverse) {
    ifft2(vc);
  } else {
    fft2(vc);
  }
  save_complexvolume(vc,outname);

  return 0;
}
