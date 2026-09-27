/*  calc_grad_perc_dev.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2012-2014 University of Oxford  */

/*  CCOPYRIGHT  */

// Calculates the percentage deviation in the gradient fields due to non-linearities
// Relies on the output of gradient_unwarp.py

#include <iostream>
#include <string>

#include "utils/options.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;


// The two strings below specify the title and example usage that is
//  printed out as the help or usage message

string title="calc_grad_perc_dev (Version 1.0)\nCopyright(c) 2003, University of Oxford (Mark Jenkinson)";
string examples="calc_grad_perc_dev [options] --fullwarp=<warp image> --out=<basename>";

// Each (global) object below specificies as option and can be accessed
//  anywhere in this file (since they are global).  The order of the
//  arguments needed is: name(s) of option, default value, help message,
//       whether it is compulsory, whether it requires arguments
// Note that they must also be included in the main() function or they
//  will not be active.

Option<bool> verbose(string("-v,--verbose"), false,
		     string("switch on diagnostic messages"),
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<string> fullwarp(string("--fullwarp"), string(""),
		  string("full warp image from gradient_unwarp.py"),
		  true, requires_argument);
Option<string> outname(string("-o,--out"), string(""),
		  string("output basename"),
		  true, requires_argument);
int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// Local functions

// for example ... print difference of COGs between 2 images ...
int do_work(int argc, char* argv[])
{
  volume4D<float> fw, xpd, ypd, zpd, tmp;
  volume<float> mask;
  read_volume4D(fw,fullwarp.value());
  mask=fw[0]*0.0f + 1.0f;
  tmp=lrxgrad(fw[0],mask);
  xpd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrygrad(fw[0],mask);
  xpd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrzgrad(fw[0],mask);
  xpd.addvolume((tmp[0]+tmp[1])*0.5f);
  xpd[0]*=100.0/fw.xdim();
  xpd[1]*=100.0/fw.ydim();
  xpd[2]*=100.0/fw.zdim();
  tmp=lrxgrad(fw[1],mask);
  ypd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrygrad(fw[1],mask);
  ypd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrzgrad(fw[1],mask);
  ypd.addvolume((tmp[0]+tmp[1])*0.5f);
  ypd[0]*=100.0/fw.xdim();
  ypd[1]*=100.0/fw.ydim();
  ypd[2]*=100.0/fw.zdim();
  tmp=lrxgrad(fw[2],mask);
  zpd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrygrad(fw[2],mask);
  zpd.addvolume((tmp[0]+tmp[1])*0.5f);
  tmp=lrzgrad(fw[2],mask);
  zpd.addvolume((tmp[0]+tmp[1])*0.5f);
  zpd[0]*=100.0/fw.xdim();
  zpd[1]*=100.0/fw.ydim();
  zpd[2]*=100.0/fw.zdim();
  string bname=make_basename(outname.value());
  save_volume4D(xpd,bname+"_x");
  save_volume4D(ypd,bname+"_y");
  save_volume4D(zpd,bname+"_z");
  return 0;
}

////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{

  Tracer tr("main");
  OptionParser options(title, examples);

  try {
    // must include all wanted options here (the order determines how
    //  the help message is printed)
    options.add(fullwarp);
    options.add(outname);
    options.add(verbose);
    options.add(help);

    nonoptarg = options.parse_command_line(argc, argv);

    // line below stops the program if the help was requested or
    //  a compulsory option was not set
    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }

  }  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }

  // Call the local functions

  return do_work(argc,argv);
}
