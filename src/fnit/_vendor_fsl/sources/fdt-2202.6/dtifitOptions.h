/*  BpmOptions.h

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(dtifitOptions_h)
#define dtifitOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "commonopts.h"

using namespace Utilities;

namespace DTIFIT {

class dtifitOptions {
 public:
  static dtifitOptions& getInstance();
  ~dtifitOptions() { delete gopt; }

  Option<bool> verbose;
  Option<bool> help;
  Option<string> dtidatafile;
  Option<string> ofile;
  Option<string> maskfile;
  Option<string> bvecsfile;
  Option<string> bvalsfile;
  Option<string> cni;              // confounds of no interest.
  Option<bool> sse;                // Sum of squared errors
  Option<bool> wls;                // Perform Weighted Least squares for tensor fitting
  Option<bool> kurt;               // Output mean kurtosis
  Option<bool> kurtdir;            // Output kurtosis maps along each eigenvector
  Option<bool> littlebit;
  Option<bool> savetensor;
  Option<int> z_min;
  Option<int> z_max;
  Option<int> y_min;
  Option<int> y_max;
  Option<int> x_min;
  Option<int> x_max;
  Option<string> grad_file;
  FmribOption<bool> save_bvals;

  bool parse_command_line(int argc, char** argv);

 private:
  dtifitOptions();
  const dtifitOptions& operator=(dtifitOptions&);
  dtifitOptions(dtifitOptions&);

  OptionParser options;

  static dtifitOptions* gopt;

};

 inline dtifitOptions& dtifitOptions::getInstance(){
   if(gopt == NULL)
     gopt = new dtifitOptions();

   return *gopt;
 }

 inline dtifitOptions::dtifitOptions() :
  verbose(string("-V,--verbose"), false,
	  string("switch on diagnostic messages"),
	  false, no_argument),
   help(string("-h,--help"), false,
	string("display this message"),
	false, no_argument),
   dtidatafile(string("-k,--data"), string("data"),
	       string("dti data file"),
	       true, requires_argument),
   ofile(string("-o,--out"), string("dti"),
	       string("Output basename"),
	       true, requires_argument),
   maskfile(string("-m,--mask"), string("mask"),
	    string("Bet binary mask file"),
	    true, requires_argument),
   bvecsfile(string("-r,--bvecs"), string("bvecs"),
	     string("b vectors file"),
	     true, requires_argument),
   bvalsfile(string("-b,--bvals"), string("bvals"),
	     string("b values file"),
	     true, requires_argument),
   cni(string("--cni"), string(""),
	     string("Input confound regressors"),
	     false, requires_argument),
   sse(string("--sse"), false,
	     string("Output sum of squared errors"),
	     false, no_argument),
   wls(string("-w,--wls"),false,
             string("Fit the tensor with weighted least squares"),
             false, no_argument),
   kurt(string("--kurt"), false,
	     string("Output mean kurtosis map (for multi-shell data)"),
	     false,  no_argument),
   kurtdir(string("--kurtdir"), false,
	     string("Output maps of kurtosis along each eigenvector: K1, K2, and K3 (for multi-shell data)"),
	     false,  no_argument),
   littlebit(string("--littlebit"), false,
	     string("Only process small area of brain"),
	     false, no_argument),
   savetensor(string("--save_tensor"), false,
	     string("Save the elements of the tensor"),
	     false, no_argument),
   z_min(string("-z,--zmin"), 0,
	 string("min z"),
	 false, requires_argument),
   z_max(string("-Z,--zmax"), 42,
	 string("max z"),
	 false, requires_argument),
   y_min(string("-y,--ymin"), 0,
	 string("min y"),
	 false, requires_argument),
   y_max(string("-Y,--ymax"), 128,
	 string("max y"),
	 false, requires_argument),
   x_min(string("-x,--xmin"), 0,
	 string("min x"),
	 false, requires_argument),
   x_max(string("-X,--xmax"), 128,
	 string("max x"),
	 false, requires_argument),
   grad_file(string("--gradnonlin"), string("gradnonlin"),
	     string("Gradient Nonlinearity Tensor file"),
	     false, requires_argument),
   save_bvals(string("--savebvals"), false,
	     string("Save 4D file with bvalues, corrected for gradient nonlinearities"),
	     false,  no_argument),
   options("dtifit", "dtifit -k <filename>\n dtifit --verbose\n")
   {


     try {
       options.add(verbose);
       options.add(help);
       options.add(dtidatafile);
       options.add(ofile);
       options.add(maskfile);
       options.add(bvecsfile);
       options.add(bvalsfile);
       options.add(cni);
       options.add(sse);
       options.add(wls);
       options.add(kurt);
       options.add(kurtdir);
       options.add(littlebit);
       options.add(savetensor);
       options.add(z_min);
       options.add(z_max);
       options.add(y_min);
       options.add(y_max);
       options.add(x_min);
       options.add(x_max);
       options.add(grad_file);
       options.add(save_bvals);
     }
     catch(X_OptionError& e) {
       options.usage();
       cerr << endl << e.what() << endl;
     }
     catch(std::exception &e) {
       cerr << e.what() << endl;
     }

   }
}

#endif
