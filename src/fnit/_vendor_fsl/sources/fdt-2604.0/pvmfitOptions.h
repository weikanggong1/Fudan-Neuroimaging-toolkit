/*  pvmfitOptions.h

    Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 1999-2009 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(pvmfitOptions_h)
#define pvmfitOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "commonopts.h"

using namespace Utilities;

namespace PVMFIT {

class pvmfitOptions {
 public:
  static pvmfitOptions& getInstance();
  ~pvmfitOptions() { delete gopt; }

  Option<bool>   verbose;
  Option<bool>   help;
  Option<string> datafile;
  Option<string> ofile;
  Option<string> maskfile;
  Option<string> bvecsfile;
  Option<string> bvalsfile;
  Option<int>    nfibres;
  Option<int>    modelnum;
  Option<bool>   all;
  Option<bool>   cnonlinear;
  Option<bool>   cnonlinear_Fanning;
  Option<bool>   gridsearch;
  Option<bool>   use_f0;
  Option<bool>   saveBIC;

  bool parse_command_line(int argc, char** argv);

 private:
  pvmfitOptions();
  const pvmfitOptions& operator=(pvmfitOptions&);
  pvmfitOptions(pvmfitOptions&);

  OptionParser options;

  static pvmfitOptions* gopt;

};

 inline pvmfitOptions& pvmfitOptions::getInstance(){
   if(gopt == NULL)
     gopt = new pvmfitOptions();

   return *gopt;
 }

 inline pvmfitOptions::pvmfitOptions() :
  verbose(string("-V,--verbose"), false,
	  string("switch on diagnostic messages"),
	  false, no_argument),
   help(string("-h,--help"), false,
	string("display this message"),
	false, no_argument),
   datafile(string("-k,--data"), "",
	       string("data file"),
	       true, requires_argument),
   ofile(string("-o,--out"), string("pvm"),
	       string("Output basename - default='pvm'"),
	       false, requires_argument),
   maskfile(string("-m,--mask"), "",
	    string("Bet binary mask file"),
	    true, requires_argument),
   bvecsfile(string("-r,--bvecs"), "",
	     string("b vectors file"),
	     true, requires_argument),
   bvalsfile(string("-b,--bvals"), "",
	     string("b values file"),
	     true, requires_argument),
   nfibres(string("-n,--nfibres"), 1,
	     string("number of fibres to fit - default=1"),
	     false, requires_argument),
   modelnum(string("--model"), 1,
	     string("\t1:Ball-Sticks (single-shell); 2:Ball-Sticks (multi-shells); 4:Ball-Binghams"),
	     false, requires_argument),
   all(string("--all"),false,
	      string("\tFanning models: Fit all models from 1 up to N fibres and choose the best using BIC"),
	      false,no_argument),
   cnonlinear(string("--cnonlinear"),false,
	      string("Model1: Apply constrained nonlinear optimization on the diffusivity, volume fractions and their sum"),
	      false,no_argument),
   cnonlinear_Fanning(string("--cnonlinear_F"),false,
	      string("Model1: Apply constrained nonlinear optimization on the diffusivity, volume fractions and their sum.\n\t\t\tReturn n fanning angle estimates, using the Hessian of the cost function"),
	      false,no_argument),
   gridsearch(string("--gridsearch"),false,
	      string("Use grid search (on the fanning eigenvalues). Default=off"),
	      false,no_argument),
   use_f0(string("--f0"),false,
	      string("\tInclude noise floor parameter in the model"),
	      false,no_argument),
   saveBIC(string("--BIC"),false,
	      string("\tSave BIC for certain models"),
	      false,no_argument),
   options("pvmfit", "pvmfit -k <datafile> -m <maskfile> -r <bvecsfile> -b <bvalsfile> [-n 2]\n")
   {


     try {
       options.add(verbose);
       options.add(help);
       options.add(datafile);
       options.add(ofile);
       options.add(maskfile);
       options.add(bvecsfile);
       options.add(bvalsfile);
       options.add(nfibres);
       options.add(modelnum);
       options.add(all);
       options.add(cnonlinear);
       options.add(cnonlinear_Fanning);
       options.add(gridsearch);
       options.add(use_f0);
       options.add(saveBIC);
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
