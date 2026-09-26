/*  tractOptions.h

    Tim Behrens, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(tractOptions_h)
#define tractOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "commonopts.h"


using namespace Utilities;

namespace TRACT {

class tractOptions {
 public:
  static tractOptions& getInstance();
  ~tractOptions() { delete gopt; }

  Option<int> verbose;
  Option<bool> help;
  Option<string> basename;
  Option<string> maskfile;
  Option<string> rubbish_file;
  Option<string> skipmask;
  Option<string> outfile;
  Option<string> seedfile;
  Option<int> nparticles;
  Option<int> nsteps;
  Option<float> steplength;
  Option<float> c_thr;
  Option<bool> modeuler;
  Option<bool> noloopcheck;
  Option<bool> usef;
  Option<float> rseed;
  void parse_command_line(int argc, char** argv);
  void status();
 private:
  tractOptions();
  const tractOptions& operator=(tractOptions&);
  tractOptions(tractOptions&);

  OptionParser options;

  static tractOptions* gopt;

};

 inline tractOptions& tractOptions::getInstance(){
   if(gopt == NULL)
     gopt = new tractOptions();

   return *gopt;
 }

 inline tractOptions::tractOptions() :
  verbose(string("-V,--verbose"), 0,
	  string("verbose level, [0-2]"),
	  false, requires_argument),
   help(string("-h,--help"), false,
	string("display this message"),
	false, no_argument),
   basename(string("-s,--samples"), string("DTI"),
	       string("basename for samples files"),
	       true, requires_argument),
   maskfile(string("-m,--mask"), string("mask"),
	    string("Bet binary mask file"),
	    true, requires_argument),
   rubbish_file(string("--rubbish"), string(""),
	    string("Rubbish file (tracking stops here)."),
	    false, requires_argument),
  skipmask(string("--no_integrity"), string(""),
	   string("no explanation needed"),
	   false, requires_argument),
   outfile(string("-o,--out"), string("out"),
	    string("Output file"),
	    true, requires_argument),
   seedfile(string("-x,--seed"), string("Seed"),
	    string("File with seed points in. e.g 68 68 32\n                                                      78 23 52"),
	    true, requires_argument),
   nparticles(string("-P,--nparticles"), 10000,
	 string("Number of particles"),
	 false, requires_argument),
   nsteps(string("-S,--nsteps"), 1000,
	    string("Number of steps per particle"),
	    false, requires_argument),
   steplength(string("-l,steplength"), 0.5,
	      string("Steplength"),
	      false, requires_argument),
  c_thr(string("-c,--cthr"), 0.2,
	string("Curvature threshold"),
	false, requires_argument),
  modeuler(string("--modeuler"), false,
	      string("Do modified euler integration instead of simple euler"),
	      false, no_argument),
  noloopcheck(string("--noloopcheck"), false,
	 string("Don't perform loopchecking"),
	 false, no_argument),
  usef(string("-f,--usef"), false,
	 string("Use anisotropy to constrain tracking"),
	 false, no_argument),
   rseed(string("--rseed"), 0.324571,
	 string("Random seed"),
	 false, requires_argument),
   options("tract2","tract2 -s <basename> -m <maskname> -x <seedfile> -o <output>\n tract2 --help\n")
   {


     try {
       options.add(verbose);
       options.add(help);
       options.add(basename);
       options.add(maskfile);
       options.add(rubbish_file);
       options.add(skipmask);
       options.add(seedfile);
       options.add(outfile);
       options.add(nparticles);
       options.add(nsteps);
       options.add(steplength);
       options.add(c_thr);
       options.add(modeuler);
       options.add(noloopcheck);
       options.add(usef);
       options.add(rseed);
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
