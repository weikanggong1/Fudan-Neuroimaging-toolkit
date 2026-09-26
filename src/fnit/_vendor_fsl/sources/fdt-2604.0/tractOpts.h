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

using namespace Utilities;

namespace TRACT {

class tractOptions {
 public:
  static tractOptions& getInstance();
  ~tractOptions() { delete gopt; }

  Option<bool> verbose;
  Option<bool> help;
  Option<string> basename;
  Option<string> maskfile;
  Option<int> nparticles;
  Option<int> nsteps;
  Option<bool> usef;
  Option<float> rseed;
  bool parse_command_line(int argc, char** argv);

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
  verbose(string("-V,--verbose"), false,
	  string("switch on diagnostic messages"),
	  false, no_argument),
   help(string("-h,--help"), false,
	string("display this message"),
	false, no_argument),
   basename(string("-s,--samples"), string("DTI"),
	       string("basename for samples files"),
	       true, requires_argument),
   maskfile(string("-m,--mask"), string("mask"),
	    string("Bet binary mask file"),
	    true, requires_argument),
   nparticles(string("-P,--nparticles"), 10000,
	 string("Number of particles"),
	 false, requires_argument),
   nsteps(string("-S,--nsteps"), 1000,
	    string("Number of steps per particle"),
	    false, requires_argument),
   usef(string("-f,--usef"), false,
	 string("Use anisotropy to constrain tracking"),
	 false, no_argument),
   rseed(string("-s,--seed"), 0.324571,
	 string("Random seed"),
	 false, requires_argument),
   options("dtisamples", "dtisamples -k <filename>\n dtisamples --verbose\n")
   {


     try {
       options.add(verbose);
       options.add(help);
       options.add(basename);
       options.add(maskfile);
       options.add(nparticles);
       options.add(nsteps);
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
