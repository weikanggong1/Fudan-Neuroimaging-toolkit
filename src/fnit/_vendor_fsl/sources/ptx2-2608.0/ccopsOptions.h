/*  ccopsOptions.h

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 1999-2010 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(ccopsOptions_h)
#define ccopsOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "commonopts.h"

using namespace Utilities;

namespace CCOPS {

class ccopsOptions {
 public:
  static ccopsOptions& getInstance();
  ~ccopsOptions() { delete gopt; }
  
  Option<bool>      help;
  Option<bool>      verbose;

  Option<string>    inmatrix2;
  Option<string>    obasename;
  Option<string>    ptxdir;

  Option<bool>      reord1;
  Option<bool>      reord2;
  Option<bool>      reord3;

  Option<string>    excl_mask;
  Option<float>     connexity;
  Option<int>       bin;
  Option<float>     power;

  Option<string>    mask;
  Option<string>    scheme;
  Option<int>       nclusters;

  bool parse_command_line(int argc, char** argv);
  
 private:
  ccopsOptions();  
  const ccopsOptions& operator=(ccopsOptions&);
  ccopsOptions(ccopsOptions&);

  OptionParser options; 
      
  static ccopsOptions* gopt;
  
};

 inline ccopsOptions& ccopsOptions::getInstance(){
   if(gopt == NULL)
     gopt = new ccopsOptions();
   
   return *gopt;
 }

 inline ccopsOptions::ccopsOptions() :
   help(string("-h,--help"), false,
	string("Display this message"),
	false, no_argument),
   verbose(string("-v,--verbose"), false,
	string("Switch on diagnostic messages\n\n"),
	false, no_argument),

   inmatrix2(string("-i,--in"), string("fdt_matrix2"),
	     string("\tInput matrix (default='fdt_matrix2.dot')"),
	     false, requires_argument),  
   obasename(string("-b,--basename"), string(""),
	     string("Output basename"),
	     true, requires_argument),
   ptxdir(string("-d,--dir"), string(""),
	  string("Tractography Results Directory\n\n"),
	  true, requires_argument),

   reord1(string("--r1"), false,
	     string("\tDo seedspace reordering (default no)"),
	     false, no_argument), 
   reord2(string("--r2"), false,
	     string("\tDo tractspace reordering (default no)"),
	     false, no_argument), 
   reord3(string("--tractreord"), false,
	     string("Propagate seed reordering onto tract space"),
	     false, no_argument), 

   excl_mask(string("--excl"), string(""),
	     string("\tExclusion mask (in tract space, i.e. columns of the matrix)\n"),
	     false, requires_argument),  
   connexity(string("--con"), 0.0,
	     string("\tAdd connexity constraint - value between 0 and 1 (0 is no constraint). default=0"),
	     false, requires_argument), 
   bin(string("--bin"), -1, 
	 string("\tBinarise at (default -1 - no binarisation)"), 
	 false, requires_argument),
   power(string("--power"), 1, 
	 string("\tPower to raise the correlation matrix to (default 1)\n\n"), 
	 false, requires_argument),

   mask(string("--seed"), "", 
	 string("\tSeed used for tractography (required to output the actual clustering)"), 
	 false, requires_argument),
   scheme(string("--scheme"), "spectral", 
	 string("Reordering algorithm. Can be either 'spectral' (default) or 'kmeans' or 'fuzzy'"), 
	 false, requires_argument),
   nclusters(string("--nclusters"), -1, 
	  string("Number of clusters (required if scheme is kmeans or fuzzy)"), 
	  false, requires_argument),
   options("ccops - reordeing of probtrackx matrix output","ccops -b <outputbasename> -d <probtrackxDirectory> [options]")
   {
     
    
     try {
       options.add(help);
       options.add(verbose);

       options.add(inmatrix2);
       options.add(obasename);
       options.add(ptxdir);

       options.add(reord1);
       options.add(reord2);
       options.add(reord3);

       options.add(excl_mask);
       options.add(connexity);
       options.add(bin);
       options.add(power);

       options.add(mask);
       options.add(scheme);
       options.add(nclusters);
       
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





