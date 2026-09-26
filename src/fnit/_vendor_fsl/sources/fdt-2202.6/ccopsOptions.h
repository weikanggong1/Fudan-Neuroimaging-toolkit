/*  BpmOptions.h

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

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
  
  Option<bool> help;
  Option<string> inmatrix;
  Option<string> basename;
  Option<string> directory;
  Option<string> excl_mask;
  Option<bool>  reord1;
  Option<bool>  reord2;
  Option<bool>  reord3;
  Option<float> connexity;
  Option<int>   bin;
  Option<float> power;
  Option<string> mask;
  Option<string> scheme;
  Option<int>    nclusters;
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
	string("display this message"),
	false, no_argument),
   inmatrix(string("-i,--in"), string("fdt_matrix2"),
	       string("input matrix"),
	       false, requires_argument),  
   basename(string("-b,--basename"), string(""),
	       string("Output basename"),
	       true, requires_argument),
   directory(string("-d,--dir"), string("."),
	       string("Tractography Results Directory"),
	       false, requires_argument),
   excl_mask(string("-x"), string(""),
	     string("exclusion mask (in tract space)"),
	     false, requires_argument),  
   reord1(string("--r1"), bool(false),
	     string("do seedspace reordering (default no)"),
	     false, no_argument), 
   reord2(string("--r2"), bool(false),
	     string("do tractspace reordering (default no)"),
	     false, no_argument), 
   reord3(string("--tractreord"), bool(false),
	     string("propagate seed reordering onto tract space"),
	     false, no_argument), 
   connexity(string("--con"), 0.0,
	     string("add connexity constraint - value between 0 and 1 (0 is no constraint). default=0"),
	     false, requires_argument), 
   bin(string("--bin"), 0, 
	 string("binarise at (default 0 - no binarisation)"), 
	 false, requires_argument),
   power(string("-p,--power"), 1, 
	 string("power to raise the correlation matrix to (default 1)"), 
	 false, requires_argument),
   mask(string("-m,--mask"), "", 
	 string("brain mask used to output the clustered roi mask (not necessary if --dir set)"), 
	 false, requires_argument),
   scheme(string("-s,--scheme"), "spectral", 
	 string("Reordering algorithm. Can be either spectral (default) or kmeans or fuzzy"), 
	 false, requires_argument),
   nclusters(string("-k,--nclusters"), 2, 
	  string("Number of clusters to be used in kmeans or fuzzy"), 
	  false, requires_argument),
   options("ccops","")
   {
     
    
     try {
       options.add(help);
       options.add(inmatrix);
       options.add(basename);
       options.add(directory);
       options.add(excl_mask);
       options.add(reord1);
       options.add(reord2);
       options.add(reord3);
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





