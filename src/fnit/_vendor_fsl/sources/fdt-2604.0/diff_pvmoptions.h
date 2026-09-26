/*  meanoptions.h

    Mark Woolrich - FMRIB Image Analysis Group

    Copyright (C) 2002 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(Diff_pvmOptions_h)
#define Diff_pvmOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "utils/log.h"
#include "bint/bintoptions.h"

using namespace Utilities;

namespace Bint {

class Diff_pvmOptions : public BintOptions {
 public:
  static Diff_pvmOptions& getInstance();
  ~Diff_pvmOptions() { delete gopt; }
  
  Option<string> bvecsfile;  
  Option<string> bvalsfile;
  
  
 private:
  Diff_pvmOptions();  
  const Diff_pvmOptions& operator=(Diff_pvmOptions&);
  Diff_pvmOptions(Diff_pvmOptions&);
      
  static Diff_pvmOptions* gopt;
  
};

 inline Diff_pvmOptions& Diff_pvmOptions::getInstance(){
   if(gopt == NULL)
     gopt = new Diff_pvmOptions();
   
   return *gopt;
 }

 inline Diff_pvmOptions::Diff_pvmOptions() :
   BintOptions("diff_pvm", "diff_pvm --verbose\n"),   
   bvecsfile(string("-r,--bvecs"),"bvecs", 
	     string("gradient directions"), 
	     true, requires_argument),
   bvalsfile(string("-b,--bvals"),"bvals", 
	     string("b values"), 
	     true, requires_argument)

   {
     try {
       options.add(bvecsfile);
       options.add(bvalsfile);
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





