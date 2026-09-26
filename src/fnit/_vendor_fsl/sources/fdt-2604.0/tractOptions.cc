/*  dtiOptions.cc

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#define WANT_STREAM
#define WANT_MATH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "tractOptions.h"
#include "utils/options.h"

using namespace Utilities;

namespace TRACT {

tractOptions* tractOptions::gopt = NULL;

  void tractOptions::parse_command_line(int argc, char** argv)
  {
    //Do the parsing;
    try{
      for(int a = options.parse_command_line(argc, argv); a < argc; a++) ;
    }
    catch(X_OptionError& e){
      cerr<<e.what()<<endl;
      cerr<<"try: tract2 --help"<<endl;
      exit(0);
    }


    if(help.value() || ! options.check_compulsory_arguments())
      {
	options.usage();
	exit(2);
      }

  }

  void tractOptions::status()
  {
    cerr<<"basename   "<<basename.value()<<endl;
    cerr<<"maskfile   "<<maskfile.value()<<endl;
    cerr<<"seeds      "<<seedfile.value()<<endl;
    cerr<<"output     "<<outfile.value()<<endl;
    cerr<<"verbose    "<<verbose.value()<<endl;
    cerr<<"nparticles "<<nparticles.value()<<endl;
    cerr<<"nsteps     "<<nsteps.value()<<endl;
    cerr<<"usef       "<<usef.value()<<endl;
    cerr<<"rseed      "<<rseed.value()<<endl;
  }

}
