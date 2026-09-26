/*  ccopsOptions.cc

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 1999-2010 University of Oxford  */

/*  CCOPYRIGHT  */

#define WANT_STREAM
#define WANT_MATH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "ccopsOptions.h"
#include "utils/options.h"
using namespace Utilities;

namespace CCOPS {

ccopsOptions* ccopsOptions::gopt = NULL;

bool ccopsOptions::parse_command_line(int argc, char** argv)
{

  
  for(int a = options.parse_command_line(argc, argv); a < argc; a++)
    ;
  if(help.value() || ! options.check_compulsory_arguments())
    {
      options.usage();

      return false;
    }      
  return true;
}

}







