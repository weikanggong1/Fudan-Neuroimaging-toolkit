/*  Log.cc

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include "Log.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>

using namespace std;
namespace UTILS {

  Log* Log::logger = NULL;

  int Log::establishDir(const string& name)
    {
      dir = name;
      int ret;
      // make directory to place results into:

      while(true)
	{
	  if ( (ret = mkdir(dir.c_str(), 0770))!= 0){
	    switch(errno){
	    case EACCES: // don't have write permission
	      return 1;
	    case EEXIST: // directory already exists - keep adding "+" until directory is made:
	      dir = dir + "+";
	      break;
	    default:
	      cerr << "Error!: Unknown Log directory return value" << endl;
	      exit(0);
	    }
	  } else {
	    // setup logfile
	    logfileout.open((dir + "/" + logfilename).c_str(), ios::out);
	    return 0;
	  }
   	}
      return 0;
    }


void Log::setDir(const string& name)
    {
      dir = name;

      // cerr << "Results are in directory " + dir << endl;

      // setup logfile
      logfileout.open((dir + "/" + logfilename).c_str(), ios::out);
    }

}
