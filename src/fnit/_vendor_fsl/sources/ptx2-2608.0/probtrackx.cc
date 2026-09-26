/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include "newimage/newimageall.h"
#include "utils/log.h"
#include "probtrackx.h"
#include "utils/tracer_plus.h"
#include "stdlib.h"

using namespace std;
using namespace NEWIMAGE;
using namespace Utilities;
using namespace TRACT;


int main ( int argc, char **argv ){
  //Tracer_Plus::settimingon();

  probtrackxOptions& opts =probtrackxOptions::getInstance();
  Log& logger = LogSingleton::getInstance();
  opts.parse_command_line(argc,argv,logger);
  srand(opts.rseed.value());
  
  if(opts.verbose.value()>0){
    opts.status();
  }
  //Check the correct use of matrices
  if(opts.matrix2out.value() && opts.lrmask.value()==""){
    cerr<<"Error: --target2 must be specified when --omatrix2 is requested"<<endl;
    exit(1);
  }
  if(opts.matrix3out.value() && opts.mask3.value()==""){
    cerr<<"Error: --target3 (or --target3 and --lrtarget3) must be specified when --omatrix3 is requested"<<endl;
    exit(1);
  }
  if(opts.matrix4out.value() && opts.dtimask.value()==""){
    cerr<<"Error: --target4 (or --target4 and --colmask4) must be specified when --omatrix4 is requested"<<endl;
    exit(1);
  }
  if(opts.s2tout.value() && opts.targetfile.value()==""){
    cerr<<"Error: --targetmasks must be specified when --os2t is requested"<<endl;
    exit(1);
  }
  if(opts.simple.value()){
    if( opts.matrix1out.value() || opts.matrix3out.value()){
      cerr<<"Error: cannot use matrix1 and matrix3 in simple mode"<<endl;
      exit(1);
    }
    cout<<"Running in simple mode"<<endl;
    track();
  }
  else{
    if(!opts.network.value()){
      cout<<"Running in seedmask mode"<<endl;
      seedmask();
    }
    else{
      cout<<"Running in network mode"<<endl;
      nmasks();
    }
  }

  //Tracer_Plus::dump_times(logger.getDir());

  return 0;
}















