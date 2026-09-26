/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include "newimage/newimageall.h"
#include "utils/log.h"
#include "meshclass/meshclass.h"
#include "probtrackx.h"


using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;
//using namespace NEWMAT;
//////////////////////////
/////////////////////////



int main ( int argc, char **argv ){
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  Log& logger = LogSingleton::getInstance();
  opts.parse_command_line(argc,argv,logger);
  srand(opts.rseed.value());
  
  
  
  if(opts.verbose.value()>0){
    opts.status();
  }
  if(opts.mode.value()=="simple"){
    track();
    return 0;
  }

  string tmpin=opts.seedfile.value();
  if(fsl_imageexists(opts.seedfile.value())){ 
    if(fsl_imageexists(opts.mask2.value())){ twomasks();}
    else{ seedmask(); }
  }
  else if(opts.network.value()){ nmasks(); }
  else if(opts.meshfile.value()!=""){meshmask();}
  else {
    cerr << "No seed mask detected! exit without doing anything"<<endl;
    cerr << "If you are seeding from a list of coordinates, use --mode=simple"<<endl;
    return 0;
  };

  //else if(fopen(tmpin.c_str(),"r")!=NULL ){ track();}

  // else if(opts.mode.value()=="seeds_to_targets")
  //     seeds_to_targets();
  //   else if(opts.mode.value()=="seedmask")
  //     alltracts();
  //   else if(opts.mode.value()=="twomasks_symm")
  //     twomasks_symm();
  //   else if(opts.mode.value()=="waypoints")
  //     waypoints();
  //   else if(opts.mode.value()=="matrix1")
  //     matrix1();
  //   else if(opts.mode.value()=="matrix2"){
  //     if(opts.meshfile.value()=="")
  //       matrix2();
  //     else
  //       mesh_matrix2();
  //   }
  //   else if(opts.mode.value()=="maskmatrix")
  //     maskmatrix();
  //   else if(opts.mode.value()=="meshlengths")
  //     mesh_lengths();
  //else{
  //   cout <<"Invalid setting for option  mode -- try setting mode=help"<<endl;
  //}
  
  return 0;
}















