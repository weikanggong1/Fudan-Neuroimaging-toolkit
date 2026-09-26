/*  probtrackxOptions.cc

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2010 University of Oxford  */

/*  CCOPYRIGHT  */

#define WANT_STREAM
#define WANT_MATH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "probtrackxOptions.h"
#include "utils/options.h"
//#include "newmat.h"
using namespace Utilities;

namespace TRACT {

probtrackxOptions* probtrackxOptions::gopt = NULL;

  void probtrackxOptions::parse_command_line(int argc, char** argv, Log& logger)
  {
    //Do the parsing;
    try{
      for(int a = options.parse_command_line(argc, argv); a < argc; a++) ;
      // setup logger directory
      // if( mode.value()=="help"){
// 	modehelp();
// 	exit(2);
//       }
      if(help.value() || ! options.check_compulsory_arguments())
	{
	  options.usage();
	  exit(2);
	}   

      else{
	//modecheck(); // check all the correct options are set for this mode.	  
	if(forcedir.value())
	  logger.setthenmakeDir(logdir.value(),"probtrackx.log");
	else
	  logger.makeDir(logdir.value(),"probtrackx.log");
	
	cout << "Log directory is: " << logger.getDir() << endl;
	
	// do again so that options are logged
	for(int a = 0; a < argc; a++)
	  logger.str() << argv[a] << " ";
	logger.str() << endl << "---------------------------------------------" << endl << endl;	
	
      }
    
      
    }
    catch(X_OptionError& e){
      cerr<<e.what()<<endl;
      cerr<<"try: probtrackx2 --help"<<endl;
      exit(2);
    }
    
    
    
    
  }
  
  
  void probtrackxOptions::modecheck()
  {
//     bool check=true;
//     string mesg="";
//     if(mode.value()=="simple"){
//       if(outfile.value()==""){
// 	mesg+="You must set an output name in simple mode: -o\n";
// 	check=false;
//       }
//     }
    
    
//     cerr<<mesg;
//     exit(2);
  }
  
  
  
  void probtrackxOptions::status()
  {
    cout<<"basename   "<<basename.value()<<endl;
    cout<<"maskfile   "<<maskfile.value()<<endl;
    cout<<"seeds      "<<seedfile.value()<<endl;
    cout<<"output     "<<outfile.value()<<endl;
    cout<<"verbose    "<<verbose.value()<<endl;
    cout<<"nparticles "<<nparticles.value()<<endl;
    cout<<"nsteps     "<<nsteps.value()<<endl;
    cout<<"usef       "<<usef.value()<<endl;
    cout<<"rseed      "<<rseed.value()<<endl; 
    cout<<"randfib    "<<randfib.value()<<endl;
    cout<<"fibst      "<<fibst.value()<<endl;
}
  
}










