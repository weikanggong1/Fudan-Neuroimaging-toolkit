/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include "ptx_seedmask.h"
#include "streamlines.h"

using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;



void seedmask()
{ 
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  ////////////////////////////////
  //  Log& logger = LogSingleton::getInstance();
  volume<float> seeds;
  read_volume(seeds,opts.seedfile.value());
  seeds=NEWIMAGE::abs(seeds);
  seeds.binarise(0,seeds.max()+1,exclusive);
  int numseeds=0;
  for(int z=0;z<seeds.zsize();z++)
    for(int y=0;y<seeds.ysize();y++)
      for(int x=0;x<seeds.xsize();x++)
	if(seeds(x,y,z)!=0)numseeds++;



  Streamliner stline(seeds);
  Counter counter(seeds,stline,numseeds);
  counter.initialise();
  Seedmanager seedmanager(counter);


  int keeptotal=0;
  for(int z=0;z<seeds.zsize();z++){
    cout <<"sl "<<z<<endl;
    for(int y=0;y<seeds.ysize();y++){
      for(int x=0;x<seeds.xsize();x++){
	if(seeds(x,y,z)>0){
	  cout <<"run"<<endl;
	  keeptotal += seedmanager.run(x,y,z); 
	}
      }
    }
  }

  counter.save_total(keeptotal);  
  counter.save();

  cout<<"finished"<<endl;
}


