/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include "ptx_twomasks.h"
#include "streamlines.h"

using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;



void twomasks()
{ 
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  ////////////////////////////////
  //  Log& logger = LogSingleton::getInstance();
  volume<float> seeds,seeds2;
  read_volume(seeds,opts.seedfile.value());
  read_volume(seeds2,opts.mask2.value());

  if(opts.s2tout.value()){
    cerr << "Seed_to_target not available in multiple seed tractography" << endl;
    exit(0);
  }
  // correct for non-1 values
  seeds=NEWIMAGE::abs(seeds);
  seeds.binarise(0,seeds.max()+1,exclusive);
  seeds2=NEWIMAGE::abs(seeds2);
  seeds2.binarise(0,seeds2.max()+1,exclusive);

  int numseeds=0;
  for(int z=0;z<seeds.zsize();z++)
    for(int y=0;y<seeds.ysize();y++)
      for(int x=0;x<seeds.xsize();x++)
	if(seeds(x,y,z)!=0)numseeds++;	


  Streamliner stline(seeds);
  Counter counter(seeds,stline,numseeds);
  counter.initialise();
  Seedmanager seedmanager(counter);
  
  vector<int> keeptotal(2);

  stline.add_waymask(seeds2);
  keeptotal[0] = 0;
  cout << "tracking from first seed" << endl;
  for(int z=0;z<seeds.zsize();z++){
    for(int y=0;y<seeds.ysize();y++){
      for(int x=0;x<seeds.xsize();x++){
	if(seeds(x,y,z)>0){
	  keeptotal[0] += seedmanager.run(x,y,z,false,-1); 
	}
      }
    }
  }
  stline.pop_waymasks();
  
  stline.add_waymask(seeds);
  keeptotal[1] = 0;
  cout << "tracking from second seed" << endl;
  for(int z=0;z<seeds2.zsize();z++){
    for(int y=0;y<seeds2.ysize();y++){
      for(int x=0;x<seeds2.xsize();x++){
	if(seeds2(x,y,z)>0){	  
	  keeptotal[1] += seedmanager.run(x,y,z,false,-1); 
	  //keeptotal[1] += seedmanager.run(x,y,z,false,seeds2(x,y,z)-1); 
	}
      }
    }
  }
  

  counter.save_total(keeptotal);
  counter.save();
  cout<<"finished"<<endl;
}


