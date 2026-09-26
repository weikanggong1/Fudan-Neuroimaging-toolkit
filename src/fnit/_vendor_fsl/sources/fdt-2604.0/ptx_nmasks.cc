/*  Copyright (C) 2007 University of Oxford  */

/*  CCOPYRIGHT  */

#include "ptx_nmasks.h"
#include "streamlines.h"

using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;



void nmasks()
{ 
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  ////////////////////////////////
  //  Log& logger = LogSingleton::getInstance();
  vector< volume<float> > seeds;
  volume<float> tmpvol;
  vector<string> masknames;

  if(fsl_imageexists(opts.seedfile.value())){
    cerr << "Seed file must be a text file with a list of seeds in multiple masks mode" << endl;
    exit(0);
  }
  else
    read_masks(masknames,opts.seedfile.value());

  if(opts.s2tout.value()){
    cerr << "Seed_to_target not available in multiple seed tractography" << endl;
    exit(0);
  }

  for(unsigned int m=0;m<masknames.size();m++){
    read_volume(tmpvol,masknames[m]);
    tmpvol=NEWIMAGE::abs(tmpvol);
    tmpvol.binarise(0,tmpvol.max()+1,exclusive);
    seeds.push_back(tmpvol);
  }
  
  int numseeds=0;
  for(int z=0;z<seeds[0].zsize();z++)
    for(int y=0;y<seeds[0].ysize();y++)
      for(int x=0;x<seeds[0].xsize();x++)
	if(seeds[0](x,y,z)!=0)numseeds++;	


  Streamliner stline(seeds[0]);
  Counter counter(seeds[0],stline,numseeds);
  counter.initialise();
  Seedmanager seedmanager(counter);
  

  vector<int> keeptotal(seeds.size());
  for(unsigned int i=0;i<seeds.size();i++){ 

    // add all other seeds as waypoint masks
    tmpvol=0;
    for(unsigned int j=0;j<seeds.size();j++)
      if(j!=i)
	tmpvol+=seeds[j];
    if(i>=1)
      stline.pop_waymasks();
    stline.add_waymask(tmpvol);

    // start tracking
    cout << "Tracking from mask " << i+1 << endl;
    keeptotal[i] = 0;
    for(int z=0;z<seeds[i].zsize();z++){
      for(int y=0;y<seeds[i].ysize();y++){
	for(int x=0;x<seeds[i].xsize();x++){
	  if(seeds[i](x,y,z)!=0){
	    keeptotal[i] += seedmanager.run(x,y,z,false,-1); 
	  }
	}
      }
    } 
  }
  
  counter.save_total(keeptotal);
  counter.save();
  
  cout<<"finished"<<endl;
}


