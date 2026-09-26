/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include "streamlines.h"
#include "ptx_simple.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;


void track(){
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  ////////////////////////////
  Log& logger = LogSingleton::getInstance();
  if(opts.verbose.value()>1){
    logger.makeDir("particles","particle0",true,false);
  }

  volume<short int> seedref;
  if(opts.seedref.value()!=""){
    read_volume(seedref,opts.seedref.value());
  }
  else{
    read_volume(seedref,opts.maskfile.value());
  }
  CSV csv(seedref);//we need this for seed2target

  Matrix Seeds = read_ascii_matrix(opts.seedfile.value());
  if(Seeds.Ncols()!=3 && Seeds.Nrows()==3)
    Seeds=Seeds.t();

  Streamliner   stline(csv);
  Counter       counter(stline);
  counter.forceNumSeeds(Seeds.Nrows()); // csv useless in this case
  counter.initialise();
  Seedmanager   seedmanager(counter);

  srand(opts.rseed.value()); // need to reinitialise random seed because of GIFTI!!

  // convert coordinates from nifti (external) to newimage (internal)
  //   conventions - Note: for radiological files this should do nothing
  Matrix newSeeds(Seeds.Nrows(),3);
  for (int n=1; n<=Seeds.Nrows(); n++) {
    ColumnVector v(4);
    v << Seeds(n,1) << Seeds(n,2) << Seeds(n,3) << 1.0;
    v = seedref.niftivox2newimagevox_mat() * v;
    newSeeds.Row(n) << v(1) << v(2) << v(3);
  }

  time_t _time;
  _time=time(NULL);

  int keeptot=0;
  vector<int> triangles;  //to avoid connections between vertices of the same triangle...but not used for volumes
  triangles.push_back(-1);
  for(int SN=1; SN<=newSeeds.Nrows();SN++){
    counter.updateSeedLocation(SN,-1,triangles);
    float xst=newSeeds(SN,1);
    float yst=newSeeds(SN,2);
    float zst=newSeeds(SN,3);
    keeptot += seedmanager.run(xst,yst,zst,false,-1,opts.sampvox.value());
    string add="_"+num2str(Seeds(SN,1))+"_"+num2str(Seeds(SN,2))+"_"+num2str(Seeds(SN,3));

    if(opts.simpleout.value()||opts.omeanpathlength.value())
      counter.save_pathdist(add);

    counter.reset_prob();
  } //Close Seed number Loop

  cout<<endl<<"time spent tracking: "<<(time(NULL)-_time)<<" seconds"<<endl<<endl;


  cout << "save results" << endl;
  counter.save();

  cout<<"finished"<<endl;
}
