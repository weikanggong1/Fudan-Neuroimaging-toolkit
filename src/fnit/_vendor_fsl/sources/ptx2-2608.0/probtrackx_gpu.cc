/*  probtrackx_gpu.cc

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>

#include "newimage/newimage.h"
#include "utils/log.h"
#include "probtrackx.h"
#include "saveResults_ptxGPU.h"
#include "CUDA/tractographyInput.h"
#include "CUDA/tractography_gpu.cuh"


using namespace std;
using namespace Utilities;
using namespace NEWIMAGE;
using namespace TRACT;

void tractography(){

  time_t _time;
  _time=time(NULL);

  probtrackxOptions& opts =probtrackxOptions::getInstance();

  ///////////////////////
  /////// RESULTS ///////
  ///////////////////////
  volume<float>*	m_prob=new volume<float>[1];
  volume<float>*	m_prob2=new volume<float>[1];	// omeanpathlength
  float**	   	ConNet;
  float**	   	ConNetb;
  int 			nRowsNet;
  int			nColsNet;
  float**		ConMat1;
  float**	   	ConMat1b;
  int 			nRowsMat1;
  int			nColsMat1;
  float**	     	ConMat3;
  float**	  	ConMat3b;
  int 			nRowsMat3;
  int			nColsMat3;
  float*		m_s2targets;	// All seeds to all Targets: Nseeds x NTargets
  float*	 	m_s2targetsb;
  vector< vector<float> > m_save_paths;
  volume4D<float>*	m_localdir=new volume4D<float>[1];	// opathdir

  //////////////////////////////
  /////// LOAD HOST DATA ///////
  //////////////////////////////
  tractographyData data_host;
  tractographyInput input;

  input.load_tractographyData(data_host,m_prob,m_prob2,
			      ConNet,ConNetb,nRowsNet,nColsNet,
			      ConMat1,ConMat1b,nRowsMat1,nColsMat1,
			      ConMat3,ConMat3b,nRowsMat3,nColsMat3,
			      m_s2targets,m_s2targetsb,m_localdir);
  //printf("SIZE MATRIX %i %i\n",ConMat3.Nrows(),ConMat3.Ncols());

  cout<<endl<<"Time Loading Data: "<<(time(NULL)-_time)<<" seconds"<<endl<<endl; _time=time(NULL);

  int* keeptotal;
  int num_keeptotal=1;
  if(opts.network.value()){
    keeptotal=new int[nRowsNet];
    for(int i=0;i<nRowsNet;i++) keeptotal[i]=0;
    num_keeptotal=nRowsNet;
  }else{
    keeptotal=new int[1];
    keeptotal[0]=0;
  }
  tractography_gpu(data_host,m_prob,m_prob2,keeptotal,
  		   ConNet,ConNetb,ConMat1,ConMat1b,ConMat3,ConMat3b,m_s2targets,m_s2targetsb,m_save_paths,m_localdir);
  cout<<endl<<"Time Spent Tracking:: "<<(time(NULL)-_time)<<" seconds"<<endl<<endl;

  /////////////////// FINISH ///////////////////
  //printf("keeptotal %i \n",keeptotal);
  // save results
  cout << "save results" << endl;
  counter_save_total(keeptotal,num_keeptotal);
  counter_save(data_host,m_prob,m_prob2,ConNet,ConNetb,nRowsNet,nColsNet,ConMat1,ConMat1b,nRowsMat1,nColsMat1,
	       ConMat3,ConMat3b,nRowsMat3,nColsMat3,m_s2targets,m_s2targetsb,m_save_paths,m_localdir);

  return;
}

int main ( int argc, char **argv ){

  printf("PROBTRACKX2 VERSION GPU\n");

  time_t _time;
  _time=time(NULL);

  probtrackxOptions& opts = probtrackxOptions::getInstance();
  Log& logger = LogSingleton::getInstance();
  opts.parse_command_line(argc,argv,logger);

  srand(opts.rseed.value());

  // THINGS NOT IMPLEMENTED:

  if(opts.pathfile.value()!=""){
    cerr<<"Error: option '--fopd' is not available in this version"<<endl;
    exit(1);
  }

  if(opts.matrix1out.value()&&opts.matrix3out.value()){
    cerr<<"Error: option '--omatrix1' and '--omatrix3' cannot be run simultaneously in this version"<<endl;
    exit(1);
  }

  if(opts.matrix4out.value()){
    cerr<<"Error: option '--omatrix4' is not available in this version"<<endl;
    exit(1);
  }

  // NOT IMPLEMENTED - hidden
  if(opts.prefdirfile.value()!=""){
    cerr<<"Error: option '--prefdir' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.skipmask.value()!=""){
    cerr<<"Error: option '--no_integrity' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.osampfib.value()){
    cerr<<"Error: option '--osampfib' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.onewayonly.value()){
    // Maybe useful when tracking from vertices, but need to calculate the normal (not common)
    cerr<<"Error: option '--onewayonly' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.locfibchoice.value()!=""){
    cerr<<"Error: option '--locfibchoice' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.loccurvthresh.value()!=""){
    cerr<<"Error: option '--loccurvthresh' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.targetpaths.value()){
    cerr<<"Error: option '--otargetpaths' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.noprobinterpol.value()){
    cerr<<"Error: option '--noprobinterpol' is not available in this version"<<endl;
    exit(1);
  }
  if(opts.closestvertex.value()){
    cerr<<"Error: option '--closestvertex' is not available in this version"<<endl;
    exit(1);
  }


  //Check the correct use of matrices
  if(opts.matrix2out.value() && opts.lrmask.value()==""){
    cerr<<"Error: --target2 must be specified when --omatrix2 is requested"<<endl;
    exit(1);
  }
  if(opts.matrix2out.value()){
    try{
      volume<float> vol;
      read_volume(vol,opts.lrmask.value());
    }catch(...){
      cerr<<endl<<"Error: target2 must be a single volume file"<<endl;
      exit(1);
    }

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
    cerr<<"Simple mode is not available in this version"<<endl;
    exit(1);
    // if implemented, take care with --os2t

  }else{
    if(!opts.network.value()){
      cout<<"Running in seedmask mode"<<endl;
      tractography();
    }else{
      cout<<"Running in network mode"<<endl;
      tractography();
    }
  }

  cout<<endl<<"TOTAL TIME: "<<(time(NULL)-_time)<<" seconds"<<endl<<endl;

  return 0;
}
