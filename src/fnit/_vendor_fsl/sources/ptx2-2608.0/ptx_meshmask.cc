
/*  Copyright (C) 2007 University of Oxford  */

/*  CCOPYRIGHT  */

#include "ptx_meshmask.h"
#include "streamlines.h"
using namespace std;
using namespace NEWIMAGE;
using namespace TRACT;
using namespace Utilities;
using namespace PARTICLE;
using namespace mesh;



void meshmask()
{ 
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  // load seed mesh
  cout<<"loading mesh files"<<endl;
  Mesh mseeds;
  mseeds.load(opts.meshfile.value());
  if(opts.meshspace.value()!="first")
    mseeds.load_fs_label(opts.seedfile.value());
  cout<<"mesh files loaded"<<endl;

  // internally create seed mask in voxel space
  volume<float> seeds;
  if(opts.seedref.value()!="")
    read_volume(seeds,opts.seedref.value());
  else
    read_volume(seeds,opts.maskfile.value());
  seeds=0;


  
  Matrix mm_to_vox(4,4);
  if(opts.meshspace.value()=="freesurfer"){
    mm_to_vox << -1 << 0 << 0 <<  (seeds.xsize())/2
	      <<  0 << 0 << -1 << (seeds.zsize())/2
	      <<  0 << 1 << 0 <<  (seeds.ysize())/2
	      <<  0 << 0 << 0 << 1;
  }
  else if(opts.meshspace.value()=="caret"){
    cerr<<"caret not supported at the moment"<<endl;
    exit(1);
    //mm_to_vox = seeds.sform_mat();
    //mm_to_vox = mm_to_vox.i();
  }
  else if(opts.meshspace.value()=="first"){
    mm_to_vox << 1.0/seeds.xdim() << 0 << 0 << 0
	      << 0 << 1.0/seeds.ydim() << 0 << 0
	      << 0 << 0 << 1.0/seeds.zdim() << 0
	      << 0 << 0 << 0 << 1;
  }
  else{
    cerr << "unkown mesh option - exit without doing anything" << endl;
    cerr << "mesh option is either 'freesurfer' or 'caret'" << endl;
    cerr << "this is because each surface vertex coordinates are relative to a specific " << endl;
    cerr << "coordinate system which needs to be transformed into voxel space" << endl;
    exit(1);
  }


  ColumnVector fs_coord_mm(4),xyz_vox,seeddim(3);
  seeddim << seeds.xdim() << seeds.ydim() << seeds.zdim();
  ColumnVector dir(3);
  int keeptotal=0;

  // first fill seed with ones
  seeds = 0;
  int numseeds=0;
  for(vector<Mpoint*>::iterator i = mseeds._points.begin();i!=mseeds._points.end();i++){
    if((*i)->get_value() > 0 || opts.meshspace.value()=="first"){
    
      fs_coord_mm<<(*i)->get_coord().X<<(*i)->get_coord().Y<<(*i)->get_coord().Z << 1.0; 
      xyz_vox = mm_to_vox*fs_coord_mm;


      
      float x=xyz_vox(1);float y=xyz_vox(2);float z=xyz_vox(3);
      Pt newPt(x,y,z);
      (*i)->_update_coord = newPt;

      seeds(int(round(x)),int(round(y)),int(round(z))) = 1;
      numseeds++;
    }
  }

  
  ////////////////////////////////
  //  Log& logger = LogSingleton::getInstance();
  Streamliner stline(seeds);
  Counter counter(seeds,stline,numseeds);
  counter.initialise();
  Seedmanager seedmanager(counter);


  for(vector<Mpoint*>::iterator i = mseeds._points.begin();i!=mseeds._points.end();i++){
    if((*i)->get_value() > 0 || opts.meshspace.value()=="first"){
    
      fs_coord_mm<<(*i)->get_coord().X<<(*i)->get_coord().Y<<(*i)->get_coord().Z << 1.0; 
      //      xyz_vox = seeds.qform_mat().i()*fs_coord_mm
            xyz_vox = mm_to_vox*fs_coord_mm;
	    //xyz_vox = fs_coord_mm;
      

      float x=xyz_vox(1);float y=xyz_vox(2);float z=xyz_vox(3);
      Pt newPt(x,y,z);
      (*i)->_update_coord = newPt;

		//      seeds(round(x),round(y),round(z)) = 1;
    
      cout <<"run"<<endl;
      dir << (*i)->local_normal().X << (*i)->local_normal().Y << (*i)->local_normal().Z;
 
      if(opts.meshspace.value()=="first")
	dir*=-1.0;

     keeptotal += seedmanager.run(x,y,z,true,-1,dir); 
	
    }
  }
  mseeds.update();
  //  mseeds.save("test.vtk",3);

  //return;
  
  //   for(int z=0;z<seeds.zsize();z++){
  //     cout <<"sl "<<z<<endl;
  //     for(int y=0;y<seeds.ysize();y++){
  //       for(int x=0;x<seeds.xsize();x++){
  // 	if(seeds(x,y,z)>0){
  // 	  cout <<"run"<<endl;
  // 	  dir << (*i)->local_normal().X << (*i)->local_normal().Y << (*i)->local_normal().Z;
  // 	  keeptotal += seedmanager.run(x,y,z,true,-1,dir); 
  // 	}
  //       }
  //     }
  //   }
  
  counter.save_total(keeptotal);  
  counter.save();
  
  cout<<"finished"<<endl;
}


