/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <cmath>
#include "newimage/newimageall.h"
#include <vector>
#include <algorithm>

using namespace std;
using namespace NEWIMAGE;
using namespace NEWMAT;

string matf2coordf(string matf){
  size_t pos=matf.rfind("/");
  if(pos!=string::npos)
    matf.replace(pos,1,"/coords_for_");
  else
    matf="coords_for_"+matf;

  return matf;
}


int main ( int argc, char **argv ){
  if(argc<5){
    cout<<"usage: fdt_matrix_ops <matrix1> <seedmask1> <matrix2> <seedmask2> ... [inclusion mask] <output>"<<endl;
    cout<<"creates one big uber-matrix from lots of cute wee dinky ones"<<endl;
    cout<<"If seedmasks overlap, it just takes the values from the first of them"<<endl;
    cout<<"If you specify an incluson mask (optional), only voxels that are"<<endl;
    cout<<"inside this mask will be included in the output"<<endl;
    exit(0);
  }
  int var=argc-1;
  bool incmaskyn = (float(var)/2.0)==int(float(var)/2.0);
  int Nmats;
  if(incmaskyn) Nmats=var/2-1;
  else Nmats=(var-1)/2;
  
  vector<volume<int>* > masks;
  vector<volume<float>* > mats;
  vector<volume<int>* > coords;
  volume<int> incmask;
  volume<int> totalmask;
  for(int i=0;i<Nmats;i++){
    volume<float> *tmp= new volume<float>;
    volume<int> *tmpcoords= new volume<int>;
    volume<int> *tmpmask= new volume<int>;
    string matfile=string(argv[ 2*i + 1 ]);
    string coordfile=matf2coordf(matfile);
    read_volume(*tmp,matfile);
    read_volume(*tmpcoords,coordfile);
    read_volume(*tmpmask,argv[ 2*(i + 1) ]);
    mats.push_back(tmp);
    masks.push_back(tmpmask);
    coords.push_back(tmpcoords);
    if(i==0) totalmask=*tmpmask;
    else totalmask=totalmask+ *tmpmask;

}
  if(incmaskyn){
    read_volume(incmask,argv[var-1]);
    incmask=incmask*totalmask;
  }else{
    incmask=totalmask;
  }
  
  vector<volume<int>* >lookups;
  for(int i=0;i<Nmats;i++){
    volume<int> *lu = new volume<int>;
    *lu=*masks[i];int conrow=0;
    for(int z=0;z<(*lu).zsize();z++){
      for(int y=0;y<(*lu).ysize();y++){
	for(int x=0;x<(*lu).xsize();x++){
	  if((*masks[i])(x,y,z)>0){
	    (*lu)(x,y,z)=conrow;
	    conrow++;
	  }
	}
      }
    }
    lookups.push_back(lu);
  }
  
  int nvoxels=0;

  for(int z=0;z<incmask.zsize();z++) {
    for(int y=0;y<incmask.ysize();y++){
      for(int x=0;x<incmask.xsize();x++){
	  if(incmask(x,y,z)>0) nvoxels++;
      }
    }
  }
  
  volume<float> output(nvoxels,(*mats[0]).ysize(),1);
  volume<float> outcoords(nvoxels,3,1);
  int newrow=0;
  for(int z=0;z<incmask.zsize();z++) {
    for(int y=0;y<incmask.ysize();y++){
      for(int x=0;x<incmask.xsize();x++){
	if(incmask(x,y,z)>0){
	  bool found=false;
	  for(unsigned int i=0;i<mats.size();i++){
	    if(!found){
	      if((*masks[i])(x,y,z)>0){
		int oldrow=(*lookups[i])(x,y,z);
		for(int col=0;col<(*mats[i]).ysize();col++){
		  output(newrow,col,0)=(*mats[i])(oldrow,col,0);
		}
		//cout<<x<<" "<<y<<" "<<z<<endl;
		//cout<<newrow<<" "<<oldrow<<endl;
		//cout<<(*coords[i])(oldrow,0,0)<<" "<<(*coords[i])(oldrow,1,0)<<" "<<(*coords[i])(oldrow,2,0)<<endl;
		outcoords(newrow,0,0)=(*coords[i])(oldrow,0,0);
		outcoords(newrow,1,0)=(*coords[i])(oldrow,1,0);
		outcoords(newrow,2,0)=(*coords[i])(oldrow,2,0);
		//cout<<"yep"<<endl;
		found=true;
		newrow++;
	      }
	    }
	    
	  }
	  
	}
      }
    }
  }

  for(int i=0;i<Nmats;i++){
    delete mats[i];
    delete coords[i];
    delete masks[i];
    delete lookups[i];
}
  save_volume(output,argv[argc-1]);
  string coordout=matf2coordf(string(argv[argc-1]));
  save_volume(outcoords,coordout);
 return 0;
}
 


















