/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>
#include <algorithm>
#include "newimage/newimageall.h"


using namespace std;
using namespace NEWIMAGE;

int main(int argc, char **argv){
  
  if(argc!=3){
    cerr<<"Usage: medianfilter <input> <output>"<<endl;
    cerr<<"performs 26 neighbourhood median filtering"<<endl;
    exit(1);
  }
  volume<float> volin,volout;
  read_volume(volin,argv[1]);
  volout.reinitialize(volin.xsize(),volin.ysize(),volin.zsize());
  copybasicproperties(volin,volout);
  vector<float> vec(27);
  
  for(int z=volin.minz();z<=volin.maxz();z++){
    for(int y=volin.miny();y<=volin.maxy();y++){
      for(int x=volin.minx();x<=volin.maxx();x++){
	int count=0;

	for(int k=-1;k<=1;k++){
	  for(int j=-1;j<=1;j++){
	    for(int i=-1;i<=1;i++){
	      vec[count]=volin(x+i,y+j,z+k);
	      count++;
	    }
	  }
	}
	
	sort(vec.begin(),vec.end());
	volout(x,y,z)=vec[13];
	
      }
      
    }
  }
  save_volume(volout,argv[2]);
  return 0;
}
