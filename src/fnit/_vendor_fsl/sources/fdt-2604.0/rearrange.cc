/*  Copyright (C) 2010 University of Oxford  */
/*  Stam Sotiropoulos  */
/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>


#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

//Rearranges a dataset according to the angular distance between the corresponding bvecs entry and a reference vector
int main ( int argc, char *argv[]){
  if(argc<5 || argc>6){
    cerr<<" "<<endl;
    cerr<<"usage: rearrange data bvecs ref_dyads [brain_mask] output"<<endl<<endl;
    cerr<<"Rearranges a dataset according to the angular distance"<<endl;
    cerr<<"between the corresponding bvecs entry and a reference vector"<<endl;
    cerr<<" "<<endl;
    exit(1);
  }

  ColumnVector ref_vector(3);
  Matrix bvecs;   //Read Input
  volume4D<float> data, ref; volume<float> mask;

  read_volume4D(data,argv[1]);
  bvecs=read_ascii_matrix(argv[2]);
  read_volume4D(ref,argv[3]);
  if (argc==6)
    read_volume(mask,argv[4]);
  else{
    mask.reinitialize(data.xsize(),data.ysize(),data.zsize());
    mask=1;
  }

  volume4D<float> output(data.xsize(),data.ysize(),data.zsize(),data.tsize());
  copybasicproperties(data,output);output=0;

  if(bvecs.Nrows()>3) bvecs=bvecs.t();   //Make sure the bvecs entries are normalized unit vectors
  for(int i=1;i<=bvecs.Ncols();i++){
    float tmpsum=sqrt(bvecs(1,i)*bvecs(1,i)+bvecs(2,i)*bvecs(2,i)+bvecs(3,i)*bvecs(3,i));
    if(tmpsum!=0){
      bvecs(1,i)=bvecs(1,i)/tmpsum;
      bvecs(2,i)=bvecs(2,i)/tmpsum;
      bvecs(3,i)=bvecs(3,i)/tmpsum;
    }
  }
  for(int z=data.minz();z<=data.maxz();z++){   //Rearrange data for each voxel
    for(int y=data.miny();y<=data.maxy();y++){
      for(int x=data.minx();x<=data.maxx();x++){
	if (mask(x,y,z)!=0){
	  vector<pair<float,int> > dot;  //Keep in the first entry of each pair the dot product value and the in the second the index
	  pair<float,int> ftmp;
	  ref_vector(1)=ref(x,y,z,0); 	  ref_vector(2)=ref(x,y,z,1); 	  ref_vector(3)=ref(x,y,z,2);
	  for (int n=1; n<=bvecs.Ncols(); n++){     //Get the dot product of each bvecs entry with the reference orientation in this voxel
	    if (bvecs(1,n)==0 && bvecs(2,n)==0 && bvecs(3,n)==0)
	      ftmp.first=2.0;              //The b=0 entries are first in the sequence
	    else
	      ftmp.first=fabs(ref_vector(1)*bvecs(1,n)+ref_vector(2)*bvecs(2,n)+ref_vector(3)*bvecs(3,n));
	    ftmp.second=n-1;
	    dot.push_back(ftmp);
	    }
	  sort(dot.begin(),dot.end());     //Sort the dot products in ascending order
	  reverse(dot.begin(),dot.end());  //Reverse the ordering, so that it is in descending order
	  for (int n=0; n<bvecs.Ncols(); n++)//Get the dot product of each bvecs entry with the reference orientation in this voxel
	    output(x,y,z,n)=data(x,y,z,dot[n].second);
	}
      }
    }
    cout<<z+1<<" slices processed"<<endl;
  }
  save_volume4D(output,argv[argc-1]);
  return 0;
}
