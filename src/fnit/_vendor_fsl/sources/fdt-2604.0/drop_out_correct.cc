/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include "newimage/newimageall.h"
#include <vector>
using namespace std;
using namespace NEWIMAGE;

int main ( int argc, char **argv ){
  if(argc<7){
    cerr<<"usage: dropout_correct <reference> <brain_mask> <output_basename> <bvals> <%thresh> <avg1> <avg2> ..."<<endl;
    exit(0);
  }
  
  vector<volume4D<float> > imvec;
  imvec.reserve(argc-6);
  
  vector<volume4D<float> > outvec;
  outvec.reserve(argc-6);
  
  volume4D<float> tmpim;
  cout<<"number of averages "<<argc-6<<endl;
  
//    for (int i=0;i<argc;i++){
//      cerr<<argv[i]<<endl;
   
//    }

  for(int i=6;i<argc;i++){
    read_volume4D(tmpim,argv[i]);
    cerr<<argv[i]<<" "<<"numvols "<<tmpim.tsize()<<endl;
    imvec.push_back(tmpim);
    outvec.push_back(tmpim);
    for(int j=tmpim.tsize();j>0;j--){
      tmpim.deletevolume(j-1);
    }
    
  }
  int ipthr=atoi(argv[5]);
  Matrix bvals;
  bvals=read_ascii_matrix(argv[4]);
  
  volume<float> ref;
  read_volume(ref,argv[1]);
  volume<float> mask;
  read_volume(mask,argv[2]);
  vector<bool> tmp(imvec.size());
  volume4D<float> median;
  int ok_count=0,buggered_count=0,vox_count=0,sum=0;
  vector<float> slice_tot(imvec.size());
  vector<float> slice_mean(imvec.size());
  float ref_slice_tot=0,ref_slice_mean=0;
  cerr<<"numvols "<<imvec[0].tsize()<<endl;
  for(int t=0;t<imvec[0].tsize();t++){
    cout<<"Volume "<<t<<endl;
    for(int z=0;z<imvec[0].zsize();z++){
      //initialise slice specific variables
      vox_count=0;
      ref_slice_tot=0;
      for(unsigned int i=0;i<imvec.size();i++){
	slice_tot[i]=0;
      }
  

      //add up all the values in the slice for every avg and the ref.
      for(int y=0;y<imvec[0].ysize();y++){
	for(int x=0;x<imvec[0].xsize();x++){
	  if(mask(x,y,z)>0){
	    vox_count++;
	    ref_slice_tot+=ref(x,y,z);
	    for(unsigned int i=0;i<imvec.size();i++){
	      slice_tot[i]+=imvec[i](x,y,z,t);
	    }
	    
	  }
	  
	}
	
      }
      
     
      if(vox_count>0){
	 //compute all the means
	for(unsigned int i=0;i<imvec.size();i++){
	  slice_mean[i]=slice_tot[i]/vox_count;
	}
	ref_slice_mean=ref_slice_tot/vox_count;              
	//do the testing;
	ok_count=0;
	buggered_count=0;
	for(unsigned int i=0;i<imvec.size();i++){
	  float thr=ref_slice_mean*exp(-bvals(1,t+1)/1000)*ipthr/100.0f;
	  //	  	  cerr<<t<<" "<<z<<" threshold "<<thr<<endl;
	  //	  cerr<<slice_mean[i]<<endl;
	  if(slice_mean[i]>thr){
	    ok_count++;
	    tmp[i]=true;
	  }
	  else{
	    tmp[i]=false;
	    buggered_count++;
	    cerr<<"avg "<<i<<" Vol "<<t+1<<" slice "<<z<<" rejected"<<endl;
	  }
	 
	}
	// If they all failed
	if(ok_count==0){
	  cout<<"Volume "<<t+1<<", Slice "<<z<<" failed in all averages"<<endl;
	  //	  return 0;
	}
	else{
	//replace failed slices
	
	for(unsigned int i=0;i<imvec.size();i++){
	  for(int y=0;y<imvec[0].ysize();y++){
	    for(int x=0;x<imvec[0].xsize();x++){
	      if(tmp[i]){
		outvec[i](x,y,z,t)=imvec[i](x,y,z,t);
	      }
	      else{
		//		outvec[i](x,y,z,t)=0;
		float tmpvalue=0;
		for(unsigned int j=0;j<imvec.size();j++){
		  if(tmp[j]){
		    tmpvalue+=outvec[j](x,y,z,t);
		    //		    if(t==6&&z==36&&x==59&y==35){
		    //		      cerr<<j<<" "<<outvec[j](x,y,z,t)<<" "<<ok_count<<" "<<tmpvalue<<endl;
		    //		    }

		  }
		  outvec[i](x,y,z,t)=tmpvalue/(float)ok_count;
		}
//  		if(t==6&&z==36&&x==59&y==35){
//  		  cerr<<outvec[i](x,y,z,t)<<endl;
//  		  return 0;
//  		}
				  
	      }
	      
	      
	    }
	  }
	}
      }
      }
      
      
    }
  }

  
  for(unsigned int i=0;i<imvec.size();i++){
    string oname=argv[3];
    make_basename(oname);
    oname=oname+"_"+num2str(i+1);
    save_volume4D(outvec[i],oname);
  }
  
  
}





















