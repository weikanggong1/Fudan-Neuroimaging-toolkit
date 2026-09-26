/*  Copyright (C) 2010 University of Oxford  */
/*  Stam Sotiropoulos, Saad Jbabdi    */
/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

//Computes the model 1 predicted signal from the mode of the bedpostx samples
//An f0 compartment could also be included in the model
int main ( int argc, char *argv[]){
  if(argc<3 || argc>4){
    cerr<<" "<<endl;
    cerr<<"usage: xfibres_pred bpx_dir [data_file] output"<<endl<<endl;
    cerr<<"       bpx_dir is the bedpostx output directory "<<endl;
    cerr<<"       data_file is optional, in case no mean_S0samples exists in the bpx dir, get the mean S0 from the data "<<endl;
    cerr<<" "<<endl;
    exit(1);
  }

  Matrix bvecs,bvals;   //Read Input
  volume<float> mask;
  volume<float> d,d_std,S0,f0,R, temp;
  volume4D<float> temp4D;
  vector< volume<float> > f;
  vector< volume4D<float> > dyads;

  string dir_name=argv[1], temp_name;
  temp_name=dir_name+"/bvals";
  bvals=read_ascii_matrix(temp_name);
  temp_name=dir_name+"/bvecs";
  bvecs=read_ascii_matrix(temp_name);

  if(bvecs.Nrows()>3) bvecs=bvecs.t();   //Make sure the bvecs entries are normalized unit vectors
  if(bvals.Nrows()>1) bvals=bvals.t();
  for(int i=1;i<=bvecs.Ncols();i++){
    float tmpsum=sqrt(bvecs(1,i)*bvecs(1,i)+bvecs(2,i)*bvecs(2,i)+bvecs(3,i)*bvecs(3,i));
    if(tmpsum!=0){
      bvecs(1,i)=bvecs(1,i)/tmpsum;
      bvecs(2,i)=bvecs(2,i)/tmpsum;
      bvecs(3,i)=bvecs(3,i)/tmpsum;
    }
  }

  int num_fibres=0;  //Check how many fibres exist
  while (fsl_imageexists(dir_name+"/dyads"+num2str(num_fibres+1)))
    num_fibres++;

  temp_name=dir_name+"/mean_dsamples";   //Read bedpostx results
  if (!fsl_imageexists(temp_name)){
    cout<<"No mean_dsamples file exists!"<<endl;
    exit(1);
  }
  else read_volume(d,temp_name);

  temp_name=dir_name+"/mean_S0samples";   //In case no S0samples is saved, use the mean b=0 from the data
  if (!fsl_imageexists(temp_name) && argc!=4){
    cerr<<"No mean_S0samples or data file exists!"<<endl;
    exit(1);
  }

  if (!fsl_imageexists(temp_name)){
    string data_name=argv[2];
    if (!fsl_imageexists(data_name)){
      cerr<<"Wrong data filename!"<<endl;
      exit(1);
    }
    else{
      cout<<"No mean_SOsamples found, getting the average of b=0's from data..."<<endl;
      volume4D<float> data;
      read_volume4D(data,data_name);
      S0.reinitialize(data.xsize(),data.ysize(),data.zsize());
      S0=0;
      int b0count=0;
      for (int l=1; l<=bvals.Ncols(); l++)
	if (bvals(1,l)>=0 && bvals(1,l)<=50){  //Treat as b=0, volumes with b up to 50 (imaging gradients contribution)
	  S0+=data[l-1];
	  b0count++;
	}
      S0/=b0count;
    }
  }
  else read_volume(S0,temp_name);

  for (int n=0; n<num_fibres; n++){   //Read dyads
    temp_name=dir_name+"/dyads"+num2str(n+1);
    if (!fsl_imageexists(temp_name)){
      cerr<<"No dyads"<<n+1<<" file exists!"<<endl;
      exit(1); }
    else{
      read_volume4D(temp4D,temp_name);
      dyads.push_back(temp4D);
    }
    temp_name=dir_name+"/mean_f"+num2str(n+1)+"samples";
    if (!fsl_imageexists(temp_name)){
      cerr<<"No mean_f"<<n+1<<"samples file exists!"<<endl;
      exit(1); }
    else{
      read_volume(temp,temp_name);
      f.push_back(temp);
    }
  }

  int modelnum=1;
  temp_name=dir_name+"/mean_d_stdsamples";
  if (fsl_imageexists(temp_name)){   //Read d_std if model2
    modelnum=2;
    read_volume(d_std,temp_name);
  }
  temp_name=dir_name+"/mean_Rsamples";
  if (fsl_imageexists(temp_name)){   //Read R if model3
    modelnum=3;
    read_volume(R,temp_name);
  }


  int f0_incl=0;
  temp_name=dir_name+"/mean_f0samples";
  if (fsl_imageexists(temp_name)){
    f0_incl=1;
    read_volume(f0,temp_name);
  }

  cout<<"Files for model"<<modelnum<<" with "<<num_fibres<<" fibres found"<<endl;
  if (f0_incl==1)
    cout<<"Also an f0 noise floor is assumed"<<endl<<endl;

  temp_name=dir_name+"/nodif_brain_mask";
  if (fsl_imageexists(temp_name))
    read_volume(mask,temp_name);
  else{
    mask.reinitialize(d.xsize(),d.ysize(),d.zsize());
    mask=1;
  }

  volume4D<float> output;
  output.reinitialize(d.xsize(),d.ysize(),d.zsize(),bvals.Ncols());
  copybasicproperties(d,output);
  output.setdims(d.xdim(),d.ydim(),d.zdim(),1.0);
  output=0;

  for(int z=d.minz();z<=d.maxz();z++){   //Compute predicted signal for each voxel
    for(int y=d.miny();y<=d.maxy();y++){
      for(int x=d.minx();x<=d.maxx();x++){
	if (mask(x,y,z)!=0){
	  for (int l=0; l<bvals.Ncols(); l++){ //for each datapoint
	    //Aniso signal first
	    float sumf=0; float sig2=0; float dalpha=0;
	    if (modelnum==1 || (modelnum==2 && d_std(x,y,z)<=1e-5)){ //model1 or model2 with small dstd
	      sumf=0;
	      for (int n=0; n<num_fibres; n++){
		sumf+=f[n](x,y,z);
		float angp=dyads[n](x,y,z,0)*bvecs(1,l+1)+dyads[n](x,y,z,1)*bvecs(2,l+1)+dyads[n](x,y,z,2)*bvecs(3,l+1);
		output(x,y,z,l)+=f[n](x,y,z)*std::exp(-bvals(1,l+1)*d(x,y,z)*angp*angp);
	      }
	    }
	    else if (modelnum>=2) { //model2 or model3
	      sig2=d_std(x,y,z)*d_std(x,y,z);
	      dalpha=d(x,y,z)*d(x,y,z)/sig2;
	      sumf=0;
	      for (int n=0; n<num_fibres; n++){
		sumf+=f[n](x,y,z);
		float angp=dyads[n](x,y,z,0)*bvecs(1,l+1)+dyads[n](x,y,z,1)*bvecs(2,l+1)+dyads[n](x,y,z,2)*bvecs(3,l+1);
		if (modelnum==2)
		  output(x,y,z,l)+=f[n](x,y,z)*std::exp(dalpha*log(d(x,y,z)/(d(x,y,z)+bvals(1,l+1)*angp*angp*sig2)));
		if (modelnum==3)
		  output(x,y,z,l)+=f[n](x,y,z)*std::exp(-bvals(1,l+1)*3*d(x,y,z)/(2*R(x,y,z)+1.0)*((1-R(x,y,z))*angp*angp+R(x,y,z)));
	      }
	    }
	    //Iso signal Now
	    if (modelnum==1 || d_std(x,y,z)<=1e-5){
	      if (f0_incl==1)
		output(x,y,z,l)+=f0(x,y,z)+(1-sumf-f0(x,y,z))*std::exp(-bvals(1,l+1)*d(x,y,z));
	      else
		output(x,y,z,l)+=(1-sumf)*std::exp(-bvals(1,l+1)*d(x,y,z));
	    }
	    else{
	      if (f0_incl==1)
		output(x,y,z,l)+=f0(x,y,z)+(1-sumf-f0(x,y,z))*std::exp(dalpha*log(d(x,y,z)/(d(x,y,z)+bvals(1,l+1)*sig2)));
	      else
		output(x,y,z,l)+=(1-sumf)*std::exp(dalpha*log(d(x,y,z)/(d(x,y,z)+bvals(1,l+1)*sig2)));
	    }
	    output(x,y,z,l)*=S0(x,y,z);
	  }
	}
      }
    }
    cout<<z+1<<" slices processed"<<endl;
  }
  cout<<"saving results"<<endl;
  save_volume4D(output,argv[argc-1]);
  return 0;
}
