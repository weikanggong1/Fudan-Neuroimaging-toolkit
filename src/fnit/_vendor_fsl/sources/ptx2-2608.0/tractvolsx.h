/*  tractvolsx.h

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __TRACTVOLSX_H_
#define __TRACTVOLSX_H_

/////////////////////////////////////////////////////////
//         Class TractVolsx                             //
/////////////////////////////////////////////////////////

#include <iostream>
#include <cstdlib>

#include "utils/log.h"
#include "utils/tracer_plus.h"
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "probtrackxOptions.h"

namespace TRACTVOLSX{
  class Tractvolsx
    {
    private:
      TRACT::probtrackxOptions&   opts;
      Utilities::Log&             logger;

      std::vector<NEWMAT::Matrix> thsamples;
      std::vector<NEWMAT::Matrix> phsamples;
      std::vector<NEWMAT::Matrix> fsamples;

      NEWIMAGE::volume<int>       lut_vol2mat;

      int                         nfibres;
      int                         nsamples;

      bool                        init_sample;
      int                         fibst;
      bool                        usef;

      NEWIMAGE::volume<int>       locfibchoice;

    public:
      //constructors::
      Tractvolsx(const bool& usefin=false):opts(TRACT::probtrackxOptions::getInstance()),
					   logger(Utilities::LogSingleton::getInstance()),
					   init_sample(true),fibst(0),usef(usefin){}
      ~Tractvolsx(){}
      int get_nfibres()const{return nfibres;}
      int get_nsamples()const{return nsamples;}

      void reset(const int& fibst_in){
	init_sample=true;
	fibst=fibst_in;
      }

      int sample_fibre(int col,int samp,const NEWMAT::ColumnVector& dir){
	float th,ph;NEWMAT::ColumnVector x(3);
	std::vector<int> fibvec;
	for(int fib=0;fib<nfibres;fib++){
	  float ft=fsamples[fib](samp,col);
	  if(ft>opts.fibthresh.value()){
	    th=thsamples[fib](samp,col);
	    ph=phsamples[fib](samp,col);
	    x<<std::sin(th)*std::cos(ph)<<std::sin(th)*std::sin(ph)<<std::cos(th);
	    if( std::fabs( x(1)*dir(1)+x(2)*dir(2)+x(3)*dir(3) ) > 0.766 ){ //hard-coded 40 deg threshold
	      fibvec.push_back(fib);
	    }
	  }
	}
	if(fibvec.size()==0){
	  return 0;
	}
	else{
	  double rtmp=(rand()/(double(RAND_MAX)+1)) * fibvec.size();
	  return (fibvec[ (int)std::floor(rtmp) ]);
	}
      }

      int sample_fibre(int col,int samp,const int& mode=2){
	if(mode==0){
	  return 0;
	}
	if(mode==3){//sample all
	  double rtmp=(rand()/(double(RAND_MAX)+1)) * nfibres;
	  return int(std::floor(rtmp));
	}
	else{
	  if(mode==1){//sample all>thresh
	    std::vector<int> fibvec;
	    for(int fib=0;fib<nfibres;fib++){
	      float ft=fsamples[fib](samp,col);
	      if(ft>opts.fibthresh.value()){
		fibvec.push_back(fib);
	      }
	    }
	    if(fibvec.size()==0){
	      return 0;
	    }
	    else{
	      double rtmp=(rand()/(double(RAND_MAX)+1)) * fibvec.size();
	      return (fibvec[ (int)std::floor(rtmp) ]);
	    }
	  }
	  else if(mode==2){//sample all>thresh in proportion of f (default)
	    float fsumtmp=0;
	    for(int fib=0;fib<nfibres;fib++){
	      float ft=fsamples[fib](samp,col);
	      if(ft>opts.fibthresh.value()){
		fsumtmp+=ft;  //count total weight of f in this voxel.
	      }
	    }
	    if(fsumtmp==0){
	      return(0);
	    }
	    else{
	      float ft,fsumtmp2=0;
	      float rtmp=fsumtmp * (float)rand()/float(RAND_MAX);
	      for(int fib=0;fib<nfibres;fib++){
		ft=fsamples[fib](samp,col);
		if(ft>opts.fibthresh.value())
		  fsumtmp2 += ft;
		if(rtmp<=fsumtmp2){
		  return(fib);
		}
	      }
	    }
	  }
	  else{
	    std::cerr<<"TRACTVOLSX::sample_fibre:Error - unknown mode = "<<mode<<std::endl;
	    exit(1);
	  }
	}
	return 0;
      }

      int sample_ang_prob(const std::vector<float>& probs){
	float sum=0;NEWMAT::ColumnVector cumsum(probs.size());cumsum=0;
	int ind=0;
	for (unsigned int i=0;i<probs.size();i++){
	  sum += probs[i];
	  cumsum(i+1)=sum;
	}
	float U=rand()/float(RAND_MAX);
	U *= sum;
	for(unsigned int k=1;k<=probs.size();k++){
	  if(U<cumsum(k)){
	    ind=k-1;
	    break;
	  }
	}
	return ind;
      }

      //Initialise
      void initialise(const std::string& basename,const NEWIMAGE::volume<float>& mask){
	NEWIMAGE::volume4D<float> tmpvol;
	NEWMAT::Matrix            tmpmat;

	std::cout<<"Load bedpostx samples"<<std::endl;
	if(NEWIMAGE::fsl_imageexists(basename+"_thsamples")){
	  std::cout<<"1"<<std::endl;
      NEWIMAGE::read_volume4D(tmpvol,basename+"_thsamples");
	  tmpmat=tmpvol.matrix(mask);
	  thsamples.push_back(tmpmat);
	  std::cout<<"2"<<std::endl;
	  NEWIMAGE::read_volume4D(tmpvol,basename+"_phsamples");
	  tmpmat=tmpvol.matrix(mask);
	  phsamples.push_back(tmpmat);
	  std::cout<<"3"<<std::endl;
	  NEWIMAGE::read_volume4D(tmpvol,basename+"_fsamples");
	  tmpmat=tmpvol.matrix(mask);
	  fsamples.push_back(tmpmat);

	  lut_vol2mat = tmpvol.vol2matrixkey(mask);
	  nsamples    = tmpmat.Nrows();
	  nfibres     = 1;
	}
	else{
	  int fib=1;
	  bool fib_existed=true;
	  while(fib_existed){
	    if(NEWIMAGE::fsl_imageexists(basename+"_th"+MISCMATHS::num2str(fib)+"samples")){
	      std::cout<<fib<<"_1"<<std::endl;
	      NEWIMAGE::read_volume4D(tmpvol,basename+"_th"+MISCMATHS::num2str(fib)+"samples");
	      tmpmat=tmpvol.matrix(mask);
	      thsamples.push_back(tmpmat);
	      std::cout<<fib<<"_2"<<std::endl;
	      NEWIMAGE::read_volume4D(tmpvol,basename+"_ph"+MISCMATHS::num2str(fib)+"samples");
	      tmpmat=tmpvol.matrix(mask);
	      phsamples.push_back(tmpmat);
	      std::cout<<fib<<"_3"<<std::endl;
	      NEWIMAGE::read_volume4D(tmpvol,basename+"_f"+MISCMATHS::num2str(fib)+"samples");
	      tmpmat=tmpvol.matrix(mask);
	      fsamples.push_back(tmpmat);
	      fib++;
	    }
	    else{
	      fib_existed=false;
	    }
	  }
	  if(fib==1){
	      std::cerr<<"Could not find samples to load. Exit without doing anything"<<std::endl;
	      exit(1);
	  }
	  lut_vol2mat = tmpvol.vol2matrixkey(mask);
	  nsamples = thsamples[0].Nrows();
	  nfibres  = (int)thsamples.size();
	}
	NEWIMAGE::copybasicproperties(mask,lut_vol2mat);

	std::cout<<std::endl;
	std::cout<<"nfibres  : "<<nfibres<<std::endl;
	std::cout<<"nsamples : "<<nsamples<<std::endl;
	std::cout<<std::endl;
	std::cout<<"Done loading samples."<<std::endl;

	if(opts.locfibchoice.value()!=""){
	  NEWIMAGE::read_volume(locfibchoice,opts.locfibchoice.value());
	}
      }


      NEWMAT::ColumnVector sample(const float& x,const float& y,const float&z,
			  const float& r_x,const float& r_y,const float& r_z,
			  float& prefer_x,float& prefer_y,float& prefer_z,
			  const int& sample_fib,int& sampled_fib,
			  int& newx,int& newy,int& newz){

	//Tracer_Plus tr("sample");
	////////Probabilistic interpolation
	if ( !opts.noprobinterpol.value() ) {
		int cx =(int) ceil(x),fx=(int) floor(x);
		int cy =(int) ceil(y),fy=(int) floor(y);
		int cz =(int) ceil(z),fz=(int) floor(z);

		float pcx = (cx==fx)?1:(x-fx)/(cx-fx);
		float pcy = (cy==fy)?1:(y-fy)/(cy-fy);
		float pcz = (cz==fz)?1:(z-fz)/(cz-fz);

		newx = ((float)rand()/(float)RAND_MAX)>pcx?fx:cx;
		newy = ((float)rand()/(float)RAND_MAX)>pcy?fy:cy;
		newz = ((float)rand()/(float)RAND_MAX)>pcz?fz:cz;
	} else {
	////////Nearest neighbour interpolation for determinisitic tractography
		newx = (int)MISCMATHS::round((float)x);
		newy = (int)MISCMATHS::round((float)y);
		newz = (int)MISCMATHS::round((float)z);
	}
	////////////////////////////////////

	NEWMAT::ColumnVector th_ph_f(3);

	int col = lut_vol2mat(newx,newy,newz);
	if(col==0){//outside brain mask
	  th_ph_f=0;
	  return th_ph_f;
	}

	int samp=(int)MISCMATHS::round((float)rand()/float(RAND_MAX)*(float)(nsamples-1))+1;

	float theta=0,phi=0;
	float dotmax=0,dottmp=0;
	int fibind=0;
	if(nfibres>1){//more than 1 fibre
	  if(init_sample){//go for the specified fibre on the first jump or generate at random
	    if(!opts.fibst.set())
	      fibst=sample_fibre(col,samp,opts.randfib.value());

	    theta=thsamples[fibst](samp,col);
	    phi=phsamples[fibst](samp,col);
	    init_sample=false;
	  }
	  else{
	    if(sample_fib>0){ // pick specified fibre
	      fibind=sample_fibre(col,samp,sample_fib);
	      theta=thsamples[fibind](samp,col);
	      phi=phsamples[fibind](samp,col);
	    }
	    else{
	      if((std::fabs(prefer_x)+std::fabs(prefer_y)+std::fabs(prefer_z))==0){
		prefer_x=r_x;prefer_y=r_y;prefer_z=r_z;
	      }
	      int locrule=0;
	      if(opts.locfibchoice.value()!=""){locrule=locfibchoice(newx,newy,newz);}
	      if(locrule==1){
		fibind=sample_fibre(col,samp,1);
		theta=thsamples[fibind](samp,col);
		phi=phsamples[fibind](samp,col);
	      }
	      else if(locrule==2){ // like locrule=1 but with angle threshold
		NEWMAT::ColumnVector dir(3);dir<<r_x<<r_y<<r_z;
		fibind=sample_fibre(col,samp,dir);
		theta=thsamples[fibind](samp,col);
		phi=phsamples[fibind](samp,col);
	      }
	      else if (locrule==3) {
		fibind=sample_fibre(col,samp,2);
		theta=thsamples[fibind](samp,col);
		phi=phsamples[fibind](samp,col);
	      }
	      else{ // pick closest direction
		for(int fib=0;fib<nfibres;fib++){
		  if(fsamples[fib](samp,col)>opts.fibthresh.value()){
		    float phtmp=phsamples[fib](samp,col);
		    float thtmp=thsamples[fib](samp,col);
		    dottmp=std::fabs(std::sin(thtmp)*(std::cos(phtmp)*prefer_x + std::sin(phtmp)*prefer_y) + std::cos(thtmp)*prefer_z);
		    if(dottmp>dotmax){
		      dotmax=dottmp;
		      theta=thtmp;
		      phi=phtmp;
		      fibind=fib;
		    }
		  }
		}
		if(dotmax==0){
		  theta=thsamples[0](samp,col);
		  phi=phsamples[0](samp,col);
		  fibind=0;
		}
	      }
	    }
	  }
	}
	else{
	  theta=thsamples[0](samp,col);
	  phi=phsamples[0](samp,col);
	}

	float f;
	if(usef){
	  f = fsamples[fibind](samp,col);
	}
	else{
	  f=1;
	}

	sampled_fib = fibind+1;

	th_ph_f(1)=theta;
	th_ph_f(2)=phi;
	th_ph_f(3)=f;
	return th_ph_f;
      }

      NEWMAT::ColumnVector dimensions() const{
	NEWMAT::ColumnVector dims(3);
	dims << lut_vol2mat.xdim() <<lut_vol2mat.ydim() << lut_vol2mat.zdim();
	return dims;
      }
    };
}

#endif
