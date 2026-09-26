/*  tractvolsx.h

    Tim Behrens, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __TRACTVOLSX_H_
#define __TRACTVOLSX_H_

/////////////////////////////////////////////////////////
//         Class TractVolsx                             //
/////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>
#include "stdlib.h"

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "probtrackxOptions.h"


namespace TRACTVOLSX{
  class Tractvolsx
    {
    private:
      TRACT::probtrackxOptions& opts;
      std::vector<NEWIMAGE::volume4D<float>* > thsamples;
      std::vector<NEWIMAGE::volume4D<float>* > phsamples;
      std::vector<NEWIMAGE::volume4D<float>* > fsamples;
      bool init_sample;
      int fibst;
      bool usef;

    public:
      //constructors::
      Tractvolsx(const bool& usefin=false):opts(TRACT::probtrackxOptions::getInstance()),init_sample(true),fibst(0),usef(usefin){}
      Tractvolsx():opts(TRACT::probtrackxOptions::getInstance()){}
      ~Tractvolsx(){
	for(unsigned int m=0;m<thsamples.size();m++)
	  delete thsamples[m]; //ask flitney, do you just delete the ptr??
	for(unsigned int m=0;m<phsamples.size();m++)
	  delete phsamples[m];
	for(unsigned int m=0;m<fsamples.size();m++)
	    delete fsamples[m];
      }
      inline int nfibres()const{return (int)thsamples.size();}

      void reset(const int& fibst_in){
	init_sample=true;
	fibst=fibst_in;
      }
      //Initialise
      void initialise(const std::string& basename){


	if(NEWIMAGE::fsl_imageexists(basename+"_thsamples")){
	  NEWIMAGE::volume4D<float> *tmpthptr= new NEWIMAGE::volume4D<float>;
	  NEWIMAGE::volume4D<float> *tmpphptr= new NEWIMAGE::volume4D<float>;
	  NEWIMAGE::volume4D<float> *tmpfptr= new NEWIMAGE::volume4D<float>;
	  std::cout<<"1"<<std::endl;
	  NEWIMAGE::read_volume4D(*tmpthptr,basename+"_thsamples");
	  std::cout<<"2"<<std::endl;
	  thsamples.push_back(tmpthptr);
	  std::cout<<"3"<<std::endl;
	  NEWIMAGE::read_volume4D(*tmpphptr,basename+"_phsamples");
	  std::cout<<"4"<<std::endl;
	  phsamples.push_back(tmpphptr);
	  std::cout<<"5"<<std::endl;
	  NEWIMAGE::read_volume4D(*tmpfptr,basename+"_fsamples");
	  fsamples.push_back(tmpfptr);
	  std::cout<<"6"<<std::endl;
	}
	else{
	  int fib=1;
	  bool fib_existed=true;
	  while(fib_existed){
	    if(NEWIMAGE::fsl_imageexists(basename+"_th"+MISCMATHS::num2str(fib)+"samples")){
	      NEWIMAGE::volume4D<float> *tmpthptr= new NEWIMAGE::volume4D<float>;
	      NEWIMAGE::volume4D<float> *tmpphptr= new NEWIMAGE::volume4D<float>;
	      NEWIMAGE::volume4D<float> *tmpfptr= new NEWIMAGE::volume4D<float>;
	      std::cout<<fib<<"_1"<<std::endl;
	      NEWIMAGE::read_volume4D(*tmpthptr,basename+"_th"+MISCMATHS::num2str(fib)+"samples");
	      thsamples.push_back(tmpthptr);
	      std::cout<<fib<<"_2"<<std::endl;
	      NEWIMAGE::read_volume4D(*tmpphptr,basename+"_ph"+MISCMATHS::num2str(fib)+"samples");
	      phsamples.push_back(tmpphptr);
	      std::cout<<fib<<"_3"<<std::endl;
	      NEWIMAGE::read_volume4D(*tmpfptr,basename+"_f"+MISCMATHS::num2str(fib)+"samples");
	      fsamples.push_back(tmpfptr);
	      fib++;
	    }
	    else{
	      fib_existed=false;
	    }
	  }

	}
	std::cout<<"7"<<std::endl;
      }


      NEWMAT::ColumnVector sample(const float& x,const float& y,const float&z,const float& r_x,const float& r_y,const float& r_z,
			  float& prefer_x,float& prefer_y,float& prefer_z){

	////////Probabilistic interpolation
	int cx =(int) std::ceil(x),fx=(int) std::floor(x);
	int cy =(int) std::ceil(y),fy=(int) std::floor(y);
	int cz =(int) std::ceil(z),fz=(int) std::floor(z);

	//cerr<<x<<" "<<y<<" "<<z<<" "<<cx<<" "<<cy<<" "<<cz<<" "<<fx<<" "<<fy<<" "<<fz<<endl;
	float pcx,pcy,pcz;
	if(cx==fx)
	  pcx=1;
	else
	  pcx=(x-fx)/(cx-fx);

	if(cy==fy)
	  pcy=1;
	else
	  pcy=(y-fy)/(cy-fy);

	if(cz==fz)
	  pcz=1;
	else
	  pcz=(z-fz)/(cz-fz);

	///////new xyz values from probabilistic interpolation
	int newx,newy,newz;
	float tmp=(float)rand()/float(RAND_MAX);
	if(tmp>pcx)
	  newx=fx;
	else
	  newx=cx;

	tmp=(float)rand()/float(RAND_MAX);
	if(tmp>pcy)
	  newy=fy;
	else
	  newy=cy;

	tmp=(float)rand()/float(RAND_MAX);
	if(tmp>pcz)
	  newz=fz;
	else
	  newz=cz;

	NEWMAT::ColumnVector th_ph_f(3);
	float samp=(float)rand()/float(RAND_MAX);
	samp=MISCMATHS::round(samp*((*thsamples[0]).tsize()-1));
	float theta=0,phi=0;
	float dotmax=0,dottmp=0;
	int fibind=0;
	if(thsamples.size()>1){//more than 1 fibre
	  if(init_sample){//go for the specified fibre on the first jump or generate at random
	    if(opts.randfib.value()==1){//this generates startfib at random (except for fibres where f<fibthresh)
	      std::vector<int> fibvec;
	      for(unsigned int fib=0;fib<thsamples.size();fib++){
		float ft=(*fsamples[fib])(int(newx),int(newy),int(newz),int(samp));
		if(ft>opts.fibthresh.value()){
		  fibvec.push_back(fib);
		}
	      }

	      if(fibvec.size()==0){
		fibst=0;
	      }
	      else{
		float rtmp=(float)rand()/float(RAND_MAX) * float(fibvec.size()-1);
		fibst = fibvec[ (int)MISCMATHS::round(rtmp) ];
	      }

	    }
	    else if(opts.randfib.value()==2){ //this generates startfib with probability proportional to f (except for fibres where f<fibthresh).
	      //this chooses at random but in proportion to fsamples.
	      float fsumtmp=0;
	      for(unsigned int fib=0;fib<thsamples.size();fib++){
		float ft=(*fsamples[fib])(int(newx),int(newy),int(newz),int(samp));
		if(ft>opts.fibthresh.value()){
		  fsumtmp+=ft;  //count total weight of f in this voxel.
		}
	      }

	      if(fsumtmp==0){
		fibst=0;
	      }
	      else{
		float ft,fsumtmp2=0;
		float rtmp=fsumtmp * (float)rand()/float(RAND_MAX);

		for(unsigned int fib=0;fib<thsamples.size();fib++){
		  ft=(*fsamples[fib])(int(newx),int(newy),int(newz),int(samp));
		  if(ft>opts.fibthresh.value())
		    fsumtmp2 += ft;
		  if(rtmp<=fsumtmp2){
		    fibst=(int)fib;
		    break;
		  }
		}
	      }
	    }

	    theta=(*thsamples[fibst])(int(newx),int(newy),int(newz),int(samp));
	    phi=(*phsamples[fibst])(int(newx),int(newy),int(newz),int(samp));
	    init_sample=false;
	  }
	  else{
	    if((std::fabs(prefer_x)+std::fabs(prefer_y)+std::fabs(prefer_z))==0){
	      prefer_x=r_x;prefer_y=r_y;prefer_z=r_z;
	    }
	    for(unsigned int fib=0;fib<thsamples.size();fib++){
	      if((*fsamples[fib])(int(newx),int(newy),int(newz),int(samp))>opts.fibthresh.value()){
		float phtmp=(*phsamples[fib])(int(newx),int(newy),int(newz),int(samp));
		float thtmp=(*thsamples[fib])(int(newx),int(newy),int(newz),int(samp));
		dottmp=std::fabs(std::sin(thtmp)*std::cos(phtmp)*prefer_x + std::sin(thtmp)*std::sin(phtmp)*prefer_y + std::cos(thtmp)*prefer_z);
		if(dottmp>dotmax){
		  dotmax=dottmp;
		  theta=thtmp;
		  phi=phtmp;
		  fibind=fib;
		}

	      }

	    }
	    if(dotmax==0){
	      theta=(*thsamples[0])(int(newx),int(newy),int(newz),int(samp));
	      phi=(*phsamples[0])(int(newx),int(newy),int(newz),int(samp));
	    }
	  }
	}
	else{
	  theta=(*thsamples[0])(int(newx),int(newy),int(newz),int(samp));
	  phi=(*phsamples[0])(int(newx),int(newy),int(newz),int(samp));
	}


	float f;

	if(usef){
	  f = (*fsamples[fibind])(int(newx),int(newy),int(newz),int(samp));
	}
	else
	  f=1;

	th_ph_f(1)=theta;
	th_ph_f(2)=phi;
	th_ph_f(3)=f;
	return th_ph_f;
      }

      NEWMAT::ColumnVector dimensions() const{
	NEWMAT::ColumnVector dims(3);
	dims << (*thsamples[0]).xdim() <<(*thsamples[0]).ydim() << (*thsamples[0]).zdim();
	return dims;
      }
    };
}

#endif
