/*  tractvols.h

    Tim Behrens, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __TRACTVOLS_H_
#define __TRACTVOLS_H_

/////////////////////////////////////////////////////////
//         Class TractVols                             //
/////////////////////////////////////////////////////////

#include "newimage/newimageall.h"
#include <iostream>
#include "stdlib.h"

using namespace std;
using namespace NEWIMAGE;

namespace TRACTVOLS{
  class TractVols
    {
    private:
      volume4D<float> thsamples;
      volume4D<float> phsamples;
      volume4D<float> fsamples;
      bool usef;
    public:
      //constructors::
      TractVols(const bool& usefin=false):usef(usefin){}
      ~TractVols(){}
      
      //Initialise
      void initialise(const string& basename){
	read_volume4D(thsamples,basename+"_thsamples");
	read_volume4D(phsamples,basename+"_phsamples");
	if(usef)
	  read_volume4D(fsamples,basename+"_fsamples");
      }
      

      ColumnVector sample(const float& x,const float& y,const float&z){
	// 	int r_x=(int) MISCMATHS::round(x);
	// 	int r_y=(int) MISCMATHS::round(y);
	// 	int r_z=(int) MISCMATHS::round(z);
	
	////////Probabilistic interpolation
	int cx =(int) ceil(x),fx=(int) floor(x);
	int cy =(int) ceil(y),fy=(int) floor(y);
	int cz =(int) ceil(z),fz=(int) floor(z);
	
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
	float tmp=rand(); tmp/=RAND_MAX;
	if(tmp>pcx)
	  newx=fx;
	else
	  newx=cx;
	
	tmp=rand(); tmp/=RAND_MAX;
	if(tmp>pcy)
	  newy=fy;
	else
	  newy=cy;
	
	tmp=rand(); tmp/=RAND_MAX;
	if(tmp>pcz)
	  newz=fz;
	else
	  newz=cz;
 
	ColumnVector th_ph_f(3);	
	
	
	float samp=rand(); samp/=RAND_MAX;
	samp=MISCMATHS::round(samp*(thsamples.tsize()-1));
	//float phi = phsamples(r_x,r_y,r_z,samp);
	//float theta = thsamples(r_x,r_y,r_z,samp);
		
	float phi = phsamples(int(newx),int(newy),int(newz),int(samp));
	float theta = thsamples(int(newx),int(newy),int(newz),int(samp));
	
	float f;
	
	if(usef){
	  f = fsamples(int(newx),int(newy),int(newz),int(samp));
	}
	else
	  f=1;
	
	th_ph_f(1)=theta;
	th_ph_f(2)=phi;
	th_ph_f(3)=f;
	return th_ph_f;
      }

      ColumnVector dimensions(){
	ColumnVector dims(3);
	dims << thsamples.xdim() <<thsamples.ydim() << thsamples.zdim();
	return dims;
      }
    };
}

#endif



