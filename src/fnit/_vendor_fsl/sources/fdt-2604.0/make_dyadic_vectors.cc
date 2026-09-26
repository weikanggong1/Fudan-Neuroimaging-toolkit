/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <vector>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

int main ( int argc, char **argv ){
  if(argc<4){
    cout<<"usage: make_dyadic_vectors <theta_vol> <phi_vol> [mask] <output> [perc]"<<endl;
    cout<<"[mask] is optional"<<endl;
    cout<<"[perc] is the {perc}% angle of the output cone of uncertainty (output will be in degrees)"<<endl;
    exit(1);
  }

  volume4D<float> ths,phs;
  read_volume4D(ths,argv[1]);
  read_volume4D(phs,argv[2]);
  volume<float> mask;
  string oname;
  bool ocone  = false;
  float acone = 0;
  if(argc==4){
    mask=ths[0]*0+1;
    oname=argv[3];
  }
  else if(argc>=5){
    oname=argv[4];
    read_volume(mask,argv[3]);
    if(argc==6){
      ocone = true;
      acone = atof(argv[5]);
      acone/=100.0;
      if(acone<0 || acone>1){
	cerr << "cone of uncertainty should be in percent and range between 0 and 100%" << endl;
	exit(1);
      }

    }
  }



  volume4D<float> dyadic_vecs(ths.xsize(),ths.ysize(),ths.zsize(),3);
  volume<float> disp(ths.xsize(),ths.ysize(),ths.zsize());
  dyadic_vecs=0;
  copybasicproperties(ths[0],dyadic_vecs);
  copybasicproperties(ths[0],disp);
  SymmetricMatrix dyad(3);dyad=0;
  ColumnVector dir(3);

  DiagonalMatrix dyad_D; //eigenvalues
  Matrix dyad_V; //eigenvectors

  for(int k=ths.minz();k<=ths.maxz();k++){
    for(int j=ths.miny();j<=ths.maxy();j++){
      for(int i=ths.minx();i<=ths.maxx();i++){
	if(mask(i,j,k)!=0){
	  dyad=0;
	  for(int s=ths.mint();s<=ths.maxt();s++){

	    float th=ths(i,j,k,s);
	    float ph=phs(i,j,k,s);
	    dir(1)=sin(th)*cos(ph);
	    dir(2)=sin(th)*sin(ph);
	    dir(3)=cos(th);
	    dyad << dyad+dir*dir.t();
	  }
	  dyad = dyad/float(ths.maxt()-ths.mint()+1);
	  EigenValues(dyad,dyad_D,dyad_V);
	  int maxeig;
	  if(dyad_D(1)>dyad_D(2)){
	    if(dyad_D(1)>dyad_D(3)) maxeig=1;
	    else maxeig=3;
	  }
	  else{
	    if(dyad_D(2)>dyad_D(3)) maxeig=2;
	    else maxeig=3;
	  }
	  dyadic_vecs(i,j,k,0)=dyad_V(1,maxeig);
	  dyadic_vecs(i,j,k,1)=dyad_V(2,maxeig);
	  dyadic_vecs(i,j,k,2)=dyad_V(3,maxeig);
	  disp(i,j,k)=1-dyad_D.MaximumAbsoluteValue();
	}
	else{
	  dyadic_vecs(i,j,k,0)=0;
	  dyadic_vecs(i,j,k,1)=0;
	  dyadic_vecs(i,j,k,2)=0;
	  disp(i,j,k)=0;
	}
      }
    }
  }

  dyadic_vecs.setDisplayMaximumMinimum(1,-1);
  save_volume4D(dyadic_vecs,oname);
  disp.setDisplayMaximumMinimum(1,0);
  save_volume(disp,oname+"_dispersion");

  // where we calculate the cone of uncertainty
  if(ocone){
    cout << "calculate cones" << endl;
    volume<float> cones;
    cones = ths[0]*0;
    ColumnVector meanDyad(3);
    for(int k=ths.minz();k<=ths.maxz();k++)
      for(int j=ths.miny();j<=ths.maxy();j++)
	for(int i=ths.minx();i<=ths.maxx();i++){
	  if(mask(i,j,k)==0)continue;

	  meanDyad<< dyadic_vecs(i,j,k,0) << dyadic_vecs(i,j,k,1) << dyadic_vecs(i,j,k,2);
	  ColumnVector angles(ths.tsize());
	  for(int s=0;s<ths.tsize();s++){

	    float th=ths(i,j,k,s);
	    float ph=phs(i,j,k,s);
	    dir(1)=sin(th)*cos(ph);
	    dir(2)=sin(th)*sin(ph);
	    dir(3)=cos(th);

	    angles(s+1) = acos( abs(dot(dir,meanDyad)) ) * 180.0/M_PI;
	  }
	  SortAscending(angles);
	  cones(i,j,k) = angles( MISCMATHS::round(angles.Nrows()*acone) );

	}


    cones.setDisplayMaximumMinimum(1,0);
    save_volume(cones,oname+"_cones"+num2str(acone*100));
  }

  return 0;
}
