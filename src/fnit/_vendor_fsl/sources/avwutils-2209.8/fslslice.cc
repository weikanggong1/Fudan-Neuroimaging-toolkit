/*  avwslice.cc

    Tim Behrens and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2000-2007 University of Oxford  */

/*  CCOPYRIGHT  */


#include <iostream>
#include <string>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;
using namespace MISCMATHS;


string zeropad_numeric_string(string in,unsigned int len){
  string out=in;
  while(out.length() < len){
    out="0"+out;
  }
  return out;
}


template <class T>
int fmrib_main(int argc, char *argv[])
{
  volume4D<T> grot;
  volume4D<T> tmp;
  string iname=string(argv[1]);
  read_volume4D(grot,iname);
  string base;

  if(argc==2)
    base=iname;
  else
    base=string(argv[2]);

  make_basename(base);

  for(int z=0;z<grot.zsize();z++){
    tmp=grot.ROI(0,0,z,0,grot.maxx(),grot.maxy(),z,grot.maxt());
    string oname=base+"_slice_"+zeropad_numeric_string(num2str(z),4);
    save_volume4D(tmp,oname);
  }
  return 0;
}

int main(int argc,char *argv[])
{

  Tracer tr("main");

  if(argc<2){
    cout <<"Usage: fslslice <volume>"<<endl;
    cout<<endl;
    cout <<"Usage: fslslice <volume> <output basename>"<<endl;
    cout<<endl;
    return -1;
  }
  return call_fmrib_main(dtype(string(argv[1])),argc,argv);
}
