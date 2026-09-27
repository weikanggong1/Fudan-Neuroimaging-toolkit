//     fslsplit.cc - split 4D files into 3D files for SPM
//     David Flitney and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 2000-2006 University of Oxford
//     CCOPYRIGHT

#include <string>
#include <iostream>

#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"
#include "miscmaths/miscmaths.h"

using namespace std;
using namespace NEWIMAGE;
using namespace MISCMATHS;

void print_usage(const string& progname)
{
  cout << endl;
  cout << "Usage: fslsplit <input>" << endl;
  cout << "       fslsplit <input> [output_basename] [-t/x/y/z]" << endl;
  cout << "       -t : separate images in time (default behaviour)" << endl;
  cout << "       -x : separate images in the x direction"  << endl;
  cout << "       -y : separate images in the y direction"  << endl;
  cout << "       -z : separate images in the z direction" << endl;
}

template <class T>
int fmrib_main(int argc, char *argv[])
{
  volume4D<T> input_vol,output_vol;
  string input_name=string(argv[1]);
  string output_name("vol");
  int xoff(1),yoff(1),zoff(1),toff(1),nsize;
  int digits;
  read_volume4D(input_vol,input_name);
  if ((argc>2) && (argv[2][0]!='-')) output_name=string(argv[2]);
  if (argv[argc-1][0] == '-')
  {
      if (argv[argc-1][1] == 't') toff=input_vol.tsize();
      if (argv[argc-1][1] == 'z') zoff=input_vol.zsize();
      if (argv[argc-1][1] == 'y') yoff=input_vol.ysize();
      if (argv[argc-1][1] == 'x') xoff=input_vol.xsize();
  }
  else toff=input_vol.tsize();
  nsize=xoff*yoff*zoff*toff;

  // use enough digits in the file  names for
  // the number of files being  generated (but
  // at least 4), so the file names can be
  // easily alphabetically sorted
  digits = fmax(4, int(log10(nsize) + 1));

  for(int j=0;j<nsize;j++)
  {
    output_vol=input_vol.ROI(0+j*(xoff!=1),0+j*(yoff!=1),0+j*(zoff!=1),0+j*(toff!=1),input_vol.xsize()-1-(nsize-j-1)*(xoff!=1),input_vol.ysize()-1-(nsize-j-1)*(yoff!=1),input_vol.zsize()-1-(nsize-j-1)*(zoff!=1),input_vol.tsize()-1-(nsize-j-1)*(toff!=1));
    save_volume4D(output_vol,(output_name+num2str(j,digits)));
  }
  return 0;
}


int main(int argc,char *argv[])
{
  string progname=argv[0];
  if (argc <= 1 || argc >= 5)
  {
    print_usage(progname);
    return 1;
  }

  string iname=string(argv[1]);
  return call_fmrib_main(dtype(iname),argc,argv);
}
