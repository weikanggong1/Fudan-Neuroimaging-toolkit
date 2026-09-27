//     fslroi.cc  extract cuboid ROI and/or timeseries from image
//     Stephen Smith, Matthew Webster and Mark Jenkinson, FMRIB Image Analysis Group
//     Copyright (C) 1999-2008 University of Oxford
//     CCOPYRIGHT

#include <string>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;
using namespace MISCMATHS;


void print_usage(const string& progname) {
  cout << endl;
  cout << "Usage: fslroi <input> <output> <xmin> <xsize> <ymin> <ysize> <zmin> <zsize>" << endl;
  cout << "       fslroi <input> <output> <tmin> <tsize>\n" << endl;
  cout << "       fslroi <input> <output> <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize>" << endl;
  cout << "Note: indexing (in both time and space) starts with 0 not 1! Inputting -1 for a size will set it to the full image extent for that dimension." << endl;
}


template <class T>
int fmrib_main(int argc, char *argv[])
{
  volume4D<T> input_vol,output_vol;
  read_volume4D(input_vol,string(argv[1]));

  int minx(-1),miny(-1),minz(-1),mint(-1),maxx(-1),maxy(-1),maxz(-1),maxt(-1),argindex(3);
  //Note there is currently no explicit bounds testing e.g. if maxx>inputvol.maxx() etc
  if (argc==5) //4D Timeseries only
  {
    minx=0;
    maxx=input_vol.maxx();
    miny=0;
    maxy=input_vol.maxy();
    minz=0;
    maxz=input_vol.maxz();
    mint=atoi(argv[argindex++]);
    maxt=atoi(argv[argindex++]);
    maxt+=mint-1;
  }
  else if (argc==9)  //3D Region of interest
  {
    minx=atoi(argv[argindex++]);   //N.B. could compact some of these lines with the ones below...
    maxx=atoi(argv[argindex++]);
    miny=atoi(argv[argindex++]);
    maxy=atoi(argv[argindex++]);
    minz=atoi(argv[argindex++]);
    maxz=atoi(argv[argindex++]);
    mint=0;
    maxt=input_vol.maxt();
    maxx+=minx-1;
    maxy+=miny-1;
    maxz+=minz-1;
  }
  else if (argc==11)   //4D Timeseries and Region of interest
  {
    minx=atoi(argv[argindex++]);
    maxx=atoi(argv[argindex++]);
    miny=atoi(argv[argindex++]);
    maxy=atoi(argv[argindex++]);
    minz=atoi(argv[argindex++]);
    maxz=atoi(argv[argindex++]);
    mint=atoi(argv[argindex++]);
    maxt=atoi(argv[argindex++]);
    maxx+=minx-1;
    maxy+=miny-1;
    maxz+=minz-1;
    maxt+=mint-1;
  }

  if ( maxx == minx-2 ) //Then the user input -1 for size
    maxx=input_vol.maxx();
  if ( maxy == miny-2 ) //Then the user input -1 for size
    maxy=input_vol.maxy();
  if ( maxz == minz-2 ) //Then the user input -1 for size
    maxz=input_vol.maxz();
  if ( maxt == mint-2 ) //Then the user input -1 for size
    maxt=input_vol.maxt();

  // sanity check
  if ((maxx<minx) || (maxy<miny) || (maxz<minz) || (maxt<mint)) {
    imthrow("Invalid ROI dimensions",21);
  }

  // now transform these coordinates to newimage conventions
  ColumnVector v(4);
  v << minx << miny << minz << 1.0;
  v = input_vol.niftivox2newimagevox_mat() * v;
  minx = MISCMATHS::round(v(1));
  miny = MISCMATHS::round(v(2));
  minz = MISCMATHS::round(v(3));
  v << maxx << maxy << maxz << 1.0;
  v = input_vol.niftivox2newimagevox_mat() * v;
  maxx = MISCMATHS::round(v(1));
  maxy = MISCMATHS::round(v(2));
  maxz = MISCMATHS::round(v(3));

  if (minx>maxx) { int tmpx=maxx; maxx=minx; minx=tmpx; }
  if (miny>maxy) { int tmpy=maxy; maxy=miny; miny=tmpy; }
  if (minz>maxz) { int tmpz=maxz; maxz=minz; minz=tmpz; }
  save_volume4D(input_vol.ROI(minx,miny,minz,mint,maxx,maxy,maxz,maxt),string(argv[2]));
  return 0;
}


int main(int argc,char *argv[])
{
  try {
    string progname=argv[0];
    if (argc !=5 && argc !=9 && argc !=11)
      {
	print_usage(progname);
	return 1;
      }

    string iname=string(argv[1]);
    return call_fmrib_main(dtype(iname),argc,argv);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }  catch(Exception &e) {
    exit(EXIT_FAILURE);
  }
}
