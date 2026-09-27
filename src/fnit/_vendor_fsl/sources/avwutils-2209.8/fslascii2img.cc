//     fsl2ascii.cc - convert raw ASCII text to a NIFTI image
//     Mark Jenkinson, FMRIB Image Analysis Group
//     Copyright (C) 2009 University of Oxford
//     CCOPYRIGHT

#include <string>
#include <fstream>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;
using namespace MISCMATHS;

void print_usage(const string& progname) {
  cout << endl;
  cout << "Usage: fslascii2img <input> <xsize> <ysize> <zsize> <tsize> <xdim> <ydim> <zdim> <TR>  <output>" << endl;
  cout << "  where sizes are in voxels, dims are in mm, TR in sec " << endl;
}


int do_work(int argc, char *argv[])
{
  int xsize, ysize, zsize, tsize;
  float xdim, ydim, zdim, tr;
  xsize=atoi(argv[2]);
  ysize=atoi(argv[3]);
  zsize=atoi(argv[4]);
  tsize=atoi(argv[5]);
  xdim=atof(argv[6]);
  ydim=atof(argv[7]);
  zdim=atof(argv[8]);
  tr=atof(argv[9]);
  volume4D<float> ovol(xsize,ysize,zsize,tsize);
  ovol.setdims(xdim,ydim,zdim,tr);
  string input_name=string(argv[1]);
  string output_name=string(argv[10]);
  Matrix amat;
  amat = read_ascii_matrix(input_name);
  if (xsize*ysize*zsize*tsize != amat.Nrows() * amat.Ncols()) {
    cerr << "Sizes incompatible: " <<  xsize*ysize*zsize*tsize << " voxels vs " << amat.Nrows() * amat.Ncols() << " numbers" << endl;
    cerr << "Matrix dimensions are " << amat.Nrows() << " by " << amat.Ncols() << endl;
    exit(EXIT_FAILURE);
  }
  amat = reshape(amat.t(),tsize,xsize*ysize*zsize);
  ovol.setmatrix(amat);
  save_volume4D(ovol,output_name);
  return 0;
}


int main(int argc,char *argv[])
{

  Tracer tr("main");

  string progname=argv[0];
  if (argc != 11)
  {
    print_usage(progname);
    return 1;
  }

  return do_work(argc,argv);
}
