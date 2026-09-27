/*  fslpspec.cc

    Christian F. Beckmann, FMRIB Image Analysis Group

    Copyright (C) 2003-2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;

void print_usage(const string& progname) {
  cout << "Usage: " << progname << " <input> [options] [output]" << endl;
}

ReturnMatrix calcFFT(const Matrix& Mat)
{
  Matrix res((int)ceil(Mat.Nrows()/2.0),Mat.Ncols()), empty(1,1);
  empty=0;
  ColumnVector tmpCol;
  ColumnVector FtmpCol_real;
  ColumnVector FtmpCol_imag;
  ColumnVector tmpPow;
  for(int ctr=1; ctr <= Mat.Ncols(); ctr++)
    {
      tmpCol=Mat.Column(ctr);
      if(tmpCol.Nrows()%2 != 0){
	tmpCol &= empty;}
      RealFFT(tmpCol,FtmpCol_real,FtmpCol_imag);
      tmpPow = SP(FtmpCol_real,FtmpCol_real)+SP(FtmpCol_imag,FtmpCol_imag);
      tmpPow = tmpPow.Rows(2,tmpPow.Nrows());
      //if(opts.logPower.value()) tmpPow = log(tmpPow);
      //if(res.Storage()==0){res= tmpPow;}else{res|=tmpPow;}
      res.SubMatrix(1,res.Nrows(),ctr,ctr) = tmpPow;
    }
  res.Release();
  return res;
} //Matrix calcFFT()

int generate_masks(volume<float> &mask, const volume<float> &vin)
{

  mask = binarise(vin,vin.min(),vin.max());
  return 0;
}

int fmrib_main(int argc, char* argv[])
{
  string inname=argv[1];
  string maskname="";
  string outname="";

  if(argc>2)
    outname=argv[2];
  else
    outname=inname;

  Matrix iMat, oMat;
  volume4D<float> vin,vout;
  volume<float> mask;

  read_volume4D(vin,argv[1]);
  generate_masks(mask,stddevvol(vin));
  iMat = vin.matrix(mask);

  oMat = calcFFT(iMat);

  vout.setmatrix(oMat,mask);
  copybasicproperties(vin,vout);

  return save_volume4D(vout,outname);
}


int main(int argc,char *argv[])
{
 Tracer tr("main");

  string progname=argv[0];
  if (argc<2) {
    print_usage(progname);
    return 1;
  }

  return fmrib_main(argc,argv);

}
