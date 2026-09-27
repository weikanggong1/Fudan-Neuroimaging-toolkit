//     fslhd.cc - show image header
//     Steve Smith, Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 2000-2019 University of Oxford
//     CCOPYRIGHT

#include <iomanip>
#include <iostream>

#include "NewNifti/NewNifti.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NiftiIO;
using namespace NEWIMAGE;

void reportXMLHeader(const NiftiHeader& header)
{
  cout << "<nifti_image" << endl;
  cout << "  image_offset = '" << header.vox_offset  << "'" << endl;
  cout << "  ndim = '" << header.dim[0] << "'" << endl;
  cout << "  nx = '" << header.dim[1] << "'" << endl;
  cout << "  ny = '" << header.dim[2] << "'" << endl;
  cout << "  nz = '" << header.dim[3] << "'" << endl;
  cout << "  nt = '" << header.dim[4] << "'" << endl;
  if ( header.dim[0] > 4 ) {
    cout << "  nu = '" << header.dim[5] << "'" << endl;
    cout << "  nv = '" << header.dim[6] << "'" << endl;
    cout << "  nw = '" << header.dim[7] << "'" << endl;
  }
  cout << "  dx = '" << header.pixdim[1] << "'" << endl;
  cout << "  dy = '" << header.pixdim[2] << "'" << endl;
  cout << "  dz = '" << header.pixdim[3] << "'" << endl;
  cout << "  dt = '" << header.pixdim[4] << "'" << endl;
  if ( header.dim[0] > 4 ) {
    cout << "  du = '" << header.pixdim[5] << "'" << endl;
    cout << "  dv = '" << header.pixdim[6] << "'" << endl;
    cout << "  dw = '" << header.pixdim[7] << "'" << endl;
  }
  cout << "  datatype = '" << header.datatype << "'" << endl;
  cout << "  nvox = '" << header.nElements() << "'" << endl;
  cout << "  nbyper = '" << header.bitsPerVoxel/8 << "'" << endl;
  cout << "  scl_slope = '" << header.sclSlope << "'" << endl;
  cout << "  scl_inter = '" << header.sclInter << "'" << endl;
  cout << "  intent_code = '" << header.intentCode << "'" << endl;
  cout << "  intent_p1 = '" << header.intent_p1 << "'" << endl;
  cout << "  intent_p2 = '" << header.intent_p2 << "'" << endl;
  cout << "  intent_p3 = '" << header.intent_p3 << "'" << endl;
  cout << "  intent_name = '" << header.intentName << "'" << endl;
  cout << "  toffset = '" << header.toffset << "'" << endl;
  cout << "  xyz_units = '" << XYZT_TO_SPACE(header.units) << "'" << endl;
  cout << "  time_units = '" << XYZT_TO_TIME(header.units) << "'" << endl;
  cout << "  freq_dim = '" << (int)header.freqDim() << "'" << endl;
  cout << "  phase_dim = '" << (int)header.phaseDim() << "'" << endl;
  cout << "  slice_dim = '" << (int)header.sliceDim() << "'" << endl;
  cout << "  descrip = '" << header.description << "'" << endl;
  cout << "  aux_file = '" << header.auxillaryFile << "'" << endl;
  if ( !header.isAnalyze() ) {
    cout << "  qform_code = '" << header.qformCode << "'" << endl;
    cout << "  qfac = '" << header.leftHanded() << "'" << endl;
    cout << "  quatern_b = '" << header.qB << "'" << endl;
    cout << "  quatern_c = '" << header.qC << "'" << endl;
    cout << "  quatern_d = '" << header.qD << "'" << endl;
    cout << "  qoffset_x = '" << header.qX << "'" << endl;
    cout << "  qoffset_y = '" << header.qY << "'" << endl;
    cout << "  qoffset_z = '" << header.qZ << "'" << endl;
  }
  cout << "  sform_code = '" << header.sformCode << "'" << endl;
  cout << "  sto_xyz_matrix = '";
  mat44 output(header.getSForm());
  for (int i=0;i<4;i++)
    for (int j=0;j<4;j++) {
     cout << output.m[i][j];
     if ( j==3 && i==3 )
       cout << "'" << endl;
     else
       cout << " ";
    }
  cout << "  slice_code = '" << header.sliceCode << "'" << endl;
  cout << "  slice_start = '" << header.sliceStart << "'" << endl;
  cout << "  scl_end = '" << header.sliceEnd << "'" << endl;
  cout << "  scl_duration = '" << header.sliceDuration << "'" << endl;
  cout << "/>" << endl;
}

int print_usage(const string& progname)
{
  cout << endl;
  cout << "Usage: fslhd [-x] <input>" << endl;
  cout << "       -x : instead print an XML-style NIFTI header" << endl;
  return 1;
}

int main(int argc,char *argv[])
{
  if (argc < 2)
    return print_usage(string(argv[0]));
  bool reportXML(argc==3 && strcmp(argv[1],"-x")==0 );
  try {
    string filename(return_validimagefilename(argv[argc-1]));
    NiftiHeader header(loadHeader(filename));
    if (reportXML)
      reportXMLHeader(header);
    else {
      cout << "filename\t" << filename << endl << endl;
      header.report();
    }
  }  catch ( exception& e ) { cerr << e.what() << endl; return 1;}
  return 0;
}
