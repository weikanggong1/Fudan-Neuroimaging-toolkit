//     fslcreatehd.cc - Copy certain parts of an AVW header
//     Mark Jenkinson, Steve Smith and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 2001-2018 University of Oxford
//     CCOPYRIGHT

#include <fstream>
#include <iostream>
#include <string>

#include "NewNifti/NewNifti.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NiftiIO;
using namespace NEWIMAGE;

void print_usage(const string& progname)
{
  cout << endl;
  cout << "Usage: fslcreatehd <xsize> <ysize> <zsize> <tsize> <xvoxsize> <yvoxsize> <zvoxsize> <tr> <xorigin> <yorigin> <zorigin> <datatype> <headername>" << endl;
  cout << "       fslcreatehd <nifti_xml_file> <headername>" << endl;
  cout << "  Datatype values: " << DT_UNSIGNED_CHAR << "=char, " << DT_SIGNED_SHORT  << "=short, " <<  DT_SIGNED_INT <<"=int, " << DT_FLOAT << "=float, " <<  DT_DOUBLE<< "=double" << endl;
  cout << "  In the first form, a radiological image will be created, with the origin input as a voxel co-ordinate." << endl;
  cout << "  If the output file already exists, its data ( but not geometric information ) will be copied if it has" << endl;
  cout << "  a matching number of elements." << endl;
  cout << "  In the second form, an XML-ish form of nifti header is read (as output by fslhd -x)" << endl;
  cout << "  Note that stdin is used if '-' is used in place of a filename" << endl;

}

NiftiHeader headerFromXML(vector<string> XMLreport)
{
  NiftiHeader header;
  int xyzUnits(0);
  int tUnits(0);
  int freqDim(0);
  int phaseDim(0);
  int sliceDim(0);
  for(unsigned int setting=1;setting<XMLreport.size()-1;setting++) {
    //Tokenise
    size_t pos1=XMLreport[setting].find_first_of("  ")+2;
    size_t pos2=XMLreport[setting].find_first_of(" ",pos1);
    size_t pos3=XMLreport[setting].find_first_of('\'')+1;
    size_t pos4=XMLreport[setting].find_last_of('\'');
    string field(XMLreport[setting].substr(pos1,pos2-pos1));
    string value(XMLreport[setting].substr(pos3,pos4-pos3));
    istringstream values(value);
    if ( field == "datatype" ) {
      values >> header.datatype;
    } else if ( field == "image_offset" ) {
      values >> header.vox_offset;
    } else if ( field == "sto_xyz_matrix" ) {
      mat44 sForm;
      for ( int i=0;i<4;i++ )
        for ( int j=0;j<4;j++ )
          values >> sForm.m[i][j];
      header.setSForm(sForm);
    } else if ( field == "ndim" ) {
      values >> header.dim[0];
    } else if ( field == "nx" ) {
      values >> header.dim[1];
    } else if ( field == "ny" ) {
      values >> header.dim[2];
    } else if ( field == "nz" ) {
      values >> header.dim[3];
    } else if ( field == "nt" ) {
      values >> header.dim[4];
    } else if ( field == "nu" ) {
      values >> header.dim[5];
    } else if ( field == "nv" ) {
      values >> header.dim[6];
    } else if ( field == "nw" ) {
      values >> header.dim[7];
    }else if ( field == "qfac" ) {
      values >> header.pixdim[0];
    } else if ( field == "dx" ) {
      values >> header.pixdim[1];
    } else if ( field == "dy" ) {
      values >> header.pixdim[2];
    } else if ( field == "dz" ) {
      values >> header.pixdim[3];
    } else if ( field == "dt" ) {
      values >> header.pixdim[4];
    } else if ( field == "du" ) {
      values >> header.pixdim[5];
    } else if ( field == "dv" ) {
      values >> header.pixdim[6];
    } else if ( field == "dw" ) {
      values >> header.pixdim[7];
    } else if ( field == "cal_min" ) {
      values >> header.cal_min;
    } else if ( field == "cal_max" ) {
      values >> header.cal_max;
    } else if ( field == "scl_slope" ) {
      values >> header.sclSlope;
    } else if ( field == "scl_inter" ) {
      values >> header.sclInter;
    } else if ( field == "intent_code" ) {
      values >> header.intentCode;
    } else if ( field == "intent_p1" ) {
      values >> header.intent_p1;
    } else if ( field == "intent_p2" ) {
      values >> header.intent_p2;
    } else if ( field == "intent_p3" ) {
      values >> header.intent_p3;
    } else if ( field == "intent_name" ) {
      values >> header.intentName;
    } else if ( field == "toffset" ) {
      values >> header.toffset;
    } else if ( field == "xyz_units" ) {
      values >> xyzUnits;
    } else if ( field == "time_units" ) {
      values >> tUnits;
    } else if ( field == "descrip" ) {
      header.description=value;
    } else if ( field == "aux_file" ) {
      values >> header.auxillaryFile;
    } else if ( field == "qform_code" ) {
      values >> header.qformCode;
    } else if ( field == "sform_code" ) {
      values >> header.sformCode;
    } else if ( field == "quatern_b" ) {
      values >> header.qB;
    } else if ( field == "quatern_c" ) {
      values >> header.qC;
    } else if ( field == "quatern_d" ) {
      values >> header.qD;
    } else if ( field == "qoffset_x" ) {
      values >> header.qX;
    } else if ( field == "qoffset_y" ) {
      values >> header.qY;
    } else if ( field == "qoffset_z" ) {
      values >> header.qZ;
    } else if ( field == "freq_dim" ) {
      values >> freqDim;
    } else if ( field == "phase_dim" ) {
      values >> phaseDim;
    } else if ( field == "slice_dim" ) {
      values >> sliceDim;
    } else if ( field == "slice_code" ) {
      values >> header.sliceCode;
    } else if ( field == "slice_start" ) {
      values >> header.sliceStart;
    } else if ( field == "slice_end" ) {
      values >> header.sliceEnd;
    } else if ( field == "slice_duration" ) {
      values >> header.sliceDuration;
    } else {
      //cerr << XMLreport[setting] << endl;
      //cerr << "Unknown" << endl;
    }
  }
  header.units=xyzUnits+tUnits;
  header.sliceOrdering=freqDim | ( phaseDim << 2 ) | ( sliceDim << 4 );
  header.bitsPerVoxel=header.bpvOfDatatype();
  return header;
}

int fslcreatehd_main(int argc, char *argv[])
{
  vector <NiftiExtension> extensions;
  NiftiHeader header,originalHeader;
  char *buffer(NULL);
  string filename;
  int fileread(1);
  bool existingImage(false);

  if (argc==3) /* use the XML form of header specification */
    filename = string(argv[2]);
  else
    filename = string(argv[13]);

  /* check if file already exists and if so, read the image contents */
  if (FslFileExists(filename)) {
    existingImage = true;
    filename=return_validimagefilename(filename);
    header=loadImage(filename,buffer,extensions);
    originalHeader=header;
  }

  if (argc>3) {

    /*
     * if creating a new file, we honour
     * the datatype that the user specified.
     * For existing images, the data type
     * of the image is preserved
     */
    if (!existingImage) {
      header.datatype=atoi(argv[12]);
    }

    /*
     * set number of dimensions sensibly -
     * if user specified 0 or 1 for the
     * fourth dim, we set ndim to 3
     */
    if (atoi(argv[4]) <= 1) {
      header.dim[0] = 3;
    }
    else {
      header.dim[0] = 4;
    }

    /*
     * we ignore dim/pixdim args beyond the
     * specified number of dimensions - note
     * that if 3 dims are specified, the
     * tsize/tr arguments will be ignored.
     */
    for (int i = 1; i <= header.dim[0]; i++) {
      header.dim[   i] = atoi(argv[i]);
      header.pixdim[i] = atof(argv[i+4]);
    }

    header.sformCode=2;
    header.sX[0]=-header.pixdim[1];
    header.sY[1]=header.pixdim[2];
    header.sZ[2]=header.pixdim[3];
    header.sX[3]=header.pixdim[1]*atoi(argv[9]);
    header.sY[3]=-header.pixdim[2]*atoi(argv[10]);
    header.sZ[3]=-header.pixdim[3]*atoi(argv[11]);
    header.qformCode=2;
    header.setQForm(header.getSForm());

  } else {
    /* read XML form */
    char *newstr;
    ifstream inputfile;
    vector<string> settings;

    if (strcmp(argv[1],"-")==0) {fileread=0;}
    newstr = (char *)calloc(10000,1);
    if (fileread) {
      inputfile.open (argv[1], ifstream::in | ifstream::binary);
      if (!inputfile.is_open()) {
        cerr << "Cannot open file " << argv[1] << endl;
        return EXIT_FAILURE;
      }
    }

    do {
      if (fileread) {
        inputfile.getline(newstr,9999);  // maybe use > for delimiting character remove while increase size
      } else {
        if (fgets(newstr,9999,stdin)==NULL) break;
      }
      settings.push_back(string(newstr));
    }  while (settings[settings.size()-1]!="/>");

    header=headerFromXML(settings);

    if (fileread)
      inputfile.close();
  }

  /* reset datatype in case it has been overwritten */
  if (existingImage) {
    header.datatype=originalHeader.datatype;
    header.bitsPerVoxel=header.bpvOfDatatype();
  }

  /* if previously read buffer is wrong size then make a zero image here */
  if ( !existingImage || header.nElements() != originalHeader.nElements() ) {
    if(buffer!=NULL)
      delete buffer;
    buffer = new char[header.nElements()*(header.bpvOfDatatype()/8)];
    fill(buffer,buffer+header.nElements(),0);
  }

  int filetype=FslGetEnvOutputType();
  filename=make_basename(filename)+outputExtension(filetype);

  header.bitsPerVoxel=header.bpvOfDatatype();
  header.setNiftiVersion(FslNiftiVersionFileType(filetype),FslIsSingleFileType(filetype));
  saveImage(filename,buffer,extensions,header, fslZnzFormat(filetype));

  return 0;
}


int main(int argc,char *argv[])
{
  if (argc != 14 && argc != 3)
    {
      print_usage(string(argv[0]));
      return 1;
    }
  return fslcreatehd_main(argc,argv);
}
