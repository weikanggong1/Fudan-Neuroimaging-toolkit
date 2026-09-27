//     fslcpgeom.cc - Copy certain parts of an AVW header
//     Mark Jenkinson, Steve Smith and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 2001-2018 University of Oxford
//     CCOPYRIGHT

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
  cout << "Usage: fslcpgeom <source> <destination> [-d]" << endl;
  cout << "-d : don't copy image dimensions" << endl;
}


int avwcpgeom_main(int argc, char *argv[])
{
  NiftiHeader headerSrc,headerDest;
  string fullDestinationName(return_validimagefilename(string(argv[2])));
  try {
    headerSrc=loadHeader(return_validimagefilename(string(argv[1])));
    headerDest=loadHeader(fullDestinationName);
  } catch (...) {
    perror("Error opening files");
    return EXIT_FAILURE;
  }

  short copydim( argc > 3 ? 0 : 1);
  char *buffer = NULL;
  size_t nbytes(max(headerSrc.nElements(),headerDest.nElements())*headerDest.bitsPerVoxel/8);

  if( (buffer = (char*)calloc(nbytes,1)) == NULL ) {
    perror("Unable to allocate memory for copy");
    return EXIT_FAILURE;
  }

  vector<NiftiExtension> extensions;
  loadImage(fullDestinationName,buffer,extensions,false);

  headerDest.sformCode = headerSrc.sformCode;
  headerDest.setSForm(headerSrc.getSForm());
  headerDest.qformCode = headerSrc.qformCode;
  headerDest.setQForm(headerSrc.getQForm());
  headerDest.pixdim=headerSrc.pixdim;

  if (copydim)
    headerDest.dim=headerSrc.dim;

  saveImage(fullDestinationName,
            buffer,
            extensions,
            headerDest,
            fslZnzFormat(fslFileType(fullDestinationName)));

  return 0;
}


int main(int argc,char *argv[])
{
  if (argc < 3)
  {
    print_usage(string(argv[0]));
    return 1;
  }
  return avwcpgeom_main(argc,argv);
}
