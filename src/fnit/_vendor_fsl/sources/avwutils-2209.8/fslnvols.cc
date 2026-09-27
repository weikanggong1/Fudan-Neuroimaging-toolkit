//     fslnvols.cc   REALLY advanced program for counting the volumes in a 4D file
//     Stephen Smith and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 1999-2019 University of Oxford
//     CCOPYRIGHT

#include <string>

#include "NewNifti/NewNifti.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NiftiIO;
using namespace NEWIMAGE;

void print_usage(const string& progname)
{
  cout << endl;
  cout << "Usage: fslnvols <input>" << endl;
}

int main(int argc,char *argv[])
{
  string progname=argv[0];
  if (argc != 2)
  {
    print_usage(progname);
    return 1;
  }
  if (!FslFileExists(string(argv[1])))
  {
    cout << "0" << endl;
    return 0; // the intended usage is to return "0" and not show an error
  }
  NiftiHeader header(loadHeader(return_validimagefilename(string(argv[1]))));
  cout << header.dim[4] << endl;
  return 0;
}
