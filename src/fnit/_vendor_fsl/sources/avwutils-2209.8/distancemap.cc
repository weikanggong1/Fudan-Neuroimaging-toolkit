/*  distancemap.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2003-2022 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <string>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "utils/options.h"


#define _GNU_SOURCE 1
#define POSIX_SOURCE 1

using namespace std;

using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace Utilities;


string title="distancemap (Version 3.0)\nCopyright(c) 2003-2022, University of Oxford";
string examples="distancemap [options] -i <inputimage> -o <outputimage>";


Option<bool> verbose(string("-v,--verbose"), false,
		     string("switch on diagnostic messages"),
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<bool> invert(string("--invert"), false,
		  string("invert input image"),
		  false, no_argument);
Option<bool> nn(string("--nn"), false,
		  string("nearest sparse interpolation"),
		  false, no_argument);
Option<string> inname(string("-i,--in"), string(""),
		  string("primary image filename (calc distance to positive voxels)"),
		  true, requires_argument);
Option<string> outname(string("-o,--out"), string(""),
		  string("output image filename"),
		  true, requires_argument);
Option<string> secondname(string("--secondim"), string(""),
		  string("second image filename (calc closest distance of this and primary input image, using positive voxels, negative distances mean this secondary image is the closer one)"),
		  false, requires_argument);
Option<string> sparseFile(string("--interp"), string(""),
		  string("filename for values to interpolate, valid samples given by primary image"),
		  false, requires_argument);
Option<string> labelname(string("-l,--localmax"), string(""),
		  string("local maxima output image filename"),
		  false, requires_argument);
int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// // Global variables (not options)

// find local maxima and returns volume with zero except at local max
volume<double> label_local_maxima(const volume<double>& vin,
				  const volume<double>& mask)
{
  int lmaxcount=0;
  volume<double> vout;
  vout = 0.0*vin;
  bool usemask( mask.nvoxels() > 0 );
  for (int z=vin.minz(); z<=vin.maxz(); z++) {
    for (int y=vin.miny(); y<=vin.maxy(); y++) {
      for (int x=vin.minx(); x<=vin.maxx(); x++) {
	      if ( !usemask || mask(x,y,z) > 0.5 ) {
	        if (vin(x,y,z)>vin(x-1,y-1,z-1) &&
	          vin(x,y,z)>vin(x,  y-1,z-1) &&
	          vin(x,y,z)>vin(x+1,y-1,z-1) &&
	          vin(x,y,z)>vin(x-1,y,  z-1) &&
	          vin(x,y,z)>vin(x,  y,  z-1) &&
	          vin(x,y,z)>vin(x+1,y,  z-1) &&
	          vin(x,y,z)>vin(x-1,y+1,z-1) &&
	          vin(x,y,z)>vin(x,  y+1,z-1) &&
	          vin(x,y,z)>vin(x+1,y+1,z-1) &&
	          vin(x,y,z)>vin(x-1,y-1,z) &&
	          vin(x,y,z)>vin(x,  y-1,z) &&
	          vin(x,y,z)>vin(x+1,y-1,z) &&
	          vin(x,y,z)>vin(x-1,y,  z) &&
	          vin(x,y,z)>=vin(x+1,y,  z) &&
	          vin(x,y,z)>=vin(x-1,y+1,z) &&
	          vin(x,y,z)>=vin(x,  y+1,z) &&
	          vin(x,y,z)>=vin(x+1,y+1,z) &&
	          vin(x,y,z)>=vin(x-1,y-1,z+1) &&
	          vin(x,y,z)>=vin(x,  y-1,z+1) &&
	          vin(x,y,z)>=vin(x+1,y-1,z+1) &&
	          vin(x,y,z)>=vin(x-1,y,  z+1) &&
	          vin(x,y,z)>=vin(x,  y,  z+1) &&
	          vin(x,y,z)>=vin(x+1,y,  z+1) &&
	          vin(x,y,z)>=vin(x-1,y+1,z+1) &&
	          vin(x,y,z)>=vin(x,  y+1,z+1) &&
	          vin(x,y,z)>=vin(x+1,y+1,z+1) ) {
	            vout(x,y,z) = ++lmaxcount;
	        }
	      }
      }
    }
  }
  return vout;
}

int do_work(int argc, char* argv[])
{
  volume<double> input, dmap;

  read_volume(input,inname.value());
  input.binarise(0.5f,std::numeric_limits<double>::max(),inclusive,invert.value());

  //Sparse interpolate then exit
  if (sparseFile.set()) {
    volume<double> sparseData;
    read_volume(sparseData,sparseFile.value());
    float smoothingSigma(3);
    if (nn.value()) {
      smoothingSigma = 0;
    }
    dmap=sparseInterpolate(sparseData,input,smoothingSigma);
    save_volume4D(dmap,outname.value());
    return 0;
  }

  dmap = sqrt(euclideanDistanceTransform(input));

  //Calculate two-way distancemap
  if (secondname.set()) {
    volume<double> dmap2;
    read_volume(input,secondname.value());
    input.binarise(0.5f,std::numeric_limits<double>::max(),inclusive,invert.value());
    dmap2 = sqrt(euclideanDistanceTransform(input));
    for (int z=0; z<=input.maxz(); z++)
      for (int y=0; y<=input.maxy(); y++)
        for (int x=0; x<=input.maxx(); x++)
          dmap(x,y,z) = dmap(x,y,z) <= dmap2(x,y,z) ? dmap(x,y,z) : -dmap2(x,y,z);
  }

  save_volume(dmap,outname.value());

  //Calculate derived parameters
  if ( labelname.set() ) {
    if (verbose.value())
      cout << "Finding local max" << endl;
    volume<double> label = label_local_maxima(dmap,dmap);
    save_volume(label,labelname.value());
  }

  return 0;
}
////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{

  Tracer tr("main");
  OptionParser options(title, examples);

  try {
    options.add(inname);
    options.add(outname);
    options.add(secondname);
    options.add(labelname);
    options.add(invert);
    options.add(sparseFile);
    options.add(nn);
    options.add(verbose);
    options.add(help);

    nonoptarg = options.parse_command_line(argc, argv);

    // line below stops the program if the help was requested or
    //  a compulsory option was not set
    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }

  }  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }

  // Call the local functions

  return do_work(argc,argv);
}
