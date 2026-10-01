/*  globaloptions.cc

    Peter R Bannister, FMRIB Image Analysis Group

    Copyright (C) 1999-2002 University of Oxford  */

/*  CCOPYRIGHT  */

#include "newimage/costfns.h"
#include "Globaloptions.h"

using namespace std;
using namespace NEWIMAGE;

Globaloptions* Globaloptions::gopt = NULL;

void Globaloptions::parse_command_line(int argc,char** argv)
{
  if (argc < 3) {
    print_usage(argc,argv);
    exit(1);
  }

  int n=1;
  string arg;
  char first;

  while (n<argc) {
    arg=argv[n];
    if (arg.size()<1) { n++; continue; }
    first = arg[0];
    if (first!='-') {
      inputfname = arg;
      n++;
      cerr << "WARNING: change in option usage" << endl << endl;
      cerr << "To specify the input volume the option -in should be used"
	   << endl << "Accepting the filename for now, but please update "
	   << "to new syntax in future." << endl << endl;
      continue;
    }
    // put options without arguments here
    if ( arg == "-help" ) {
      print_usage(argc,argv);
      exit(0);
    } else if ( arg == "-v" ) {
      verbose = 5;
      n++;
      continue;
    } else if ( arg == "-gdt" ) {
      gdtflag = 1;
      n++;
      continue;
    } else if ( arg == "-hist" ) {
      histflag = 1;
      n++;
      continue;
    } else if ( arg == "-stats" ) {
      statflag = 1;
      n++;
      continue;
    } else if ( arg == "-mats" ) {
      tmpmatflag = 1;
      matflag = 1;
      n++;
      continue;
    } else if ( arg == "-plots" ) {
      tmpmatflag = 1;
      plotflag = 1;
      n++;
      continue;
    } else if ( arg == "-meanvol" ) {
      meanvol = 1;
      n++;
      continue;
    } else if ( arg == "-report" ) {
      no_reporting = 0;
      n++;
      continue;
    } else if ( arg == "-fudge" ) {
      fudgeflag = 1;
      n++;
      continue;
    } else if ( arg == "-sinc_final" ) {
      sinc_final = 1;
      n++;
      continue;
      } else if ( arg == "-nn_final" ) {
      nn_final = 1;
      n++;
      continue;
      } else if ( arg == "-spline_final" ) {
      spline_final = 1;
      n++;
      continue;
    } else if ( arg == "-rmsrel" ) {
      tmpmatflag = 1;
      rmsrelflag = 1;
      n++;
      continue;
    } else if ( arg == "-rmsabs" ) {
      tmpmatflag = 1;
      rmsabsflag = 1;
      n++;
      continue;
    } else if ( arg == "-2d" ) {
      twodcorrect = 1;
      n++;
      continue;
    }

    if (n+1>=argc)
      {
	cerr << "Lacking argument to option " << arg << endl;
	exit(-1);
      }

    // put options with 1 argument here
    if (arg == "-in") {
      inputfname = argv[n+1];
      n+=2;
      continue;
    } else if ( (arg == "-o") || (arg == "-out") ) {
      outputfname = argv[n+1];
      n+=2;
      continue;
    } else if ( (arg == "-r") || (arg == "-reffile") ) {
      reffilename = argv[n+1];
      reffileflag = 1;
      n+=2;
      continue;
    } else if (arg == "-init") {
      init_transform = argv[n+1];
      n+=2;
      continue;
    } else if ( arg == "-dof") {
      no_params = atoi(argv[n+1]);
      dof = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-verbose") {
      verbose = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-bins") {
      no_bins = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-refvol") {
      refnum = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-scaling") {
      scaling = atof(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-smooth") {
      smoothsize = atof(argv[n+1]);
      n+=2;
      continue;
    } else if (arg == "-rotation") {
      rot_param = atof(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-stages") {
      no_stages = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-fov") {
      fov = atoi(argv[n+1]);
      n+=2;
      continue;
    } else if ( arg == "-cost") {
      {
	string costarg = argv[n+1];
	if (costarg == "mutualinfo") {
	  maincostfn = MutualInfo;
	} else if (costarg == "corratio") {
	  maincostfn = CorrRatio;
	} else if (costarg == "woods") {
	  maincostfn = Woods;
	} else if (costarg == "normcorr") {
	  maincostfn = NormCorr;
	} else if (costarg == "normmi") {
	  maincostfn = NormMI;
	} else if (costarg == "leastsquares") {
	  maincostfn = LeastSq;
	} else {
	  if (!no_reporting) {
	    cerr << "Unrecognised cost function type: " << costarg << endl;
	    cerr << "Using the default (NormCorr)\n";
	  }
	}
      }
      n+=2;
      continue;

    } else {
      cerr << "Unrecognised option " << arg << endl;
      exit(-1);
    }

    if (n+2>=argc)
      {
	cerr << "Lacking argument to option " << arg << endl;
	exit(-1);
      }
  }

  if (inputfname.size()<1) {
    cerr << "Input filename not found\n\n";
    print_usage(argc,argv);
    exit(2);
  }

  if (outputfname.size()<1) {
    make_basename(inputfname);
    outputfname = inputfname + "_mcf";
    if (!no_reporting)
      cerr << "No output name specified. ";
  }

  if (!no_reporting)
    cerr << "Processed data will be saved as " << outputfname << endl;
}

void Globaloptions::print_usage(int argc, char *argv[])
{
  cout << "Usage: mcflirt -in <infile> [options]\n\n"
       << "  Available options are:\n"
       << "        -out, -o <outfile>               (default is infile_mcf)\n"
       << "        -cost {mutualinfo,woods,corratio,normcorr,normmi,leastsquares}        (default is normcorr)\n"
       << "        -bins <number of histogram bins>   (default is "
       << no_bins << ")\n"
       << "        -dof  <number of transform dofs>   (default is "
       << dof << ")\n"
       << "        -refvol <number of reference volume> (default is no_vols/2)- registers to (n+1)th volume in series\n"
       << "        -reffile, -r <filename>            use a separate 3d image file as the target for registration (overrides refvol option)\n"
       << "        -scaling <num>                             (6.0 is default)\n"
       << "        -smooth <num>                      (1.0 is default - controls smoothing in cost function)\n"
       << "        -rotation <num>                    specify scaling factor for rotation optimization tolerances\n"
       << "        -verbose <num>                     (0 is least and default)\n"
       << "        -stages <number of search levels>  (default is "
       << no_stages << " - specify 4 for final sinc interpolation)\n"
       << "        -fov <num>                         (default is 20mm - specify size of field of view when padding 2d volume)\n"
       << "        -2d                                Force padding of volume\n"
       << "        -sinc_final                        (applies final transformations using sinc interpolation)\n"
       << "        -spline_final                      (applies final transformations using spline interpolation)\n"
       << "        -nn_final                          (applies final transformations using Nearest Neighbour interpolation)\n"
       << "        -init <filename>                   (initial transform matrix to apply to all vols)\n"
       << "        -gdt                               (run search on gradient images)\n"
       << "        -meanvol                           register timeseries to mean volume (overrides refvol and reffile options)\n"
       << "        -stats                             produce variance and std. dev. images\n"
       << "        -mats                              save transformation matricies in subdirectory outfilename.mat\n"
       << "        -plots                             save transformation parameters in file outputfilename.par\n"
       << "        -report                            report progress to screen\n"
       << "        -help\n";
}
