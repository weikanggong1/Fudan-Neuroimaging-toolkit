/*  Globaloptions.h

    Copyright (C) 1999-2001 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef __GLOBALOPTIONS_
#define __GLOBALOPTIONS_

#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "newimage/newimageall.h"
#include "newimage/costfns.h"

enum anglereps { Euler, Quaternion };

class Globaloptions {
 public:
  static Globaloptions& getInstance();
  ~Globaloptions() { delete gopt; }

  std::string inputfname;
  std::string outputfname;
  int verbose;
  int no_params;
  int dof;
  int no_bins;
  int refnum;
  int no_volumes;
  int gdtflag;
  int edgeflag;
  int histflag;
  int matflag;
  int statflag;
  int plotflag;
  int tmpmatflag;
  int no_reporting;
  int costmeas;
  int meanvol;
  int sinc_final;
  int spline_final;
  int nn_final;
  int rmsrelflag;
  int rmsabsflag;
  float scaling;
  short datatype;
  float smoothsize;
  float rot_param;
  int no_stages;
  int fudgeflag;
  float fov;
  int twodcorrect;

  int reffileflag;
  std::string reffilename;

  std::string init_transform;

  NEWIMAGE::costfns maincostfn;
  NEWIMAGE::Costfn *impair;
  NEWMAT::Matrix initmat;
  anglereps anglerep;
  NEWMAT::ColumnVector boundguess;

  void parse_command_line(int argc, char** argv);

 private:
  Globaloptions();

  const Globaloptions& operator=(Globaloptions&);
  Globaloptions(Globaloptions&);

  static Globaloptions* gopt;

  void print_usage(int argc, char *argv[]);

};

inline Globaloptions& Globaloptions::getInstance(){
  if(gopt == NULL)
    gopt = new Globaloptions();

  return *gopt;
}

inline Globaloptions::Globaloptions()
{
  outputfname = "";
  inputfname = "";
  no_params = 6;
  dof = 6;
  no_bins = 256;
  anglerep = Euler;
  verbose = 0;
  maincostfn = NEWIMAGE::NormCorr;
  initmat = NEWMAT::IdentityMatrix(4);
  impair = 0;
  refnum = -1;
  no_volumes = 0;
  gdtflag = 0;
  edgeflag = 0;
  histflag = 0;
  matflag = 0;
  statflag = 0;
  plotflag = 0;
  tmpmatflag = 0;
  meanvol = 0;
  costmeas = 0;
  no_reporting = 1;
  scaling = 6.0;
  smoothsize = 1.0;
  rot_param = 1;
  no_stages = 3;
  sinc_final = 0;
  spline_final = 0;
  nn_final = 0;
  rmsrelflag = 0;
  rmsabsflag = 0;
  fudgeflag = 0;
  fov = 20.0;
  twodcorrect = 0;
  reffileflag = 0;

  init_transform = "";

  boundguess.ReSize(2);
  boundguess << 10.0 << 1.0;

}

#endif
