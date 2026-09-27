/*  fslcomplex.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2000-2008 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <string>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

void print_usage(const string& progname) {
  cout << "Usage: " << progname << " <-outputtype> <input> <output> [startvol [endvol]]" << endl << endl;

  cout << "       " << progname << " -realabs complexvol absvol"
       << " [startvol [endvol]]" << endl;
  cout << "       " << progname << " -realphase complexvol phasevol"
       << " [startvol [endvol]]" << endl;
  cout << "       " << progname << " -realpolar complexvol absvol phasevol"
       << " [startvol [endvol]]" << endl;
  cout << "       " << progname << " -realcartesian complexvol realvol"
       << " imagvol [startvol [endvol]]" << endl;
  cout << "       " << progname << " -complex realvol imagvol"
       << " complexvol [startvol [endvol]]" << endl;
  cout << "       " << progname << " -complexpolar absvol phasevol"
       << " complexvol [startvol [endvol]]" << endl;
  cout << "       " << progname << " -complexsplit source dest"
       << " [startvol [endvol]]" << endl;
  cout << "       " << progname << " -complexmerge source1 source2 dest" <<endl;
  cout << "       " << progname << " -copyonly source dest" << endl;
}

void fix_start_and_end(int& start, int& end, int mint, int maxt)
{
  if (start<0) { start=mint; }
  if (start<mint) start=mint;
  if (start>maxt) start=maxt;
  if (end<0) { end=maxt; }
  if (end<mint) end=mint;
  if (end>maxt) end=maxt;
}


void realpolar(const string& fin, const string& fabs, const string& fphase,
	       int start, int end)
{
  volume<float> vreal, vimag;
  read_complexvolume(vreal, vimag, fin);
  fix_start_and_end(start,end,vreal.mint(),vreal.maxt());

  if (fabs.size()>0) {
    volume<float> vabs;
    for (int n=start; n<=end; n++) {
      vabs.addvolume(abs(vreal[n],vimag[n]));
    }
    vabs.copyproperties(vreal);
    save_volume(vabs,fabs);
  }

  if (fphase.size()>0) {
    volume<float> vphase;
    for (int n=start; n<=end; n++) {
      vphase.addvolume(phase(vreal[n],vimag[n]));
    }
    vphase.copyproperties(vreal);
    save_volume4D(vphase,fphase);
  }
}

void realcartesian(const string& fin, const string& freal, const string& fimag,
		   int start, int end)
{
  volume<float> vreal, vimag;
  read_complexvolume(vreal, vimag, fin);
  fix_start_and_end(start,end,vreal.mint(),vreal.maxt());

  if (freal.size()>0) {
    volume<float> vre;
    for (int n=start; n<=end; n++) {
      vre.addvolume(vreal[n]);
    }
    vre.copyproperties(vreal);
    save_volume(vre,freal);
  }

  if (fimag.size()>0) {
    volume<float> vim;
    for (int n=start; n<=end; n++) {
      vim.addvolume(vimag[n]);
    }
    vim.copyproperties(vimag);
    save_volume(vim,fimag);
  }
}


void complexsave(const string& freal, const string& fimag,
		 const string& fcomplex, int start, int end)
{
  volume<float> vreal, vimag;
  read_volume4D(vreal, freal);
  read_volume4D(vimag, fimag);
  fix_start_and_end(start,end,Max(vreal.mint(),vimag.mint()),
		    Min(vreal.maxt(),vimag.maxt()));

  if (fcomplex.size()>0) {
    volume<float> vre, vim;
    for (int n=start; n<=end; n++) {
      vre.addvolume(vreal[n]);
      vim.addvolume(vimag[n]);
    }
    vre.copyproperties(vreal);
    vim.copyproperties(vimag);
    save_complexvolume(vre,vim,fcomplex);
  }
}


void complexpolar(const string& fabsvol, const string& fphasevol,
		  const string& fcomplex, int start, int end)
{
  volume<float> vabs, vphase;
  read_volume4D(vabs, fabsvol);
  read_volume4D(vphase, fphasevol);
  fix_start_and_end(start,end,Max(vabs.mint(),vphase.mint()),
		    Min(vabs.maxt(),vphase.maxt()));

  if (fcomplex.size()>0) {
    volume<float> vre, vim;
    for (int n=start; n<=end; n++) {
      vre.addvolume(real(vabs[n],vphase[n]));
      vim.addvolume(imag(vabs[n],vphase[n]));
    }
    vre.copyproperties(vabs);
    vim.copyproperties(vphase);
    save_complexvolume(vre,vim,fcomplex);
  }
}

void complexsplit(const string& fsource, const string& fdest,
		  int start, int end)
{
  volume4D<float> vreal, vimag;
  read_complexvolume(vreal, vimag, fsource);
  fix_start_and_end(start,end,Max(vreal.mint(),vimag.mint()),
		    Min(vreal.maxt(),vimag.maxt()));

  if (fdest.size()>0) {
    volume4D<float> vre, vim;
    for (int n=start; n<=end; n++) {
      vre.addvolume(vreal[n]);
      vim.addvolume(vimag[n]);
    }
    vre.copyproperties(vreal);
    vim.copyproperties(vimag);
    save_complexvolume(vre,vim,fdest);
  }
}

void complexmerge(const string& fsource1, const string& fsource2,
		  const string& fdest)
{
  volume<float> vr1, vi1, vr2, vi2;
  read_complexvolume(vr1, vi1, fsource1);
  read_complexvolume(vr2, vi2, fsource2);

  if (!samesize(vr1,vi1)) {
    cerr << "Could not process image " << fsource1 << " correctly." << endl;
    return;
  }
  if (!samesize(vr2,vi2)) {
    cerr << "Could not process image " << fsource2 << " correctly." << endl;
    return;
  }
  if (fdest.size()>0) {
    volume4D<float> vdr, vdi;
    for (int n=vr1.mint(); n<=vr1.maxt(); n++) {
      vdr.addvolume(vr1[n]);
      vdi.addvolume(vi1[n]);
    }
    for (int n=vr2.mint(); n<=vr2.maxt(); n++) {
      vdr.addvolume(vr2[n]);
      vdi.addvolume(vi2[n]);
    }
    vdr.copyproperties(vr1);
    vdi.copyproperties(vr2);
    save_complexvolume(vdr,vdi,fdest);
  }
}


void copyonly(const string& fsource, const string& fdest)
{
  volume4D<float> vreal, vimag;
  read_complexvolume(vreal, vimag, fsource);
  save_complexvolume(vreal, vimag, fdest);
}


int main(int argc,char *argv[])
{

  Tracer tr("main");

  string progname=argv[0];
  if (argc<4) {
    print_usage(progname);
    return -1;
  }

  string arg = argv[1];
  int start=-1, end=-1;

  if (arg=="-realabs") {
    if (argc<4) { print_usage(progname); return -1; }
    if (argc>=5)  start=atoi(argv[4]);
    if (argc>=6)  end=atoi(argv[5]);
    realpolar(argv[2],argv[3],"",start,end);
  } else if (arg=="-realphase") {
    if (argc<4) { print_usage(progname); return -1; }
    if (argc>=5)  start=atoi(argv[4]);
    if (argc>=6)  end=atoi(argv[5]);
    realpolar(argv[2],"",argv[3],start,end);
  } else if (arg=="-realpolar") {
    if (argc<5) { print_usage(progname); return -1; }
    if (argc>=6)  start=atoi(argv[5]);
    if (argc>=7)  end=atoi(argv[6]);
    realpolar(argv[2],argv[3],argv[4],start,end);
  } else if (arg=="-realcartesian") {
    if (argc<5) { print_usage(progname); return -1; }
    if (argc>=6)  start=atoi(argv[5]);
    if (argc>=7)  end=atoi(argv[6]);
    realcartesian(argv[2],argv[3],argv[4],start,end);
  } else if (arg=="-complex") {
    if (argc<5) { print_usage(progname); return -1; }
    if (argc>=6)  start=atoi(argv[5]);
    if (argc>=7)  end=atoi(argv[6]);
    complexsave(argv[2],argv[3],argv[4],start,end);
  } else if (arg=="-complexpolar") {
    if (argc<5) { print_usage(progname); return -1; }
    if (argc>=6)  start=atoi(argv[5]);
    if (argc>=7)  end=atoi(argv[6]);
    complexpolar(argv[2],argv[3],argv[4],start,end);
  } else if (arg=="-complexsplit") {
    if (argc<4) { print_usage(progname); return -1; }
    if (argc>=5)  start=atoi(argv[4]);
    if (argc>=6)  end=atoi(argv[5]);
    complexsplit(argv[2],argv[3],start,end);
  } else if (arg=="-complexmerge") {
    if (argc<5) { print_usage(progname); return -1; }
    complexmerge(argv[2],argv[3],argv[4]);
  } else if (arg=="-copyonly") {
    if (argc<4) { print_usage(progname); return -1; }
    copyonly(argv[2],argv[3]);
  } else {
    print_usage(progname); return -1;
  }

  return 0;
}
