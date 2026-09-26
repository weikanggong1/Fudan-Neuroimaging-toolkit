/*    Copyright (C) 2012 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include "utils/options.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "warpfns/warpfns.h"
#include "warpfns/fnirt_file_reader.h"
#include "csv.h"
#include "stdlib.h"
#include "string.h"


using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;


string title="surf2surf - conversions between surface formats and/or conventions";
string examples="Usage: surf2surf -i <inputSurface> -o <outputSurface> [options]";


Option<string> surfin(string("-i,--surfin"),string(""),
		      string("input surface"),
		      true,requires_argument);
Option<string> surfout(string("-o,--surfout"),string(""),
		       string("output surface"),
		       true,requires_argument);
Option<string> convin(string("--convin"),string("caret"),
		      string("input convention [default=caret] - only used if output convention is different"),
		      false,requires_argument);
Option<string> convout(string("--convout"),string("caret"),
		       string("output convention [default=same as input]"),
		       false,requires_argument);
Option<string> volin(string("--volin"),string(""),
		     string("\tinput ref volume - Must set this if changing conventions"),
		     false,requires_argument);
Option<string> volout(string("--volout"),string(""),
		      string("output ref volume [default=same as input]"),
		      false,requires_argument);
Option<string> xfm(string("--xfm"),"",
		   string("\tin-to-out ascii matrix or out-to-in warpfield [default=identity]"),
		   false,requires_argument);
Option<string> otype(string("--outputtype"),"GIFTI_BIN_GZ",
		     string("output type: ASCII, VTK, GIFTI_ASCII, GIFTI_BIN, GIFTI_BIN_GZ (default)"),
		     false,requires_argument);
Option<string> vals(string("--values"),"",
		     string("set output scalar values (e.g. --values=mysurface.func.gii or --values=1)"),
		     false,requires_argument);




int main(int argc,char *argv[]){

  OptionParser options(title,examples);


  options.add(surfin);
  options.add(surfout);
  options.add(convin);
  options.add(convout);
  options.add(volin);
  options.add(volout);
  options.add(xfm);
  options.add(otype);
  options.add(vals);


  options.parse_command_line(argc,argv);
  if(!options.check_compulsory_arguments(true)){
    options.usage();
    return(1);
  }

  // check options
  if(convin.value()!=convout.value()){
    if(volin.value()==""){
      cerr<<"Please specify input reference volume"<<endl;
      return(1);
    }
  }

  volume<short int> refvolin,refvolout;
  CSV csv;
  if(volin.set()){
    read_volume(refvolin,volin.value());
    csv.reinitialize(refvolin);
    csv.set_convention(convin.value());
  }
  if(volout.set()){
    read_volume(refvolout,volout.value());
  }
  else{
    refvolout.reinitialize(refvolin.xsize(),refvolin.ysize(),refvolin.zsize());
    refvolout=refvolin;
  }

  csv.load_rois(surfin.value());
  //csv.get_mesh(0).print("grot.txt");

  if(convin.value()!=convout.value()){
    bool isWarp=false;
    volume4D<float> vox2vox_warp;
    Matrix          vox2vox;
    ColumnVector old_dims(3);
    old_dims << refvolin.xdim() << refvolin.ydim() << refvolin.zdim();
    if(xfm.set()){
      if(fsl_imageexists(xfm.value())){
	isWarp=true;
	FnirtFileReader ffr(xfm.value());
	vox2vox_warp=ffr.FieldAsNewimageVolume4D(true);
      }
      else{
	vox2vox=read_ascii_matrix(xfm.value());
      }
    }
    else
      vox2vox=IdentityMatrix(4);

    csv.set_refvol(refvolout);
    if(convout.set()){
      if(!isWarp)
	csv.switch_convention(convout.value(),vox2vox,old_dims);
      else
	csv.switch_convention(convout.value(),vox2vox_warp,refvolin,refvolout);
    }
    else{
      if(!isWarp)
	csv.switch_convention(convin.value(),vox2vox,old_dims);
      else
	csv.switch_convention(convin.value(),vox2vox_warp,refvolin,refvolout);
    }
  }

  if(vals.value()!=""){
    if(meshExists(vals.value())){
      CsvMesh m;
      m.load(vals.value());
      //OUT(m._pvalues.size());
      //OUT(csv.get_mesh(0).nvertices());
      if((int)m._pvalues.size()!=csv.get_mesh(0).nvertices()){
	cerr<<"values mesh does not have the correct number of vertices"<<endl;
	exit(1);
      }
      vector<float> v = m.getValuesAsVectors();
      ColumnVector vv(v.size());
      for(unsigned int i=0;i<v.size();i++)
	vv(i+1)=v[i];
      csv.get_mesh(0).set_pvalues(vv);
    }
    else{
      csv.get_mesh(0).set_pvalues((float)atoi(vals.value().c_str()));
    }
  }

  //csv.save_roi(0,surfout.value());
  int o;
  if(otype.value()=="ASCII")
    o=CSV_ASCII;
  else if(otype.value()=="VTK")
    o=CSV_VTK;
  else if(otype.value()=="GIFTI_ASCII")
    o=GIFTI_ENCODING_ASCII;
  else if(otype.value()=="GIFTI_BIN")
    o=GIFTI_ENCODING_B64BIN;
  else if(otype.value()=="GIFTI_BIN_GZ")
    o=GIFTI_ENCODING_B64GZ;
  else{
    cerr<<"Unknown format "<<otype.value()<<endl;
    exit(1);
  }
  csv.get_mesh(0).save(surfout.value(),o);


  return 0;
}
