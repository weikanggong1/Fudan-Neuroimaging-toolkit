/*    Copyright (C) 2012 University of Oxford  */

/*  CCOPYRIGHT  */
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <cstdlib>
#include <string>
#include <iostream>

#include "utils/options.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "warpfns/warpfns.h"
#include "warpfns/fnirt_file_reader.h"
#include "newimage/newimageall.h"

#include "csv.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;



string title="";
string examples="";


Option<string> datafile(string("--data"),string(""),
		   string("data to project onto surface"),
		   true,requires_argument);
Option<string> surf(string("--surf"),string(""),
		    string("surface file"),
		    true,requires_argument);
Option<string> surfvol(string("--meshref"),string(""),
		    string("surface volume ref (default=same as data)"),
		    false,requires_argument);
Option<string> xfm(string("--xfm"),string(""),
		   string("data2surf transform (default=Identity)"),
		   false,requires_argument);
Option<string> meshspace(string("--meshspace"),string("caret"),
		   string("meshspace (default='caret')"),
		   false,requires_argument);
Option<string> out(string("--out"),string(""),
		   string("output file"),
		   true,requires_argument);
Option<float> dist(string("--step"),1,
		   string("average over step (mm - default=1)"),
		   false,requires_argument);
Option<float> dir(string("--direction"),0,
		  string("if>0 goes towards brain (default=0 ie both directions)"),
		  false,requires_argument);
Option<string> operation(string("--operation"),"mean",
			 string("what to do with values: 'mean' (default), 'max', 'median', 'last'"),
			 false,requires_argument);
Option<bool> surfout(string("--surfout"),false,
			 string("output surface file, not ascii matrix (valid only for scalars)"),
			 false,no_argument);

float calc_mean(const vector<float>& vec){
  float ret=0;
  if(vec.size()==0)
    return ret;
  for(unsigned int i=0;i<vec.size();i++)
    ret+=vec[i];
  return(ret/float(vec.size()));
}
float calc_max(const vector<float>& vec){
  float ret=0,tmp;
  if(vec.size()==0)
    return ret;
  for(unsigned int i=0;i<vec.size();i++){
    tmp=vec[i];
    if(tmp>=ret||i==0)ret=tmp;
  }
  return ret;
}

float calc_median(vector<float> vec){
  float median;
  size_t size = vec.size();

  sort(vec.begin(),vec.end());

  if (size  % 2 == 0)
  {
      median = (vec[size / 2 - 1] + vec[size / 2]) / 2;
  }
  else
  {
      median = vec[size / 2];
  }

  return median;

}
float calc_last(const vector<float>& vec){
  return vec[ vec.size()-1 ];
}

// debug
void outvals(const vector<float>& vals){
  cout<<endl;
  for(unsigned int i=0;i<vals.size();i++)
    cout<<vals[i]<<" ";
  cout<<endl;
}

int main(int argc,char *argv[]){

  OptionParser options(title,examples);

  options.add(datafile);
  options.add(surf);
  options.add(surfvol);
  options.add(xfm);
  options.add(meshspace);
  options.add(out);
  options.add(dist);
  options.add(dir);
  options.add(operation);
  options.add(surfout);


  options.parse_command_line(argc,argv);
  if(!options.check_compulsory_arguments(true)){
    options.usage();
    return(1);
  }

  cout<<"read data"<<endl;
  volume4D<float> data;
  read_volume4D(data,datafile.value());

  volume<short int> refvol;
  if(surfvol.value()!="")
    read_volume(refvol,surfvol.value());
  else
    copyconvert(data[0],refvol);

  cout<<"read surface"<<endl;
  CSV csv(refvol);
  csv.set_convention(meshspace.value());
  csv.load_rois(surf.value());


  ColumnVector surfdim(3),datadim(3);
  surfdim << refvol.xdim() << refvol.ydim() << refvol.zdim();
  datadim << data[0].xdim()<<data[0].ydim()<<data[0].zdim();

  bool isWarp=false;
  volume4D<float> data2surf_warp;
  Matrix          surf2data,data2surf;
  if(xfm.value() != ""){
    if(fsl_imageexists(xfm.value())){
      isWarp=true;
      FnirtFileReader ffr(xfm.value());
      data2surf_warp = ffr.FieldAsNewimageVolume4D(true);
    }
    else{
      data2surf = read_ascii_matrix(xfm.value());
      surf2data = data2surf.i();
    }
  }
  else{
    surf2data = IdentityMatrix(4);
  }

  float step = dist.value(); // mm

  cout<<"project data"<<endl;
  // loop over surface vertices
  // for each vertex, go along normal for a fixed distance, and average the data


  Matrix surfdata(csv.get_mesh(0).nvertices(),data.tsize());
  surfdata=0;

  ColumnVector x(3),n(3),data_vox(3),nn(3);
  int ix,iy,iz;
  int nsteps=10;
  for(int i=0;i<csv.get_mesh(0).nvertices();i++){

    x=csv.get_vertex_as_vox(0,i);
    n=csv.get_normal_as_vox(0,i);
    if(dir.value()<0) n*=-1;

    if(n.MaximumAbsoluteValue()>0)
      n/=sqrt(n.SumSquare());

    ColumnVector xx(3);
    vector< vector<float> > vals(data.tsize());
    //OUT(step);
    //OUT(n.t());
    for(int s=0;s<nsteps;s++){
      xx<<x(1)+(float)s/float(nsteps-1.0)*step/surfdim(1)*n(1)
	<<x(2)+(float)s/float(nsteps-1.0)*step/surfdim(2)*n(2)
	<<x(3)+(float)s/float(nsteps-1.0)*step/surfdim(3)*n(3);
      //OUT(xx.t());
      if(!isWarp)
	data_vox = vox_to_vox(xx,surfdim,datadim,surf2data);
      else
	data_vox = NewimageCoord2NewimageCoord(data2surf_warp,false,refvol,data[0],xx);

      ix = (int)MISCMATHS::round((float)data_vox(1));
      iy = (int)MISCMATHS::round((float)data_vox(2));
      iz = (int)MISCMATHS::round((float)data_vox(3));

      for(int j=0;j<data.tsize();j++){
	vals[j].push_back(data(ix,iy,iz,j));
      }

      if(dir.value()==0){
	xx<<x(1)-(float)s/float(nsteps-1.0)*step/surfdim(1)*n(1)
	  <<x(2)-(float)s/float(nsteps-1.0)*step/surfdim(2)*n(2)
	  <<x(3)-(float)s/float(nsteps-1.0)*step/surfdim(3)*n(3);

	if(!isWarp)
	  data_vox = vox_to_vox(xx,surfdim,datadim,surf2data);
	else
	  data_vox = NewimageCoord2NewimageCoord(data2surf_warp,false,refvol,data[0],xx);

	ix = (int)MISCMATHS::round((float)data_vox(1));
	iy = (int)MISCMATHS::round((float)data_vox(2));
	iz = (int)MISCMATHS::round((float)data_vox(3));

	for(int j=0;j<data.tsize();j++){
	  vals[j].push_back(data(ix,iy,iz,j));
	}

      }
    }

    for(int j=1;j<=data.tsize();j++){
      if(operation.value()=="mean"){
	surfdata(i+1,j) = calc_mean(vals[j-1]);
      }
      else if(operation.value()=="median")
	surfdata(i+1,j) = calc_median(vals[j-1]);
      else if(operation.value()=="max")
	surfdata(i+1,j) = calc_max(vals[j-1]);
      else if(operation.value()=="last")
	surfdata(i+1,j) = calc_last(vals[j-1]);
      else{
	cerr<<"Unrecognised operation " << operation.value()<<endl;
	exit(1);
      }

    }

  }


  cout<<"save"<<endl;

  //for(int t=0;t<data.tsize();t++){
  //osurfs[t]->save_roi(0,out.value()+"_"+num2str(t+1)+".asc");
  //}

  if(!surfout.value())
    write_ascii_matrix(surfdata,out.value());
  else{
    csv.get_mesh(0).set_pvalues(surfdata.Column(1));
    csv.get_mesh(0).save(out.value());
  }

  return 0;
}
