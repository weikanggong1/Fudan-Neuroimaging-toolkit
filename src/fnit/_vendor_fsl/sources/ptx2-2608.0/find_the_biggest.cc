/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "meshclass/meshclass.h"
#include "newimage/newimageall.h"

#include "csv_mesh.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace mesh;

void biggest_from_volumes(vector<string> innames,string oname){
  vector<volume<float> > tmpvec;
  tmpvec.reserve(innames.size());
  volume<float> tmp;
  cout<<"number of inputs "<<innames.size()<<endl;
  cout<<"Indices"<<endl;
  for(unsigned int i=0;i<innames.size();i++){
    cout<<i+1<<" "<<innames[i]<<endl;
    read_volume(tmp,innames[i]);
    tmpvec.push_back(tmp);
  }
  volume<int> output(tmp.xsize(),tmp.ysize(),tmp.zsize());
  copybasicproperties(tmp,output);output=0;

  for(int z=tmp.minz();z<=tmp.maxz();z++){
    for(int y=tmp.miny();y<=tmp.maxy();y++){
      for(int x=tmp.minx();x<=tmp.maxx();x++){
	RowVector bum(innames.size());
	Matrix bugger;
	ColumnVector index;
	for(unsigned int i=0;i<innames.size();i++ ){
	    bum(i+1)=tmpvec[i](x,y,z);
	}
	bugger=MISCMATHS::max(bum,index);
	if(bum.MaximumAbsoluteValue()!=0)
	  output(x,y,z)=(int)index.AsScalar();
	else
	  output(x,y,z)=0;
      }
    }
  }

  output.setDisplayMaximumMinimum(innames.size(),0);
  save_volume(output,oname);

}

void biggest_from_surfaces(vector<string> innames,string oname){
  vector<CsvMesh> meshes;
  CsvMesh m;int nv=0,nt=0;

  cout<<"number of inputs "<<innames.size()<<endl;
  cout<<"Indices"<<endl;
  for(unsigned int i=0;i<innames.size();i++){
    cout<<i+1<<" "<<innames[i]<<endl;
    nt++;
    m.load(innames[i]);
    meshes.push_back(m);
    // test size
    if(i==0)nv=m.nvertices();
    else{
      if(m.nvertices()!=nv){
	cerr<<"Number of vertices incompatible amongst input"<<endl;
	exit(1);
      }
    }
  }

  Matrix allvals(nv,nt);
  for(unsigned int i=0;i<meshes.size();i++){
    int cnt=1;
    for(int j = 0; j< meshes[i].nvertices();j++){
      allvals(cnt,i+1)=meshes[i].get_pvalue(j);
      cnt++;
    }
  }

  for(int i=1;i<=nv;i++){
    if(allvals.Row(i).MaximumAbsoluteValue()==0){
      m.set_pvalue(i-1,0);
      continue;
    }
    int jmax=1;float vm=0;
    for(int j=1;j<=nt;j++){
      if(j==1)vm=allvals(i,j);
      if(allvals(i,j)>=vm){vm=allvals(i,j);jmax=j;}
    }
    m.set_pvalue(i-1,jmax);
  }

  m.save(oname,meshFileType(innames[0]));

}
bool test_input(const vector<string>& filenames){
  int nsurfs=0,nvols=0;
  for(unsigned int i=0;i<filenames.size();i++){
    if(meshExists(filenames[i])){nsurfs++;}
    else if(fsl_imageexists(filenames[i])){nvols++;}
    else{
      cerr<<"Cannot open "<<filenames[i]<<" for reading"<<endl;
      exit(1);
    }
  }
  if(nvols>0 && nsurfs>0){
    cerr<<"Please use list of EITHER volumes OR surfaces"<<endl;
    exit(1);
  }
  return (nvols>0?true:false);
}
int main ( int argc, char **argv ){
  if(argc<3){
    cerr<<" "<<endl;
    cerr<<"usage: find_the_biggest <lots of volumes/surfaces> output"<<endl;
    cerr<<"  output is index in order of inputs"<<endl;
    cerr<<"  please use EITHER volumes OR surfaces"<<endl;
    cerr<<" "<<endl;
    exit(1);
  }

  vector<string> innames;
  innames.clear();
  for(int i=1;i<=argc-2;i++){
    innames.push_back(argv[i]);
  }
  bool isvols=test_input(innames);
  if(!isvols){
    biggest_from_surfaces(innames,argv[argc-1]);
  }
  else{
    biggest_from_volumes(innames,argv[argc-1]);
  }

  return(0);

}
