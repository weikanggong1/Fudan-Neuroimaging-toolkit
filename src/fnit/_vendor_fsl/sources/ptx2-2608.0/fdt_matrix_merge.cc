/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <vector>
#include <string>
#include <iostream>

#include "miscmaths/miscmaths.h"
#include "miscmaths/SpMat.h"
#include "streamlines.h"

using namespace std;
using namespace MISCMATHS;

void read_file_names(const string& filename,vector<string>& filelist){
  ifstream fs(filename.c_str());
  string tmp;
  if(fs){
    fs>>tmp;
    do{
      filelist.push_back(tmp);
      fs>>tmp;
    }while(!fs.eof());
    }
  else{
    cerr<<filename<<" does not exist"<<endl;
    exit(0);
  }
}

int main(int argc, char** argv){

  if(argc<2){
    cout<<"fdt_matrix_merge <matlist> <outfile> [-omatrix4]"<<endl;
    exit(1);
  }

  vector<string> filelist;
  read_file_names(argv[1],filelist);
  bool domat4 = false;

  if( argc>3 ){ if( string(argv[3])=="-omatrix4"){ domat4=true; } }

  cout<<"merge "<<filelist.size()<<" files"<<endl<<endl;
  if( !domat4 ){
    SpMat<float> omat(filelist[0]);
    SpMat<float>* tmpmat;
    for(unsigned int i=1;i<filelist.size();i++){
      cout<<filelist[i]<<endl;
      cout<<"...load new file"<<endl;
      tmpmat = new SpMat<float>(filelist[i]);
      omat += *tmpmat;
      delete(tmpmat);
   }
    cout<<"save"<<endl;
    omat.Print(argv[2]);
  }
  else{
    cout<<"Matrix4 merge"<<endl;
    string file1=filelist[0]+"1.mtx";
    ifstream fs1(file1.c_str());
    if (!fs1) {
      cerr << "Could not open file " << file1 << " for reading" << endl;
      return -1;
    }
    int nrows, ncols;
    fs1 >> nrows;
    fs1 >> ncols;
    fs1.close();
    SpMat_HCP omat(nrows,ncols,filelist[0]);
    SpMat_HCP* tmpmat;

    //int testcol=15442, testrow=135590;
    for(unsigned int i=1;i<filelist.size();i++){
      cout<<filelist[i]<<endl;
      cout<<"...loading new file"<<endl;
      tmpmat = new SpMat_HCP(nrows,ncols,filelist[i]);
      cout<<"...increment"<<endl;
      //tmpmat->Print(testrow,testcol);
      omat += (*tmpmat);
      //omat.Print(testrow,testcol);
      delete(tmpmat);
    }
    cout<<"save"<<endl;
    omat.SaveTrajFile(argv[2]);
  }
  return 0;
}
