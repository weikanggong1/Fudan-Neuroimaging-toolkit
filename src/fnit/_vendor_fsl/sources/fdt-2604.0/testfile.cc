/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include "newimage/newimageall.h"
#include <vector>
using namespace std;
using namespace NEWIMAGE;
/*int read_avg_file (vector<vector<int> >& avgs,const string fname){
  avgs.clear();
  ifstream avg_file(fname.c_str());
  if(!avg_file){return -1;}
  avno=0;
  int start,length;
  string myline;
  else{
      avg_file.getline(myline);
      istringstream mylinestr(myline);
      mylinestr>>start;
      mylinestr>>length;
      cerr<<start<<" "<<length<<endl;
      mylinestr>>start;
      mylinestr>>length;
      cerr<<start<<" "<<length<<endl;

    }
  }
  
  return 0;
}
*/

int main(int argc, char** argv){
  volume4D<float> data;
  //volume<float> mask;
  //  Matrix datam;
  cerr<<"pants"<<endl;
  //  read_volume4D(data,"dat");//data,opts.datafile.value());
  //cerr<<"pants1"<<endl;
  //  read_volume(mask,"mas");//mask,opts.maskfile.value());
  //cerr<<"pants2"<<endl;
  //datam=data.matrix(mask);  
  //cerr<<"pants3"<<endl;
  
  //vector<vector<int> > poo;
  //string af="avg_file";
  //read_avg_file(poo,af);
  Matrix a(2,1);
  a(1,1)=10.2;
  a(2,1)=4.1;
  element_mod_n(a,M_PI);
  cerr<<a<<endl;

}
