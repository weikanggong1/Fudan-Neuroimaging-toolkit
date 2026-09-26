/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include "newimage/newimageall.h"
#include <vector>
using namespace std;
using namespace NEWIMAGE;
using namespace NEWMAT;







int main ( int argc, char **argv ){
 if(argc<6){
    cerr<<"usage: sausages <refvol> <coordfile> <output> <start1> <end1> ... <startn> <endn>"<<endl;
    exit(0);
  }
 
 
 
 volume<int> out;
 read_volume(out,argv[1]);
 out*=0;

 volume<int> coord;
 read_volume(coord,argv[2]);
 vector<pair<int,int> > chunks;
 if(argc/2 != argc/2.0f ){
   cerr<<"your chunks are not in pairs"<<endl;
   return -1;
 }
 pair<int,int> tmppair;
 for(int i=4;i<argc;i+=2){
   int tmp=atoi(argv[i]);
   if(tmp>coord.xsize()){
     cerr<<"There aren't "<<argv[i]<<" elements in "<<argv[2]<<endl; 
     return -1;
   }
   else{
     tmppair.first=tmp;
   }
   tmp=atoi(argv[i+1]);
   if(tmp>coord.xsize()){
     cerr<<"There aren't "<<argv[i+1]<<" elements in "<<argv[2]<<endl; 
     return -1;
   }
   else{
     tmppair.second=tmp;
   }
   chunks.push_back(tmppair);
 }
 
 for(unsigned int chunkno=0;chunkno < chunks.size();chunkno++){
   for(int i=chunks[chunkno].first;i<=chunks[chunkno].second;i++){
     out(coord(i,0,0),coord(i,1,0),coord(i,2,0))=chunkno+1;
   }
 }
 
 save_volume_dtype(out,argv[3],dtype(string(argv[1])));
 return 0;
}
 









