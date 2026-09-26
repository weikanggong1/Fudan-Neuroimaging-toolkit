/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace MISCMATHS;
using namespace NEWIMAGE;

int read_avg_file (vector<vector<int> >& avgs,const string fname){
  avgs.clear();
  ifstream avg_file(fname.c_str());
  string myline;
  bool nobbsize=true;

  int row = 0;

  if(!avg_file){return -1;}
  else{
    while(nobbsize){
      avgs.push_back(vector<int>());

      nobbsize=false;
      getline(avg_file,myline);

      int pos=0;
      while(pos!=int(string::npos)) {
	pos = myline.find(",",pos);
	if(pos!=int(string::npos)){
	  myline.replace(pos,1," ");
	  pos++;
	}
      }

      istringstream mylinestr(myline.c_str());

      while(!mylinestr.eof()){

	string startstr;
	int start;
	int length;
	mylinestr >> startstr;
	if(isNumber(startstr)){
	  nobbsize=true;
	  start = atoi(startstr.c_str());
	  mylinestr >> length;

	  //cerr<< start <<" " << length << " ";

	  for(int i=start; i<start+length; i++)
	    {
	      avgs[row].push_back(i);
	    }
	}
      }
      row++;
      cerr<<endl;
    }
  }


  row--;
  avgs.pop_back();

//   for(int r=0;r<row;r++)
//     {
//       cout << "size=" << avgs[r].size() << endl;
//       for(int c=0;c<avgs[r].size();c++)
// 	cout << avgs[r][c] << " ";
//       cout << endl;
//     }

  return 0;
}


int main ( int argc, char **argv ){
  if(argc<5){
    cerr<<"usage: replacevols input4D avg_file volno volno ... volno output4D"<<endl;
    cerr<<"volno starts at zero!!!"<<endl;
    exit(0);
  }

  volume4D<float> data4D;
  read_volume4D(data4D,argv[1]);
  vector<vector<int> > avgs;
  int ret;
  ret = read_avg_file(avgs, argv[2]);
  if(ret==-1){
    cerr<<"can't find "<<argv[2]<<endl;
    return -1;
  }

  vector<int> repvols;
  for(int i=3;i<=argc-2;i++){
    repvols.push_back(atoi(argv[i]));
  }


  for(unsigned int j=0;j<avgs[0].size();j++){//loop over volume numbers


    //Next loop is within volume number over averages just
    // Working out which ones to replace and which to keep.

    vector<int> repthis,keepthis;
    for(unsigned int i=0;i<avgs.size();i++){ //loop over averages
      bool replaced=false;
      for(unsigned int r=0;r<repvols.size();r++){// loop over things to be replaced
	if(avgs[i][j]==repvols[r]){
	  replaced=true;
	  repthis.push_back(avgs[i][j]);
	}
      }
      if(!replaced){
	keepthis.push_back(avgs[i][j]);
      }

    }


    if(repthis.size()>0){

      cerr<<"Replacing volumes: ";
      for(unsigned int r=0;r<repthis.size();r++){
	cerr<<repthis[r]<<" ";
      }
      cerr <<endl;
      cerr<<"with the average of volumes: ";
      for(unsigned int r=0;r<keepthis.size();r++){
	cerr<<keepthis[r]<<" ";
      }
      cerr<<endl;


      if(keepthis.size()>0){
	// Next loop makes the average of all the ones we are keeping
	volume<float> tmp;
	tmp=data4D[keepthis[0] ];

	for(unsigned int n=1;n<keepthis.size();n++){
	  tmp=tmp+data4D[keepthis[n] ];
	}
	tmp=tmp/keepthis.size(); //Average of all non-replaced ones.



	//Next loop replaces all the ones to be replaced with this average
	for(unsigned int n=0;n<repthis.size();n++){
	  data4D[repthis[n] ]=tmp; //replacing.
	}


      }
      else{
	cerr<<"Error: Volume number "<<j<<" has no averages to keep!!"<<endl;;
	return -1;
      }
    }//repthis.size>0


  }// loop over volume numbers




  //   vector<volume<float> > tmpvec;
//   float test=(argc-3)/2.0f;
//   if(round(test)!=test){
//     cerr<<"Either you have different nums of volnos and vols, or you haven't specified input or output"<<endl;
//     return(0);
//   }

//   //  int numchanges=(argc-3)/2;
//   tmpvec.reserve((argc-3)/2);
//   vector<int> volnos;
//   volume<float> tmp;
//   tmpvec.reserve((argc-3)/2);

//   cout<<"number of vols to be replaced "<<(argc-3)/2<<endl;
//   for(int i=2;i<=(argc-2);i+=2){
//     volnos.push_back(atoi(argv[i]));
//     read_volume(tmp,argv[i+1]);
//     tmpvec.push_back(tmp);
//   }

//   for(int i=0;i<(int)volnos.size();i++){
//     int num=volnos[i];
//     data4D[num]=tmpvec[i];
//   }

  save_volume4D(data4D,argv[argc-1]);
}
