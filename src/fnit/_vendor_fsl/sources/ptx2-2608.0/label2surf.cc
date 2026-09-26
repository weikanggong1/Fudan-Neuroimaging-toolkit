/*  label2surf.cc

    Transform a set of label files into a surface

    Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2010 University of Oxford */

/*  CCOPYRIGHT */

#include <string>
#include <vector>
#include <iostream>

#include "utils/options.h"
#include "miscmaths/miscmaths.h"
#include "csv_mesh.h"

using namespace std;
using namespace Utilities;
using namespace MISCMATHS;

string title="label2surf \n\t Transform a group of labels into a surface";
string examples="label2surf -s <surface> -o <outputsurface> -l <labels>";

Option<bool> verbose(string("-v,--verbose"),false,
                     string("switch on diagnostic messages"),
                     false,no_argument);
Option<bool> help(string("-h,--help"),false,
                  string("display this message"),
                  false,no_argument);
Option<string> isurf(string("-s,--surf"),"",
                     string("input surface"),
                     true,requires_argument);
Option<string> osurf(string("-o,--out"),"",
                     string("output surface"),
                     true,requires_argument);
Option<string> labels(string("-l,--labels"),"",
                      string("ascii list of label files"),
                      true,requires_argument);


void read_fnames(vector<string>& fnames,const string& filename){
  fnames.clear();
  ifstream fs(filename.c_str());
  string tmp;
  if(fs){
    fs>>tmp;
    do{
      fnames.push_back(tmp);
      fs>>tmp;
    }while(!fs.eof());
  }
  else{
    cerr<<filename<<" does not exist"<<endl;
    exit(0);
  }
}

void read_label(vector<int>& IDs,const string& labelfile){
  IDs.clear();
  ifstream fs(labelfile.c_str());
  if (!fs) {
    cerr << "Could not open label file " << labelfile << endl;
    exit(1);
  }
  string cline;
  // skip first line
  std::getline(fs, cline);
  // skip header
  cline = skip_alpha(fs);
  // read number of vertices
  string ss="";
  fs >> ss;
  float nrows = atof(ss.c_str());
  for(int r=1;r<=nrows;r++){
    for(int c=1;c<=5;c++){
      if(!fs.eof()){
        fs >> ss;
        while ( !isNumber(ss) && !fs.eof() ) {
          fs >> ss;
        }
        if(c==1){IDs.push_back(atoi(ss.c_str()));}
      }
    }
  }
}


int main(int argc,char *argv[]){
  OptionParser options(title,examples);

  options.add(verbose);
  options.add(help);
  options.add(isurf);
  options.add(osurf);
  options.add(labels);

  options.parse_command_line(argc,argv);
  if ( (help.value()) || (!options.check_compulsory_arguments(true)) ){
    options.usage();
    return 1;
  }

  ////////
  if(verbose.value())
    cout<<"read input surface"<<endl;

  CsvMesh m;
  m.load(isurf.value());
  m.reset_pvalues();
  m.reset_tvalues();

  if(verbose.value())
    cout<<"read input labels"<<endl;

  vector<string> labs;
  vector<int>    IDs;
  read_fnames(labs,labels.value());
  for(unsigned int i=0;i<labs.size();i++){
    if(verbose.value())
      cout<<"   label " << i+1 << endl;
    read_label(IDs,labs[i]);
    m.set_pvalues(IDs,1);
  }

  m.save(osurf.value());

  return 0;
}
