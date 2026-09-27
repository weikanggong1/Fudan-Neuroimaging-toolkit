/*  fslselectvols.cc

    Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2012 University of Oxford */

/*  CCOPYRIGHT */

#include <iostream>
#include <string>

#include "utils/options.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

bool file_exists (const std::string& name) {
    ifstream file(name);
    return file.good();
}

void get_vols(Matrix& id_vols,const vector<string>& vols){
  if ( vols.size() == 1 && file_exists(vols[0]) ) {
    id_vols = read_ascii_matrix(vols[0]);
    if( id_vols.Ncols() == 1 )
      id_vols = id_vols.t();
  } else {
    id_vols.ReSize(1, vols.size());
    for(int64_t i=0;i<vols.size();i++)
      id_vols(1,i+1)=std::stoi(vols[i]);
  }
}

string title=string("fslselectvols\n")+
  string(" Select volumes from a 4D time series and output a subset 4D volume\n");
string examples=string("fslselectvols -i <input> -o <output> --vols=0,1,6,7\n")+
  string("fslselectvols -i <input> -o <output> --vols=volumes_list.txt");

Option<bool> help(string("-h,--help"), false,
		string("display this help text"),
		false,no_argument);
Option<string> ifilename(string("-i,--in"), string(""),
		string("\tinput file name (4D image)"),
		true, requires_argument);
Option<string> ofilename(string("-o,--out"), string(""),
		string("output file name (4D image)"),
		true, requires_argument);
Option<vector<string>> vols(string("--vols"), {""},
		string("\tlist of volumes to extract (comma-separated list or ascii file)"),
		true, requires_argument);
Option<bool>   calc_mean(string("-m"), false,
		string("\toutput mean instead of concat"),
		false, no_argument);
Option<bool>   calc_var(string("-v"), false,
		string("\toutput variance instead of concat"),
		false, no_argument);


int main(int argc,char *argv[]){
  OptionParser options(title, examples);
  try{
    options.add(help);
    options.add(ifilename);
    options.add(ofilename);
    options.add(vols);
    options.add(calc_mean);
    options.add(calc_var);
    options.parse_command_line(argc, argv);
    if ( (help.value()) || (!options.check_compulsory_arguments(true)) ){
      options.usage();
      exit(EXIT_FAILURE);
    } else {
      volume4D<float> data, subdata;
      read_volume4D(data,ifilename.value());

      Matrix list;
      get_vols(list,vols.value());

      bool meancalc( calc_mean.value() ), varcalc( calc_var.value() );

      if(!meancalc && !varcalc)
	      subdata.reinitialize(data.xsize(),data.ysize(),data.zsize(),list.Ncols());
      else
	      subdata.reinitialize(data.xsize(),data.ysize(),data.zsize(),1);
      subdata=0;
      copybasicproperties(data,subdata);
      float nImages(list.Ncols()); //float to avoid risk of integer math
      for(int z=0;z<data.zsize();z++)
	      for(int y=0;y<data.ysize();y++)
	        for(int x=0;x<data.xsize();x++) {
	          double v(0),v2(0);
	          for(int i=1;i<=list.Ncols();i++) {
              int index( list(1,i) );
              double value( data(x,y,z,index) );
              if( varcalc || meancalc ) {
	              v += value;
	              v2 += value * value;
              } else
		            subdata(x,y,z,i-1) = value;
	          }
	          if(meancalc)
	            subdata(x,y,z,0)= v/nImages;
	          if(varcalc)
	            subdata(x,y,z,0)=(nImages/(nImages-1)) * (v2/nImages - v*v/(nImages*nImages));
	        }
      save_volume4D(subdata,ofilename.value());
      return 0;
    }
  } catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  catch(std::exception &e) {
    cerr << argv[0] << " failed due to error: " << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  // either mean or variance but not both
  if(calc_mean.value() && calc_var.value()){
    cerr<<" Cannot output mean *and* variance. Please choose one or the other but not both"<<endl;
    exit(EXIT_FAILURE);
  }


}
