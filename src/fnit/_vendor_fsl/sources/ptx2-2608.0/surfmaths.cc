/*   surfmaths.cc
     uniary and binary operations from fslmaths applied to surfaces

     Saad Jbabdi and Matthew Webster, FMRIB Image Analysis Group

     Copyright (C) 2012-2014 University of Oxford  */


/* CCOPYRIGHT */

#include <string>
#include <iostream>

#include "newmesh/newmesh.h"
#include "miscmaths/miscmaths.h"
#include "utils/fsl_isfinite.h"
#include "cprob/libprob.h"

using namespace std;
using namespace MISCMATHS;
using namespace NEWMESH;


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

int printUsage(const string& programName)
{
  cout << "\nUsage: surfmaths <first_input> [operations and inputs] <output> " << endl;

  cout << "\nBasic unary operations:" << endl;
  cout << " -exp   : exponential" << endl;
  cout << " -log   : natural logarithm" << endl;
  cout << " -sin   : sine function" << endl;
  cout << " -cos   : cosine function" << endl;
  cout << " -tan   : tangent function" << endl;
  cout << " -asin  : arc sine function" << endl;
  cout << " -acos  : arc cosine function" << endl;
  cout << " -atan  : arc tangent function" << endl;
  cout << " -sqr   : square" << endl;
  cout << " -sqrt  : square root" << endl;
  cout << " -recip : reciprocal (1/current surface)" << endl;
  cout << " -abs   : absolute value" << endl;
  cout << " -bin   : use (current surface>0) to binarise" << endl;
  cout << " -nan   : replace NaNs (improper numbers) with 0" << endl;

  cout << "\nBinary operations:" << endl;
  cout << "  (some inputs can be either a surface or a number)" << endl;
  cout << " -add   : add following input to current surface" << endl;
  cout << " -sub   : subtract following input from current surface" << endl;
  cout << " -mul   : multiply current surface by following input" << endl;
  cout << " -div   : divide current surface by following input" << endl;
  cout << " -mas   : use (following surface>0) to mask current surface" << endl;
  cout << " -thr   : use following number to threshold current surface (zero anything below the number)" << endl;

  cout << "\ne.g. surfmaths inputSurface -add inputSurface2 outputSurface" << endl;
  cout << "     surfmaths inputSurface -add 2.5 outputSurface" << endl;
  cout << "     surfmaths inputSurface -add 2.5 -mul inputSurface2 outputSurface\n" << endl;

  return 1;
}

void loadSurface(const newmesh& iSurf,newmesh& tmpSurf,const string& fname){
  tmpSurf.load_gifti(fname,false);
  if(tmpSurf.npvalues()!=iSurf.npvalues()){
    cerr<<"Error: surfaces do not have the same number of vertices"<<endl;
    exit(1);
  }
}


int check_for_output_name(int i, int argc_1)
{
  if (i>argc_1) {
    cerr << "Error: no output filename specified!" << endl;
    exit(1);
  }
  return 0;
}


int inputParser(int argc, char *argv[]){
  newmesh inputSurface;
  inputSurface.load_gifti(string(argv[1]),false);
  int i=2;
  for (i = 2; i < argc-1; i++){
    newmesh temp_surface;

    /***************************************************************/
    /********************Binary Operations**************************/
    /***************************************************************/
    if (string(argv[i])=="-mas"){
      loadSurface(inputSurface,temp_surface,string(argv[++i]));
      for(int v=0;v<temp_surface.npvalues();v++)
	inputSurface.set_pvalue(v, temp_surface.get_pvalue(v)>0?inputSurface.get_pvalue(v):0 );
    }
    /***************************************************************/
    else if (string(argv[i])=="-add"){
      i++;
      if (isNumber(string(argv[i]))){
	for(int v=0;v<inputSurface.npvalues();v++)
	  inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)+atof(argv[i]));
      }
      else{
	loadSurface(inputSurface,temp_surface,string(argv[i]));
	for(int v=0;v<temp_surface.npvalues();v++)
	  inputSurface.set_pvalue(v, temp_surface.get_pvalue(v)+inputSurface.get_pvalue(v) );
      }
    }
    /***************************************************************/
    else if (string(argv[i])=="-sub"){
      i++;
      if (isNumber(string(argv[i]))){
	for(int v=0;v<inputSurface.npvalues();v++)
	  inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)-atof(argv[i]));
      }
      else{
	loadSurface(inputSurface,temp_surface,string(argv[i]));
	for(int v=0;v<temp_surface.npvalues();v++)
	  inputSurface.set_pvalue(v, temp_surface.get_pvalue(v)-inputSurface.get_pvalue(v) );
      }
    }
    /***************************************************************/
    else if (string(argv[i])=="-mul"){
      i++;
      if (isNumber(string(argv[i]))){
	for(int v=0;v<inputSurface.npvalues();v++)
	  inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)*atof(argv[i]));
      }
      else{
	loadSurface(inputSurface,temp_surface,string(argv[i]));
	for(int v=0;v<temp_surface.npvalues();v++)
	  inputSurface.set_pvalue(v, temp_surface.get_pvalue(v)*inputSurface.get_pvalue(v) );
      }
    }
    /***************************************************************/
    else if (string(argv[i])=="-div"){
      i++;
      if (isNumber(string(argv[i]))){
	for(int v=0;v<inputSurface.npvalues();v++)
	  inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)/atof(argv[i]));
      }
      else {
	loadSurface(inputSurface,temp_surface,string(argv[i]));
	for(int v=0;v<temp_surface.npvalues();v++)
	  inputSurface.set_pvalue(v, temp_surface.get_pvalue(v)/inputSurface.get_pvalue(v) );
      }
    }
    /***************************************************************/
    /******************** Unary Operations *************************/
    /***************************************************************/
    else if (string(argv[i])=="-thr"){
      for(int v=0;v<inputSurface.npvalues();v++){
	inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)>=atof(argv[i+1])?inputSurface.get_pvalue(v):0);
      }
      i++;
    }
    /***************************************************************/
    else if (string(argv[i])=="-sqrt"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,inputSurface.get_pvalue(v)>=0?std::sqrt(inputSurface.get_pvalue(v)):0);
    }
    /***************************************************************/
    else if (string(argv[i])=="-pow"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,pow(inputSurface.get_pvalue(v),atof(argv[i+1])));
      i++;
    }
    /***************************************************************/
    else if (string(argv[i])=="-sqr"){
    for(int v=0;v<inputSurface.npvalues();v++)
      inputSurface.set_pvalue(v,inputSurface.get_pvalue(v)*inputSurface.get_pvalue(v));
    }
    /***************************************************************/
    else if (string(argv[i])=="-recip"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)!=0?1/inputSurface.get_pvalue(v):0);
    }
    /***************************************************************/
    else if (string(argv[i])=="-exp"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::exp((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-log"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,inputSurface.get_pvalue(v)>0?std::log((double)inputSurface.get_pvalue(v)):0);
    }
    /***************************************************************/
    else if (string(argv[i])=="-cos"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::cos((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-sin"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::sin((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-tan"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::tan((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-asin"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::asin((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-acos"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::acos((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-atan"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::atan((double)inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-abs"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,std::fabs(inputSurface.get_pvalue(v)));
    }
    /***************************************************************/
    else if (string(argv[i])=="-bin"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v, inputSurface.get_pvalue(v)!=0?1:0);
    }
     /******************************************************/
    else if (string(argv[i])=="-nan"){
      for(int v=0;v<inputSurface.npvalues();v++)
	inputSurface.set_pvalue(v,!isfinite(inputSurface.get_pvalue(v))?0:inputSurface.get_pvalue(v));
     }
    else{
      cerr<<"unknown option "<<string(argv[i])<<endl;
      exit(1);
    }
  }
  inputSurface.save(string(argv[argc-1]));
  return 0;
}


int main(int argc,char *argv[]){
  if (argc < 2)
    return printUsage(string(argv[0]));

  if(string(argv[1]) == "-h" || string(argv[1]) == "--help") {
    printUsage(string(argv[0]));
    exit(0);
  }
  return inputParser(argc,argv);
}
