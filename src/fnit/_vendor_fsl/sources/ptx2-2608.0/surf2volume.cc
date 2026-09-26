/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <iostream>

#include "csv.h"
#include "newimage/newimageio.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/miscprob.h"
#include "miscmaths/SpMat.h"
#include "meshclass/meshclass.h"

using namespace std;
using namespace NEWIMAGE;
using namespace MISCMATHS;
using namespace mesh;


int main(int argc, char** argv){

  if(argc<2){
    cout<<"surf2vol <surf> <refvol> <outvol> <convention>"<<endl;
    exit(1);
  }

  volume<short int> refvol;
  read_volume(refvol,argv[2]);

  CSV csv(refvol);
  csv.set_convention(argv[4]);
  csv.load_rois(argv[1]);
  csv.save_surfvol(argv[3],false);
  //  csv.save_normalsAsVol(0,string(argv[3])+"_normal");


//   // Save surface normal
//   volume4D<float> odir(refvol.xsize(),
// 		       refvol.ysize(),
// 		       refvol.zsize(),
// 		       3);
//   copybasicproperties(refvol,odir);
//   odir=0;

//   ColumnVector pos(3),dir(3);
//   for(int i=0;i<csv.get_mesh(0).nvertices();i++){
//     pos=csv.get_vertex_as_vox(0,i);
//     dir=csv.get_normal_as_vox(0,i);

//     odir((int)round((float)pos(1)),
// 	 (int)round((float)pos(2)),
// 	 (int)round((float)pos(3)),0)=dir(1);
//     odir((int)round((float)pos(1)),
// 	 (int)round((float)pos(2)),
// 	 (int)round((float)pos(3)),1)=dir(2);
//     odir((int)round((float)pos(1)),
// 	 (int)round((float)pos(2)),
// 	 (int)round((float)pos(3)),2)=dir(3);

//   }
//   odir.setDisplayMaximumMinimum(1,-1);
//   save_volume4D(odir,string(argv[3])+"_normal");

//   Matrix n(csv.get_mesh(0).nvertices(),3);
//   for(int i=0;i<csv.get_mesh(0).nvertices();i++){
//     dir=csv.get_normal(0,i);
//     n.Row(i+1)<<dir(1)<<dir(2)<<dir(3);
//   }
//   write_ascii_matrix(n,"normalsCSV.txt");

  //csv.save_roi(0,"surf.txt");

  return 0;
}
