/*  tractographyInput.cc

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <string>
#include <vector>

#include "utils/log.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimage.h"

#include "probtrackxOptions.h"
#include <CUDA/tractographyInput.h>

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace TRACT;

tractographyInput::tractographyInput(){}

// Helper function to convert between conventions
// for saving the coordinates (for instance RADIOLOGICAL/NEUROLOGICAL)
void applycoordchange(float* coords, const Matrix& old2new_mat)
{
  ColumnVector v(4);
  v << coords[0] << coords[1] << coords[2] << 1.0;
  v = old2new_mat * v;
  coords[0] = MISCMATHS::round(v(1));
  coords[1] = MISCMATHS::round(v(2));
  coords[2] = MISCMATHS::round(v(3));
}

void  tractographyInput::load_mesh(    string&              filename,
                                    vector<float>&    vertices,
                                    vector<int>&      faces,
                                    vector<int>&      locs,
                                    int&                nlocs,
                                    bool                wcoords,
                                    int                  nroi,
                                    vector<float>&    coords)
{
  int type=meshFileType(filename);
  if (type==ASCII){
    load_mesh_ascii(filename,vertices,faces,locs,nlocs,wcoords,nroi,coords);
  }else if (type==VTK){
    load_mesh_vtk(filename,vertices,faces,locs,nlocs,wcoords,nroi,coords);
  }else if (type==GIFTI){
    load_mesh_gifti(filename,vertices,faces,locs,nlocs,wcoords,nroi,coords);
  }else{
    cerr<<"Error reading file: "<<filename<<": Unknown format"<<endl;
    exit(1);
  }
}

void tractographyInput::load_mesh_ascii(    string&              filename,
                                            vector<float>&    vertices,
                                            vector<int>&      faces,
                                            vector<int>&      locs,
                                            int&                nlocs,
                                            bool                wcoords,
                                            int                  nroi,
                                            vector<float>&    coords)
{
  // load a freesurfer ascii mesh
  ifstream f(filename.c_str());
  if (f.is_open()){
    // reading the header
    string header;
    getline(f, header);
    string::size_type pos = header.find("#!ascii");
    if (pos == string::npos) {
      cerr<<"Loading ascii file: error in the header"<<endl;exit(1);
    }
    // reading the size of the mesh
    size_t NVertices, NFaces;
    f>>NVertices>>NFaces;

    size_t posV,posF,initV,posLV;
    posV=vertices.size();    // maybe there were some more vertices before
    posLV=locs.size();
    initV=posV;
    posF=faces.size();
    vertices.resize(posV+NVertices*3);
    locs.resize(posLV+NVertices);
    faces.resize(posF+NFaces*3);

    // check if some vertices should be ignored ... because the associated value
    // if all==0 then accept everything
    // if all!=0 then accept everything
    // if zeros and non-zeros then accept only non-zeros
    // store values in a temporal vector
    // and read the points
    bool allMesh=true;
    vector<float> values;
    values.resize(NVertices);
    bool zeros=false;
    bool nonzeros=false;
    for (size_t i=0; i<NVertices; i++){
      f>>vertices[posV]>>vertices[posV+1]>>vertices[posV+2]>>values[i]; // write from file to vector
      if(values[i]==0) zeros=true;
      else nonzeros=true;
      posV=posV+3;
    }
    if (zeros&&nonzeros) allMesh=false;    // then some values should be ignored

    // storing locations: use same structure for active-nonactive vertex
    size_t auxCount=posV-NVertices*3;
    size_t local_loc=0;
    for (size_t i=0; i<NVertices; i++){
      if(allMesh||values[i]!=0){
        locs[posLV]=nlocs;
        posLV++;
        nlocs++;
        if (wcoords){
          coords.push_back(MISCMATHS::round(vertices[auxCount]));
          coords.push_back(MISCMATHS::round(vertices[auxCount+1]));
          coords.push_back(MISCMATHS::round(vertices[auxCount+2]));
          coords.push_back(nroi);
          coords.push_back(local_loc);
          auxCount+=3;
        }
      }else{
        locs[posLV]=-1;
        posLV++;
        auxCount+=3;
      }
      local_loc++;
    }
    // reading the triangles
    for (size_t i=0; i<NFaces; i++){
      int p0, p1, p2;
      float val;
      f>>p0>>p1>>p2>>val;
      faces[posF]=initV+p0*3;
      faces[posF+1]=initV+p1*3;
      faces[posF+2]=initV+p2*3;
      posF=posF+3;
    }

    f.close();
  }else {cout<<"Loading ascii file: error opening file: "<<filename<<endl; exit(1);}
}

void tractographyInput::load_mesh_vtk(string&              filename,
                                      vector<float>&    vertices,
                                      vector<int>&      faces,
                                      vector<int>&      locs,
                                      int&                nlocs,
                                      bool                wcoords,
                                      int                  nroi,
                                      vector<float>&    coords)
{
  ifstream f(filename.c_str());
  if (f.is_open()){
    // reading the header
    string header;
    getline(f, header);
    string::size_type pos = header.find("# vtk DataFile Version 3.0");
    if (pos == string::npos) {
      cerr<<"Loading vtk file: error in the header"<<endl;exit(1);
    }
    getline(f,header);
    getline(f,header);
    getline(f,header);
    size_t NVertices, NFaces;
    f>>header>>NVertices>>header;
    size_t posV,posF,initV,posLV;
    posV=vertices.size();
    posLV=locs.size();
    initV=posV;
    vertices.resize(posV+NVertices*3);
    locs.resize(posLV+NVertices);

    // reading the points
    // if is not possible to define values, then all vertices are activated
    int local_loc=1;
    for (size_t i=0; i<NVertices; i++){
      f>>vertices[posV]>>vertices[posV+1]>>vertices[posV+2];
      locs[posLV]=nlocs;
      if (wcoords){
        coords.push_back(MISCMATHS::round(vertices[posV]));
        coords.push_back(MISCMATHS::round(vertices[posV+1]));
        coords.push_back(MISCMATHS::round(vertices[posV+2]));
        coords.push_back(nroi);
        coords.push_back(local_loc);
        local_loc++;
      }
      posV=posV+3;
      posLV++;
      nlocs++;
    }
    f>>header>>NFaces>>header;
    posF=faces.size();
    faces.resize(posF+NFaces*3);

    // reading the triangles
    for (size_t i=0; i<NFaces; i++){
      int p0, p1, p2;
      int j;
      f>>j>>p0>>p1>>p2;
      faces[posF]=initV+p0*3;
      faces[posF+1]=initV+p1*3;
      faces[posF+2]=initV+p2*3;
      posF=posF+3;
    }
    // reading the values (csv_mesh.cc) !!!!
    // I think is not possible to define values in VTK surfaces
    f.close();
  }else{
    cout<<"Loading vtk file: error opening file: "<<filename<<endl; exit(1);
  }
}

void tractographyInput::load_mesh_gifti(string&              filename,
                                        vector<float>&    vertices,
                                        vector<int>&      faces,
                                        vector<int>&      locs,
                                        int&                nlocs,
                                        bool                wcoords,
                                        int                  nroi,
                                        vector<float>&    coords)
{
  fslsurface_name::fslSurface<float,unsigned int> surf;
  read_surface(surf,filename);
  size_t posV,posF,initV,posLV;
  posV=vertices.size();
  posLV=locs.size();
  initV=posV;
  size_t count=0;
  for (vector< fslsurface_name::vertex<float> >::iterator  i= surf.vbegin(); i!= surf.vend();++i){
    vertices.resize(posV+3);
    vertices[posV]=i->x;
    vertices[posV+1]=i->y;
    vertices[posV+2]=i->z;
    posV=posV+3;
    count++;
  }
  posF=faces.size();
  for (vector<unsigned int>::const_iterator i=surf.const_facebegin(); i!=surf.const_faceend(); i+=3){
    faces.resize(posF+3);
    faces[posF]=initV+*i*3;
    faces[posF+1]=initV+*(i+1)*3;
    faces[posF+2]=initV+*(i+2)*3;
    posF=posF+3;
  }
  // read the values
  bool allMesh=true;
  vector<float> values;
  bool zeros=false;
  bool nonzeros=false;
  if (surf.getNumberOfScalarData()>0){
    count=0;    // reset
    for (vector<float>::const_iterator i= surf.const_scbegin(0); i!= surf.const_scend(0);++i){
      count++;
      values.resize(count);
      values[count-1]=*i;
      if (values[count-1]==0) zeros=true;
      else nonzeros=true;
    }
    if (zeros&&nonzeros) allMesh=false;
    int local_loc=1;
    size_t auxCount=posV-count*3;
    for (size_t i=0; i<count; i++){
      locs.resize(posLV+1);
      if (allMesh||values[i]!=0){
        locs[posLV]=nlocs;
        if (wcoords){
          coords.push_back(MISCMATHS::round(vertices[auxCount]));
          coords.push_back(MISCMATHS::round(vertices[auxCount+1]));
          coords.push_back(MISCMATHS::round(vertices[auxCount+2]));
          coords.push_back(nroi);
          coords.push_back(local_loc);
          local_loc++;
          auxCount+=3;
        }
        posLV++;
        nlocs++;
      }else{
        locs[posLV]=-1;
        posLV++;
        auxCount+=3;
      }
    }
  }else{    // it is possible that gifti surfaces do not have values
    int local_loc=0;
    size_t auxCount=posV-count*3;
    for (size_t i=0; i<count; i++){
      locs.resize(posLV+1);
      locs[posLV]=nlocs;
      if (wcoords){
        coords.push_back(MISCMATHS::round(vertices[auxCount]));
        coords.push_back(MISCMATHS::round(vertices[auxCount+1]));
        coords.push_back(MISCMATHS::round(vertices[auxCount+2]));
        coords.push_back(nroi);
        coords.push_back(local_loc);
        local_loc++;
        auxCount+=3;
      }
      posLV++;
      nlocs++;
    }
  }
}

void  tractographyInput::load_volume(    string&        filename,
                                        int*          Ssizes,
                                        float*        Vout,
                                        int&          nlocs,
                                        bool          reset,
                                        bool          wcoords,
                                        int            nroi,
                                        vector<float>&    coords)
{
  //reset: if true, it sets -1 voxels not present in mask,
  //reset must be false if mixed volume: stop / exclusion / targetREF, networkREF
  int local_loc=1;
  volume<float> tmpvol;
  read_volume(tmpvol,filename);
  for (int z=0;z<Ssizes[2];z++){
    for (int y=0;y<Ssizes[1];y++){
      for (int x=0;x<Ssizes[0];x++){
        if (tmpvol(x,y,z)==0){
          if (reset) Vout[z*Ssizes[0]*Ssizes[1]+y*Ssizes[0]+x]=-1;
        }else{
          size_t posVox = z*Ssizes[0]*Ssizes[1]+y*Ssizes[0]+x;
          Vout[posVox]=nlocs;
          nlocs++;
          if (wcoords){
            // write coords
            coords.push_back(x);
            coords.push_back(y);
            coords.push_back(z);
            coords.push_back(nroi);
            coords.push_back(local_loc);
            local_loc++;
          }
        }
      }
    }
  }
}

void  tractographyInput::init_surfvol(int*           Ssizes,
                                      Matrix&        mm2vox,
                                      vector<float>& vertices,
                                      int*           faces,
                                      int            sizefaces,
                                      int            initfaces,
                                      vector<int>&   voxFaces,
                                      int*           voxFacesIndex,
                                      vector<int>&   locsV)
{
  volume<int> surfvol(Ssizes[0],Ssizes[1],Ssizes[2]);
  surfvol=-1;
  vector<ColumnVector> crossed;
  vector<vector <int> > triangles;
  ColumnVector x1(4),x2(4),x3(4),xx1(3),xx2(3),xx3(3);
  int total=0;
  for (int j=0;j<sizefaces;j=j+3){
    // all vertices non-activated ?
    // int idTri=int(VoxFaces[pos+j]/3);
    if ((locsV[int(faces[j]/3)]==-1) && (locsV[int(faces[j+1]/3)]==-1) && (locsV[int(faces[j+2]/3)]==-1)){
      continue;
    }


    x1 << vertices[faces[j]] << vertices[faces[j]+1] << vertices[faces[j]+2] << 1;
    x2 << vertices[faces[j+1]] << vertices[faces[j+1]+1] << vertices[faces[j+1]+2] << 1;
    x3 << vertices[faces[j+2]] << vertices[faces[j+2]+1] << vertices[faces[j+2]+2] << 1;
    x1 = mm2vox*x1;
    x2 = mm2vox*x2;
    x3 = mm2vox*x3;

    xx1=x1.SubMatrix(1,3,1,1);
    xx2=x2.SubMatrix(1,3,1,1);
    xx3=x3.SubMatrix(1,3,1,1);

    float tri[3][3] = {
      {float(xx1(1)), float(xx1(2)), float(xx1(3))},
      {float(xx2(1)), float(xx2(2)), float(xx2(3))},
      {float(xx3(1)), float(xx3(2)), float(xx3(3))}};
    csv_tri_crossed_voxels(tri,crossed);

    for (unsigned int i=0;i<crossed.size();i++){    // loop over voxels crossed by triangle tid
      int voxX = MISCMATHS::round(crossed[i](1));
      int voxY = MISCMATHS::round(crossed[i](2));
      int voxZ = MISCMATHS::round(crossed[i](3));
      if (voxX>=0 && voxX<Ssizes[0]  && voxY>=0 && voxY<Ssizes[1] && voxZ>=0 && voxZ<Ssizes[2]){    // in the limits
        int val = surfvol(voxX,voxY,voxZ);
        if (val==-1){             // this voxel hasn't been labeled yet
          vector<int> t(1);
          t[0]=j;             // this position is relative to this portion of faces !!!!!!
          triangles.push_back(t); // add to set of triangles that cross voxels
          surfvol(voxX,voxY,voxZ)=triangles.size()-1;
          total++;
        }else{                 // voxel already labeled as "crossed"
          triangles[val].push_back(j);     // add this triangle to the set that cross this voxel
          total++;
        }
      }else{
        printf("Warning: Ignoring some vertices because they are defined outside the limits\n");
        printf("Please check that your meshspace is defined correctly\n");
      }
    }
  }
  // maybe it has already some elements (for instance, several waypoints)
  int initvoxFaces = voxFaces.size();
  voxFaces.resize(initvoxFaces+total);
  // ....like sparse matrix
  int index= 0;
  voxFacesIndex[0]=initvoxFaces;
  int count=0;
  for (int z=0;z<Ssizes[2];z++){
    for (int y=0;y<Ssizes[1];y++){
      for (int x=0;x<Ssizes[0];x++){
        int val = surfvol(x,y,z);
        if (val!=-1){
          vector<int> t;
          t.insert(t.end(),triangles[val].begin(),triangles[val].end());  // get position of the triangles (faces) crossed by this voxel
          for (unsigned int i=0;i<t.size();i++){
            voxFaces[initvoxFaces+count]=initfaces+t[i];         // store their positions (maybe position relative !!! -> add initfaces)
            count++;
          }
          voxFacesIndex[index+1]=voxFacesIndex[index]+t.size();
        }else{
          voxFacesIndex[index+1]=voxFacesIndex[index];
        }
        index++;
      }
    }
  }
}

void  tractographyInput::csv_tri_crossed_voxels(    float             tri[3][3],
                                                    vector<ColumnVector>&    crossed)
{
  int minx=(int)MISCMATHS::round(tri[0][0]);
  int miny=(int)MISCMATHS::round(tri[0][1]);
  int minz=(int)MISCMATHS::round(tri[0][2]);
  int maxx=minx,maxy=miny,maxz=minz;
  crossed.clear();
  int i=0;int tmpi;
  do{
    tmpi=(int)MISCMATHS::round(tri[i][0]);
    minx=tmpi<minx?tmpi:minx;
    maxx=tmpi>maxx?tmpi:maxx;
    tmpi=(int)MISCMATHS::round(tri[i][1]);
    miny=tmpi<miny?tmpi:miny;
    maxy=tmpi>maxy?tmpi:maxy;
    tmpi=(int)MISCMATHS::round(tri[i][2]);
    minz=tmpi<minz?tmpi:minz;
    maxz=tmpi>maxz?tmpi:maxz;
    i++;
  }while (i<=2);

  float boxcentre[3],boxhalfsize[3]={.5,.5,.5};
  ColumnVector v(3);int s=1;
  for (int x=minx-s;x<=maxx+s;x+=1){
    for (int y=miny-s;y<=maxy+s;y+=1){
      for (int z=minz-s;z<=maxz+s;z+=1){
        boxcentre[0]=(float)x;
        boxcentre[1]=(float)y;
        boxcentre[2]=(float)z;
        if (triBoxOverlap(boxcentre,boxhalfsize,tri)){
          v<<x<<y<<z;
          crossed.push_back(v);
        }
      }
    }
  }
}

// This method read the ROIs without saving IDs and reusing volume and Surface
// for exclusion & stop masks, targetREF and network REFerence
void  tractographyInput::load_rois_mixed(    string        filename,
                                            Matrix        mm2vox,
                                            float*        Sdims,
                                            int*          Ssizes,
                                            // Output
                                            MaskData&    data)
{
  data.sizesStr=new int[4];
  data.sizesStr[0]=0;
  data.sizesStr[1]=0;
  data.sizesStr[2]=0;
  data.sizesStr[3]=0;
  data.NVols=0;
  data.NSurfs=0;
  vector<int> locsVec;
  vector<float> verticesVec;
  vector<int> facesVec;
  vector<int> voxFacesVec;
  vector<float> nullV;

  size_t sizeVol = Ssizes[0]*Ssizes[1]*Ssizes[2];

  if (fsl_imageexists(filename)){
    // filename is a volume
    data.volume=new float[sizeVol];
    //memset(data.volume,-1,sizeVol*sizeof(float));
    load_volume(filename,Ssizes,data.volume,data.nlocs,true,false,0,nullV);
    data.NVols=1;
  }else if (meshExists(filename)){
    load_mesh(filename,verticesVec,facesVec,locsVec,data.nlocs,false,0,nullV);
    data.NSurfs=1;
  }else{
    // file name is ascii text file
    vector<string> fnames;
    ifstream fs(filename.c_str());
    string tmp;
    if (fs){
      fs>>tmp;
      do{
        fnames.push_back(tmp);
        fs>>tmp;
      }while (!fs.eof());
    }else{
      cerr<<filename<<" does not exist"<<endl;
      exit(0);
    }

    for (unsigned int i=0;i<fnames.size();i++){
      if (fsl_imageexists(fnames[i])){
        if (data.NVols==0){
          data.volume=new float[sizeVol];
          //memset(data.volume,-1,sizeVol*sizeof(float));
          data.NVols=1;
          load_volume(fnames[i],Ssizes,data.volume,data.nlocs,true,false,0,nullV);
        }else{
          load_volume(fnames[i],Ssizes,data.volume,data.nlocs,false,false,0,nullV);
          // do not unset voxels that are not present, maybe they are in other volume file
        }
      }else if (meshExists(fnames[i])){
        load_mesh(fnames[i],verticesVec,facesVec,locsVec,data.nlocs,false,0,nullV);
        data.NSurfs=1;
      }else{
        cerr<<"load_rois_mixed: Unknown file type: "<<fnames[i]<<endl;
        exit(1);
      }
    }
  }
  if (data.NSurfs){
    data.VoxFacesIndex=new int[sizeVol+1];
    init_surfvol(Ssizes,mm2vox,verticesVec,&facesVec[0],facesVec.size(),
                 0,voxFacesVec,data.VoxFacesIndex,locsVec);

    data.locs=new int[locsVec.size()];
    data.sizesStr[0]=locsVec.size();
    for (unsigned int i=0;i<locsVec.size();i++) data.locs[i]=locsVec[i];

    data.vertices=new float[verticesVec.size()];
    data.sizesStr[1]=verticesVec.size();
    for (unsigned int i=0;i<verticesVec.size();i++) data.vertices[i]=verticesVec[i];

    data.faces=new int[facesVec.size()];
    data.sizesStr[2]=facesVec.size();
    for (unsigned int i=0;i<facesVec.size();i++) data.faces[i]=facesVec[i];

    data.VoxFaces=new int[voxFacesVec.size()];
    data.sizesStr[3]=voxFacesVec.size();
    for (unsigned int i=0;i<voxFacesVec.size();i++) data.VoxFaces[i]=voxFacesVec[i];
  }
}

void  tractographyInput::load_rois( // Input
  string             filename,
  Matrix              mm2vox,
  float*              Sdims,    // Or Matrix2 sizes
  int*                Ssizes,
  int                  wcoords,
  volume<float>&     refVol,
  // Output
  MaskData&          data,
  Matrix&              coords)
{
  //wcoords:0 do not write, 1 write only coords, 2 write also ROI-id and position

  data.sizesStr=new int[3];
  data.sizesStr[0]=0;
  data.sizesStr[1]=0;
  data.sizesStr[2]=0;
  data.sizesStr[3]=0;
  data.NVols=0;
  data.NSurfs=0;
  vector<int> locsVec;
  vector<float> verticesVec;
  vector<int> facesVec;
  vector<int> voxFacesVec;
  vector<float> coordsV;

  size_t sizeVol = Ssizes[0]*Ssizes[1]*Ssizes[2];

  if (fsl_imageexists(filename)){
    // filename is a volume
    data.volume=new float[sizeVol];
    //memset(data.volume,-1,sizeSeed*sizeof(float));
    load_volume(filename,Ssizes,data.volume,data.nlocs,true,wcoords,0,coordsV);
    data.NVols=1;
    data.IndexRoi=new int[1];
    data.IndexRoi[0]=0;
    data.sizesStr[4]=1;
  }else if (meshExists(filename)){
    load_mesh(filename,verticesVec,facesVec,locsVec,data.nlocs,wcoords,0,coordsV);
    size_t sizeVox2Face = sizeVol+1;
    data.VoxFacesIndex=new int[sizeVox2Face];
    init_surfvol(Ssizes,mm2vox,verticesVec,&facesVec[0],facesVec.size(),
                 0,voxFacesVec,data.VoxFacesIndex,locsVec);
    data.NSurfs=1;
    data.IndexRoi=new int[1];
    data.IndexRoi[0]=0;
    data.sizesStr[4]=1;
  }else{
    // file name is ascii text file
    vector<string> fnames;
    ifstream fs(filename.c_str());
    string tmp;
    if (fs){
      fs>>tmp;
      do{
        fnames.push_back(tmp);
        if (fsl_imageexists(tmp)) data.NVols++;
        if (meshExists(tmp)) data.NSurfs++;
        fs>>tmp;
      }while (!fs.eof());
    }else{
      cerr<<filename<<" does not exist"<<endl;
      exit(0);
    }
    size_t size_AllSeedvols = data.NVols*sizeVol;
    data.volume=new float[size_AllSeedvols];
    //memset(data.volume,-1,size_AllSeedvols*sizeof(float));
    data.IndexRoi=new int[data.NVols+data.NSurfs];
    size_t size_AllVox2Face = data.NSurfs*(sizeVol+1);
    data.VoxFacesIndex=new int[size_AllVox2Face];

    int nv=0;
    int ns=0;
    int nroi=0;
    int lastfacesSize=0;

    for (unsigned int i=0;i<fnames.size();i++){
      if (fsl_imageexists(fnames[i])){
        size_t posSeedvol = static_cast<size_t>(nv)*sizeVol;
        load_volume(fnames[i],Ssizes,&data.volume[posSeedvol],data.nlocs,true,wcoords,nroi,coordsV);
        data.IndexRoi[nv]=nroi;
        nv++;
        nroi++;
      }else if (meshExists(fnames[i])){
        size_t posSeedsurf = size_t(ns)*(sizeVol+1);
        load_mesh(fnames[i],verticesVec,facesVec,locsVec,data.nlocs,wcoords,nroi,coordsV);
        init_surfvol(Ssizes,mm2vox,verticesVec,&facesVec[lastfacesSize],facesVec.size()-lastfacesSize,lastfacesSize,
                     voxFacesVec,&data.VoxFacesIndex[posSeedsurf],locsVec);
        data.IndexRoi[data.NVols+ns]=nroi;
        ns++;
        nroi++;
        lastfacesSize=facesVec.size();
      }else{
        cerr<<"load_rois: Unknown file type: "<<fnames[i]<<endl;
        exit(1);
      }
    }
    data.sizesStr[4]=nroi;
  }
  if (data.NSurfs){
    data.locs=new int[locsVec.size()];
    data.sizesStr[0]=locsVec.size();
    for (unsigned int i=0;i<locsVec.size();i++){
      data.locs[i]=locsVec[i];
    }

    data.vertices=new float[verticesVec.size()];
    data.sizesStr[1]=verticesVec.size();
    for (unsigned int i=0;i<verticesVec.size();i++) data.vertices[i]=verticesVec[i];

    data.faces=new int[facesVec.size()];
    data.sizesStr[2]=facesVec.size();
    for (unsigned int i=0;i<facesVec.size();i++) data.faces[i]=facesVec[i];

    data.VoxFaces=new int[voxFacesVec.size()];
    data.sizesStr[3]=voxFacesVec.size();
    for (unsigned int i=0;i<voxFacesVec.size();i++) data.VoxFaces[i]=voxFacesVec[i];
  }
  if (wcoords){
    if(wcoords==2){
      int nRows=coordsV.size()/5;
      coords.ReSize(nRows,5);
      int posV=0;
      for (int i=0;i<nRows;i++){
        coords.Row(i+1) << coordsV[posV]
                        << coordsV[posV+1] << coordsV[posV+2]
                        << coordsV[posV+3] << coordsV[posV+4];
        posV=posV+5;
      }
    }else{        // wcoords is 1
      int nRows=coordsV.size()/5;
      coords.ReSize(nRows,3);
      int posV=0;
      float Newcoords[3];
      for (int i=0;i<nRows;i++){
        Newcoords[0]=coordsV[posV];
        Newcoords[1]=coordsV[posV+1];
        Newcoords[2]=coordsV[posV+2];
        applycoordchange(Newcoords,refVol.niftivox2newimagevox_mat().i());
        coords.Row(i+1) << Newcoords[0]
                        << Newcoords[1] << Newcoords[2];
        posV=posV+5;
      }
    }
  }
}

void search_triangles(// Input
  int                id_vertex,
  int                id_search,
  vector<int>     facesVec,
  // Output
  int*              matrix1_locs,
  int*              matrix1_idTri,
  int*              matrix1_Ntri)
{
  int id=id_search*3;
  int num_triangles=0;
  for(unsigned int i=0;i<facesVec.size();i++){
    if(facesVec[i]==id){
      matrix1_locs[MAX_TRI_SEED*id_vertex+num_triangles]=id_vertex;
      matrix1_idTri[MAX_TRI_SEED*id_vertex+num_triangles]=i/3;
      num_triangles++;
    }
  }
  matrix1_Ntri[id_vertex]=num_triangles;
}

void  tractographyInput::load_rois_matrix1(    tractographyData&    tData,
                                            // Input
                                            string                  filename,
                                            Matrix                  mm2vox,
                                            float*                  Sdims,
                                            int*                    Ssizes,
                                            bool                    wcoords,
                                            volume<float>&         refVol,
                                            // Output
                                            MaskData&              data,
                                            Matrix&                  coords)
{
  // a maximum of 12 triangles per seed
  tData.matrix1_locs=new int[12*tData.nseeds];
  tData.matrix1_idTri=new int[12*tData.nseeds];
  tData.matrix1_Ntri=new int[tData.nseeds];

  data.sizesStr=new int[3];
  data.sizesStr[0]=0;
  data.sizesStr[1]=0;
  data.sizesStr[2]=0;
  data.sizesStr[3]=0;
  data.NVols=0;
  data.NSurfs=0;
  vector<int> locsVec;
  vector<float> verticesVec;
  vector<int> facesVec;
  vector<int> voxFacesVec;
  vector<float> coordsV;

  size_t sizeVol = Ssizes[0]*Ssizes[1]*Ssizes[2];

  if (fsl_imageexists(filename)){
    // filename is a volume
    data.volume=new float[sizeVol];
    //memset(data.volume,-1,sizeVol*sizeof(float));
    load_volume(filename,Ssizes,data.volume,data.nlocs,true,wcoords,0,coordsV);
    data.NVols=1;
    data.IndexRoi=new int[1];
    data.IndexRoi[0]=0;
    data.sizesStr[4]=1;

    for(int i=0;i<data.nlocs;i++){
      tData.matrix1_locs[MAX_TRI_SEED*i]=i;
      tData.matrix1_idTri[MAX_TRI_SEED*i]=-1;
      tData.matrix1_Ntri[i]=1;
    }
  }else if (meshExists(filename)){
    load_mesh(filename,verticesVec,facesVec,locsVec,data.nlocs,wcoords,0,coordsV);
    data.VoxFacesIndex=new int[sizeVol+1];
    init_surfvol(Ssizes,mm2vox,verticesVec,&facesVec[0],facesVec.size(),
                 0,voxFacesVec,data.VoxFacesIndex,locsVec);
    data.NSurfs=1;
    data.IndexRoi=new int[1];
    data.IndexRoi[0]=0;
    data.sizesStr[4]=1;
    for(int i=0;i<data.nlocs;i++){
      search_triangles(i,i,facesVec,tData.matrix1_locs,tData.matrix1_idTri,tData.matrix1_Ntri);
    }
  }else{
    // file name is ascii text file
    vector<string> fnames;
    ifstream fs(filename.c_str());
    string tmp;
    if (fs){
      fs>>tmp;
      do{
        fnames.push_back(tmp);
        if (fsl_imageexists(tmp)) data.NVols++;
        if (meshExists(tmp)) data.NSurfs++;
        fs>>tmp;
      }while (!fs.eof());
    }else{
      cerr<<filename<<" does not exist"<<endl;
      exit(0);
    }
    data.volume=new float[data.NVols*sizeVol];
    //memset(data.volume,-1,data.NVols*sizeVol*sizeof(float));
    data.IndexRoi=new int[data.NVols+data.NSurfs];
    data.VoxFacesIndex=new int[data.NSurfs*(sizeVol+1)];

    int nv=0;
    int ns=0;
    int nroi=0;
    int lastfacesSize=0;

    int last_loc=0;
    int locs_from_volume=0;
    for (unsigned int file=0;file<fnames.size();file++){
      if (fsl_imageexists(fnames[file])){
        load_volume(fnames[file],Ssizes,&data.volume[nv*sizeVol],
                    data.nlocs, true,wcoords,nroi,coordsV);
        data.IndexRoi[nv]=nroi;
        nv++;
        nroi++;

        for(int i=last_loc;i<data.nlocs;i++){
          tData.matrix1_locs[MAX_TRI_SEED*i]=i;
          tData.matrix1_idTri[MAX_TRI_SEED*i]=-1;
          tData.matrix1_Ntri[i]=1;
        }
        locs_from_volume+=data.nlocs-last_loc;
      }else if (meshExists(fnames[file])){
        load_mesh(fnames[file],verticesVec,facesVec,locsVec,data.nlocs,wcoords,nroi,coordsV);
        init_surfvol(Ssizes,mm2vox,verticesVec,
                     &facesVec[lastfacesSize],facesVec.size()-lastfacesSize,lastfacesSize,
                     voxFacesVec,&data.VoxFacesIndex[ns*(sizeVol+1)],locsVec);
        data.IndexRoi[data.NVols+ns]=nroi;
        ns++;
        nroi++;
        lastfacesSize=facesVec.size();

        for(int i=last_loc;i<data.nlocs;i++){
          search_triangles(i,i-locs_from_volume,facesVec,tData.matrix1_locs,tData.matrix1_idTri,tData.matrix1_Ntri);
        }
      }else{
        cerr<<"load_rois: Unknown file type: "<<fnames[file]<<endl;
        exit(1);
      }
      last_loc=data.nlocs;
    }
    data.sizesStr[4]=nroi;
  }
  if (data.NSurfs){
    data.locs=new int[locsVec.size()];
    data.sizesStr[0]=locsVec.size();
    for (unsigned int i=0;i<locsVec.size();i++){
      data.locs[i]=locsVec[i];
    }

    data.vertices=new float[verticesVec.size()];
    data.sizesStr[1]=verticesVec.size();
    for (unsigned int i=0;i<verticesVec.size();i++) data.vertices[i]=verticesVec[i];

    data.faces=new int[facesVec.size()];
    data.sizesStr[2]=facesVec.size();
    for (unsigned int i=0;i<facesVec.size();i++) data.faces[i]=facesVec[i];

    data.VoxFaces=new int[voxFacesVec.size()];
    data.sizesStr[3]=voxFacesVec.size();
    for (unsigned int i=0;i<voxFacesVec.size();i++) data.VoxFaces[i]=voxFacesVec[i];
  }
  if (wcoords){
    int nRows=coordsV.size()/5;
    coords.ReSize(nRows,5);
    int posV=0;
    float Newcoords[3];
    for (int i=0;i<nRows;i++){
      Newcoords[0]=coordsV[posV];
      Newcoords[1]=coordsV[posV+1];
      Newcoords[2]=coordsV[posV+2];
      applycoordchange(Newcoords,refVol.niftivox2newimagevox_mat().i());
      coords.Row(i+1) << Newcoords[0]
                      << Newcoords[1] << Newcoords[2]
                      << coordsV[posV+3] << coordsV[posV+4];
      posV=posV+5;
    }
  }
}

size_t tractographyInput::load_seeds_rois(    tractographyData&    tData,
                                            string                  seeds_filename,
                                            string                  ref_filename,
                                            float*                  Sdims,
                                            int*                    Ssizes,
                                            int                      convention,
                                            float*&                  seeds,
                                            int*&                    seeds_ROI,
                                            Matrix&                  mm2vox,
                                            float*                  vox2mm,
                                            volume<float>*&        m_prob,
                                            bool                    initialize_m_prob,
                                            volume<float>*&        m_prob2,
                                            bool                    initialize_m_prob2,
                                            volume4D<float>*&    m_localdir,
                                            volume<float>&        refVol) // reference
{
  Log& logger = LogSingleton::getInstance();
  probtrackxOptions& opts=probtrackxOptions::getInstance();
  vector<float> nullV;
  size_t nseeds=0;

  if (fsl_imageexists(seeds_filename)){
    // a volume file
    if(opts.network.value()){
      cerr<<"Seed is a volume file - please turn off --network option or change the seed to a list of files"<<endl;
      exit(1);
    }
    volume<float> seedsVol;
    read_volume(seedsVol,seeds_filename);
    refVol=seedsVol;
    Sdims[0]=seedsVol.xdim();
    Sdims[1]=seedsVol.ydim();
    Sdims[2]=seedsVol.zdim();
    Ssizes[0]=seedsVol.xsize();
    Ssizes[1]=seedsVol.ysize();
    Ssizes[2]=seedsVol.zsize();
    size_t sizeVol = Ssizes[0]*Ssizes[1]*Ssizes[2];
    set_vox2mm(convention,Sdims,Ssizes,seedsVol,mm2vox,vox2mm);

    seeds=new float[3*sizeVol];
    for (int z=0;z<Ssizes[2];z++){
      for (int y=0;y<Ssizes[1];y++){
        for (int x=0;x<Ssizes[0];x++){
          if (seedsVol(x,y,z)){
            seeds[nseeds*3]=x;
            seeds[nseeds*3+1]=y;
            seeds[nseeds*3+2]=z;
            nseeds++;
          }
        }
      }
    }
    if (initialize_m_prob){
      m_prob->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
      copybasicproperties(seedsVol,*m_prob);
      *m_prob=0;
    }
    if (initialize_m_prob2){
      m_prob2->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
      copybasicproperties(seedsVol,*m_prob2);
      *m_prob2=0;
    }
    if(opts.opathdir.value()){     // OPATHDIR
      m_localdir->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2],6);
      copybasicproperties(seedsVol,*m_localdir);
      *m_localdir=0;
    }
    return nseeds;
  }else if (meshExists(seeds_filename)){
    // a surface file
    if(opts.network.value()){
      cerr<<"Seed is a surface file - please turn off --network option or change the seed to a list of files"<<endl;
      exit(1);
    }
    if (ref_filename==""){
      cerr<<"Error: need to set a reference volume when defining a surface-based seed mask"<<endl;
      exit(1);
    }else{
      if (fsl_imageexists(ref_filename)){
        read_volume(refVol,ref_filename);
        Sdims[0]=refVol.xdim();
        Sdims[1]=refVol.ydim();
        Sdims[2]=refVol.zdim();
        Ssizes[0]=refVol.xsize();
        Ssizes[1]=refVol.ysize();
        Ssizes[2]=refVol.zsize();
        set_vox2mm(convention,Sdims,Ssizes,refVol,mm2vox,vox2mm);
        if (initialize_m_prob){
          m_prob->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
          copybasicproperties(refVol,*m_prob);
          *m_prob=0;
        }
        if (initialize_m_prob2){
          m_prob2->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
          copybasicproperties(refVol,*m_prob2);
          *m_prob2=0;
        }
        if(opts.opathdir.value()){     // OPATHDIR
          m_localdir->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2],6);
          copybasicproperties(refVol,*m_localdir);
          *m_localdir=0;
        }
      }else{
        cerr<<"Reference volume "<<ref_filename<<" does not exist"<<endl;
        exit(1);
      }
      int nlocs=0;
      vector<int> locs;
      vector<float> vertices;
      vector<int> faces;
      load_mesh(seeds_filename,vertices,faces,locs,nlocs,false,0,nullV);
      seeds=new float[vertices.size()*3];
      int loc=0;
      float c1,c2,c3;
      for (unsigned int vertex=0;vertex<vertices.size();vertex+=3){
        if (locs[loc]!=-1){    // if activated
          // get voxel that correspond to the vertex
          c1=vertices[vertex];
          c2=vertices[vertex+1];
          c3=vertices[vertex+2];
          seeds[nseeds*3]=mm2vox(1,1)*c1+mm2vox(1,2)*c2+mm2vox(1,3)*c3+mm2vox(1,4);
          seeds[nseeds*3+1]=mm2vox(2,1)*c1+mm2vox(2,2)*c2+mm2vox(2,3)*c3+mm2vox(2,4);
          seeds[nseeds*3+2]=mm2vox(3,1)*c1+mm2vox(3,2)*c2+mm2vox(3,3)*c3+mm2vox(3,4);
          nseeds++;
        }
        if(opts.s2tout.value()){
          tData.seeds_vertices.push_back(vertices[vertex]);
          tData.seeds_vertices.push_back(vertices[vertex+1]);
          tData.seeds_vertices.push_back(vertices[vertex+2]);
          if (locs[loc]!=-1){
            tData.seeds_act.push_back(1);
          }else{
            tData.seeds_act.push_back(0);
          }
        }
        loc++;
      }
      if(opts.s2tout.value()){
        tData.seeds_mesh_info.push_back(loc);
        int ntri=0;
        for (unsigned int tri=0;tri<faces.size();tri+=3){
          tData.seeds_faces.push_back(faces[tri]/3);
          tData.seeds_faces.push_back(faces[tri+1]/3);
          tData.seeds_faces.push_back(faces[tri+2]/3);
          ntri++;
        }
        tData.seeds_mesh_info.push_back(ntri);
      }
    }
    return nseeds;
  }else{
    // ascii text file with a list of files
    // get dims from the first volume found (if any)
    vector<float> seedsV; //temporal vector, I do not know the size in advance (several seed files), maybe I need to resize it

    bool found_vol=false;    //Seed space dims needed
    vector<string> fnames;
    ifstream fs(seeds_filename.c_str()); string tmp;
    if (fs){
      fs>>tmp;
      do{
        fnames.push_back(tmp);
        fs>>tmp;
      }while (!fs.eof());
    }else{
      cerr<<"Seed file "<<seeds_filename<<" does not exist"<<endl;
      exit(1);
    }
    // read all volumes first to search a reference volume
    for (unsigned int i=0;i<fnames.size();i++){
      if (fsl_imageexists(fnames[i])){
        volume<float> seedsVol;
        read_volume(seedsVol,fnames[i]);
        if (!found_vol){
          refVol=seedsVol;
          Sdims[0]=seedsVol.xdim();
          Sdims[1]=seedsVol.ydim();
          Sdims[2]=seedsVol.zdim();
          Ssizes[0]=seedsVol.xsize();
          Ssizes[1]=seedsVol.ysize();
          Ssizes[2]=seedsVol.zsize();
          set_vox2mm(convention,Sdims,Ssizes,seedsVol,mm2vox,vox2mm);
          if (initialize_m_prob){
            m_prob->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
            copybasicproperties(seedsVol,*m_prob);
            *m_prob=0;
          }
          if (initialize_m_prob2){
            m_prob2->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
            copybasicproperties(seedsVol,*m_prob2);
            *m_prob2=0;
          }
          if(opts.opathdir.value()){     // OPATHDIR
            m_localdir->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2],6);
            copybasicproperties(seedsVol,*m_localdir);
            *m_localdir=0;
          }
        }else{
          if (Sdims[0]!=seedsVol.xdim()||Sdims[1]!=seedsVol.ydim()||Sdims[2]!=seedsVol.zdim()||
              Ssizes[0]!=seedsVol.xsize()||Ssizes[1]!=seedsVol.ysize()||Ssizes[2]!=seedsVol.zsize()){
            cerr<<"Seed volumes must have same dimensions"<<endl;
            exit(1);
          }
        }
        found_vol=true;
      }
    }
    if (!found_vol){
      if (ref_filename==""){
        cerr<<"Error: need to set a reference volume when defining a surface-based seed mask"<<endl;
        exit(1);
      }else if (fsl_imageexists(ref_filename)){
        read_volume(refVol,ref_filename);
        Sdims[0]=refVol.xdim();
        Sdims[1]=refVol.ydim();
        Sdims[2]=refVol.zdim();
        Ssizes[0]=refVol.xsize();
        Ssizes[1]=refVol.ysize();
        Ssizes[2]=refVol.zsize();
        set_vox2mm(convention,Sdims,Ssizes,refVol,mm2vox,vox2mm);
        if (initialize_m_prob){
          m_prob->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
          copybasicproperties(refVol,*m_prob);
          *m_prob=0;
        }
        if (initialize_m_prob2){
          m_prob2->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2]);
          copybasicproperties(refVol,*m_prob2);
          *m_prob2=0;
        }
        if(opts.opathdir.value()){     // OPATHDIR
          m_localdir->reinitialize(Ssizes[0],Ssizes[1],Ssizes[2],6);
          copybasicproperties(refVol,*m_localdir);
          *m_localdir=0;
        }
      }else{
        cerr<<"Reference volume "<<ref_filename<<" does not exist"<<endl;
        exit(1);
      }
    }

    if(fnames.size()==1&&opts.network.value()){
      cerr<<"Seed is a single ROI - please turn off --network option or change the seed to a list of >1 files"<<endl;
      exit(1);
    }

    // for network mode, to know the ROI id of each seed
    size_t* sizes_rois= new size_t[fnames.size()];
    int last_num_seeds=0;

    // read all volumes & surfaces
    for (unsigned int i=0;i<fnames.size();i++){
      if (meshExists(fnames[i])){
        int nlocs=0;
        vector<int> locs;
        vector<float> vertices;
        vector<int> faces;
        load_mesh(fnames[i],vertices,faces,locs,nlocs,false,0,nullV);
        seedsV.resize(seedsV.size()+vertices.size()*3);
        int loc=0;
        float c1,c2,c3;
        float s1,s2,s3;
        for (unsigned int vertex=0;vertex<vertices.size();vertex+=3){
          if (locs[loc]!=-1){    // if activated
            // get voxel that correspond to the vertex
            c1=vertices[vertex];
            c2=vertices[vertex+1];
            c3=vertices[vertex+2];
            s1=mm2vox(1,1)*c1+mm2vox(1,2)*c2+mm2vox(1,3)*c3+mm2vox(1,4);
            s2=mm2vox(2,1)*c1+mm2vox(2,2)*c2+mm2vox(2,3)*c3+mm2vox(2,4);
            s3=mm2vox(3,1)*c1+mm2vox(3,2)*c2+mm2vox(3,3)*c3+mm2vox(3,4);
            if (s1>=0 && s1<Ssizes[0] && s2>=0 && s2<Ssizes[1] && s3>=0 && s3<Ssizes[2]){
              seedsV[nseeds*3]=s1;
              seedsV[nseeds*3+1]=s2;
              seedsV[nseeds*3+2]=s3;
              nseeds++;
            }else{
              printf("Warning: Ignoring some seeds because they are defined outside the limits\n");
              printf("Please check that your meshspace is defined correctly\n");
            }
          }
          if(opts.s2tout.value()){
            tData.seeds_vertices.push_back(vertices[vertex]);
            tData.seeds_vertices.push_back(vertices[vertex+1]);
            tData.seeds_vertices.push_back(vertices[vertex+2]);
            if (locs[loc]!=-1){
              tData.seeds_act.push_back(1);
            }else{
              tData.seeds_act.push_back(0);
            }
          }
          loc++;
        }
        sizes_rois[i]=(nseeds-last_num_seeds);
        last_num_seeds=nseeds;
        if(opts.s2tout.value()){
          tData.seeds_mesh_info.push_back(loc);
          int ntri=0;
          for (unsigned int tri=0;tri<faces.size();tri+=3){
            tData.seeds_faces.push_back(faces[tri]/3);
            tData.seeds_faces.push_back(faces[tri+1]/3);
            tData.seeds_faces.push_back(faces[tri+2]/3);
            ntri++;
          }
          tData.seeds_mesh_info.push_back(ntri);
        }
      }else if (fsl_imageexists(fnames[i])){
        volume<float> seedsVol;
        read_volume(seedsVol,fnames[i]);
        size_t sizeVol = Ssizes[0]*Ssizes[1]*Ssizes[2];
        seedsV.resize(seedsV.size()+3*sizeVol);    //max
        for (int z=0;z<Ssizes[2];z++){
          for (int y=0;y<Ssizes[1];y++){
            for (int x=0;x<Ssizes[0];x++){
              if (seedsVol(x,y,z)){
                seedsV[nseeds*3]=x;
                seedsV[nseeds*3+1]=y;
                seedsV[nseeds*3+2]=z;
                nseeds++;
              }
            }
          }
        }
        sizes_rois[i]=(nseeds-last_num_seeds);
        last_num_seeds=nseeds;
      }else{
        cerr<<"Unknown file type: "<<fnames[i]<<endl;
        exit(1);
      }
    }
    seeds=new float[nseeds*3];
    memcpy(seeds,&seedsV[0],nseeds*3*sizeof(float));
    if(opts.network.value()){
      seeds_ROI=new int[nseeds];
      size_t pos=0;
      for (unsigned int i=0;i<fnames.size();i++){
        for (size_t j=0;j<sizes_rois[i];j++){
          seeds_ROI[pos]=i;
          pos++;
        }
      }
    }
    // write in a file the number of seeds of each ROI, for statistics
    Matrix M_sizes_rois(fnames.size(),1);
    for (unsigned int i=0;i<fnames.size();i++){
      M_sizes_rois(i+1,1)=sizes_rois[i];
    }
    write_ascii_matrix(M_sizes_rois,logger.appendDir("NumSeeds_of_ROIs"));
    ////

    return nseeds;
  }
}

void  tractographyInput::set_vox2mm(int                convention,
                                    float*            Sdims,
                                    int*              Ssizes,
                                    volume<float>    vol,
                                    Matrix&            mm2vox,
                                    float*            vox2mm)
{
  // VOX2MM
  Matrix Mvox2mm(4,4);
  if (convention==0){
    // caret
    Mvox2mm = vol.newimagevox2mm_mat();
    mm2vox= Mvox2mm.i();
  }else if (convention==1){
    // freesurfer
    Matrix mat(4,4);
    mat << -1/Sdims[0] << 0 << 0 << Ssizes[0]/2
        << 0 << 0 << -1/Sdims[1] << Ssizes[2]/2
        << 0 << 1/Sdims[2] << 0 << Ssizes[1]/2
        << 0 << 0 << 0 << 1;
    mm2vox=mat;
    Mvox2mm=mm2vox.i();
  }else if (convention==2){
    // first
    Mvox2mm << Sdims[0] << 0 << 0 << 0
            << 0 << Sdims[1] << 0 << 0
            << 0 << 0 << Sdims[2] <<0
            << 0 << 0 << 0 << 1;
    Matrix Q = vol.newimagevox2mm_mat();
    if (Q.Determinant()>0){
      Matrix mat(4,4);
      // mat<<-1<<0,0,Ssizex-1
      // <<0<<1<<0<<0
      // <<0<<0<<1<<0
      // <<0<<0<<0<<1;
      mat=IdentityMatrix(4);
      mm2vox=(mat*Mvox2mm).i();
    }else{
      mm2vox=Mvox2mm.i();
    }
  }else if (convention==2){
    // vox
    Mvox2mm=IdentityMatrix(4);
    mm2vox=IdentityMatrix(4);
  }
  vox2mm[0]=Mvox2mm(1,1); vox2mm[1]=Mvox2mm(1,2); vox2mm[2]=Mvox2mm(1,3); vox2mm[3]=Mvox2mm(1,4);
  vox2mm[4]=Mvox2mm(2,1); vox2mm[5]=Mvox2mm(2,2); vox2mm[6]=Mvox2mm(2,3); vox2mm[7]=Mvox2mm(2,4);
  vox2mm[8]=Mvox2mm(3,1); vox2mm[9]=Mvox2mm(3,2); vox2mm[10]=Mvox2mm(3,3); vox2mm[11]=Mvox2mm(3,4);
  vox2mm[12]=Mvox2mm(4,1); vox2mm[13]=Mvox2mm(4,2); vox2mm[14]=Mvox2mm(4,3); vox2mm[15]=Mvox2mm(4,4);
}

void tractographyInput::load_tractographyData(tractographyData&    tData,
                                              volume<float>*&        m_prob,
                                              volume<float>*&        m_prob2,
                                              float**&              ConNet,
                                              float**&              ConNetb,
                                              int&                    nRowsNet,
                                              int&                    nColsNet,
                                              float**&              ConMat1,
                                              float**&              ConMat1b,
                                              int&                    nRowsMat1,
                                              int&                    nColsMat1,
                                              float**&              ConMat3,
                                              float**&              ConMat3b,
                                              int&                    nRowsMat3,
                                              int&                    nColsMat3,
                                              float*&                  m_s2targets,
                                              float*&                  m_s2targetsb,
                                              volume4D<float>*&    m_localdir)
{
  printf("Loading tractography data\n");
  probtrackxOptions& opts=probtrackxOptions::getInstance();
  Log& logger = LogSingleton::getInstance();

  tData.nparticles=opts.nparticles.value();
  tData.nsteps=opts.nsteps.value();
  tData.steplength=opts.steplength.value();
  tData.distthresh=opts.distthresh.value();
  tData.curv_thr=opts.c_thr.value();
  tData.fibthresh=opts.fibthresh.value();
  tData.sampvox=opts.sampvox.value();
  if (opts.fibst.set()) tData.fibst=opts.fibst.value()-1;  //0,1,2 ....
  else tData.fibst=-1;   //not set
  tData.usef=opts.usef.value();

  // set convention
  int convention=0;
  // 0 -> caret . default
  // 1 -> freesurfer
  // 2 -> first
  // 3 -> vox

  if (opts.meshspace.value()=="caret") convention=0;
  else if (opts.meshspace.value()=="freesurfer") convention=1;
  else if (opts.meshspace.value()=="first") convention=2;
  else if (opts.meshspace.value()=="vox") convention=3;
  else{
    cerr<<"Error: Unknown convention "<<opts.meshspace.value()<<endl;
    exit(1);
  }

  /////////////////////////////////////////////
  ////////        LOAD SEEDS &       //////////
  //////// READ DIMS-SIZE SEED-SPACE //////////
  ////////          &VOX2MM          //////////
  ////////    &initialise m_prob     //////////
  /////////////////////////////////////////////
  tData.Sdims=new float[3];
  tData.Ssizes=new int[3];
  Matrix mm2vox;
  tData.vox2mm=new float[16];

  volume<float> refVol;

  tData.nseeds=load_seeds_rois(tData,opts.seedfile.value(),opts.seedref.value(),
                               tData.Sdims,tData.Ssizes,convention,tData.seeds,tData.seeds_ROI,
                               mm2vox,tData.vox2mm,m_prob,opts.simpleout.value(),
                               m_prob2,(opts.simpleout.value()&&opts.omeanpathlength.value()),
                               m_localdir,refVol);
  printf("Number of Seeds: %i\n",tData.nseeds);

  /////////////////////////
  //////// LOAD MASK //////
  /////////////////////////
  volume<float> mask3D;
  read_volume(mask3D,opts.maskfile.value());
  tData.mask= new float[mask3D.xsize()*mask3D.ysize()*mask3D.zsize()];
  for(int z=0;z<mask3D.zsize();z++){
    for(int y=0;y<mask3D.ysize();y++){
      for(int x=0;x<mask3D.xsize();x++){
        tData.mask[z*mask3D.xsize()*mask3D.ysize()+y*mask3D.xsize()+x]=mask3D(x,y,z);
      }
    }
  }
  tData.Ddims = new float[3];
  tData.Ddims[0]=mask3D.xdim();
  tData.Ddims[1]=mask3D.ydim();
  tData.Ddims[2]=mask3D.zdim();
  tData.Dsizes = new int[3];
  tData.Dsizes[0]=mask3D.xsize();
  tData.Dsizes[1]=mask3D.ysize();
  tData.Dsizes[2]=mask3D.zsize();

  /////////////////////////
  ////// LOAD SAMPLES /////
  /////////////////////////
  string basename=opts.basename.value();
  volume4D<float> tmpvol;
  Matrix tmpmat;
  tData.nvoxels=0;
  tData.nsamples=0;
  tData.nfibres=0;
  if(fsl_imageexists(basename+"_thsamples")){
    // only 1 fibre
    read_volume4D(tmpvol,basename+"_thsamples");
    tmpmat=tmpvol.matrix(mask3D);
    tData.nvoxels=tmpmat.Ncols();
    tData.nsamples=tmpmat.Nrows();
    tData.nfibres=1;
    size_t samplesSize = tData.nfibres*tData.nsamples*tData.nvoxels;
    tData.thsamples=new float[samplesSize];
    tData.phsamples=new float[samplesSize];
    tData.fsamples=new float[samplesSize];
    for(int s=0;s<tData.nsamples;s++){
      for(int v=0;v<tData.nvoxels;v++){
        tData.thsamples[s*tData.nvoxels  +v]=tmpmat(s+1,v+1);
      }
    }
    read_volume4D(tmpvol,basename+"_phsamples");
    tmpmat=tmpvol.matrix(mask3D);
    for(int s=0;s<tData.nsamples;s++){
      for(int v=0;v<tData.nvoxels;v++){
        tData.phsamples[s*tData.nvoxels+v]=tmpmat(s+1,v+1);
      }
    }
    read_volume4D(tmpvol,basename+"_fsamples");
    tmpmat=tmpvol.matrix(mask3D);
    for(int s=0;s<tData.nsamples;s++){
      for(int v=0;v<tData.nvoxels;v++){
        tData.fsamples[s*tData.nvoxels+v]=tmpmat(s+1,v+1);
      }
    }
  }else{
    // several fibres
    bool fib_existed=true;
    while(fib_existed){
      if(fsl_imageexists(basename+"_th"+num2str(tData.nfibres+1)+"samples")) tData.nfibres++;
      else fib_existed=false;
    }
    read_volume4D(tmpvol,basename+"_th1samples");
    tmpmat=tmpvol.matrix(mask3D);
    tData.nvoxels=tmpmat.Ncols();
    tData.nsamples=tmpmat.Nrows();
    size_t samplesSize = tData.nfibres*tData.nsamples*tData.nvoxels;
    tData.thsamples=new float[samplesSize];
    tData.phsamples=new float[samplesSize];
    tData.fsamples=new float[samplesSize];
    for(int f=0;f<tData.nfibres;f++){
      if(f>0){
        read_volume4D(tmpvol,basename+"_th"+num2str(f+1)+"samples");
        tmpmat=tmpvol.matrix(mask3D);
      }
      for(int s=0;s<tData.nsamples;s++){
        for(int v=0;v<tData.nvoxels;v++){
          tData.thsamples[f*tData.nsamples*tData.nvoxels+s*tData.nvoxels+v]=tmpmat(s+1,v+1);
        }
      }
      read_volume4D(tmpvol,basename+"_ph"+num2str(f+1)+"samples");
      tmpmat=tmpvol.matrix(mask3D);
      for(int s=0;s<tData.nsamples;s++){
        for(int v=0;v<tData.nvoxels;v++){
          tData.phsamples[f*tData.nsamples*tData.nvoxels+s*tData.nvoxels+v]=tmpmat(s+1,v+1);
        }
      }
      read_volume4D(tmpvol,basename+"_f"+num2str(f+1)+"samples");
      tmpmat=tmpvol.matrix(mask3D);
      for(int s=0;s<tData.nsamples;s++){
        for(int v=0;v<tData.nvoxels;v++){
          tData.fsamples[f*tData.nsamples*tData.nvoxels+s*tData.nvoxels+v]=tmpmat(s+1,v+1);
        }
      }
    }
  }

  ///////////////////////////
  // CALCULATE LUT_VOL2MAT //
  ///////////////////////////
  volume<int> tmpvol2 = tmpvol.vol2matrixkey(mask3D);
  tData.lut_vol2mat = new int[mask3D.xsize()*mask3D.ysize()*mask3D.zsize()]; //coordenates to number of voxel
  for(int z=0;z<mask3D.zsize();z++){
    for(int y=0;y<mask3D.ysize();y++){
      for(int x=0;x<mask3D.xsize();x++){
        tData.lut_vol2mat[z*mask3D.ysize()*mask3D.xsize()+y*mask3D.xsize()+x]=tmpvol2(x,y,z);
      }
    }
  }

  //////////////////////////
  ////// Seeds_to_DTI //////
  ////// DTI_to_Seeds //////
  //////////////////////////
  float* Seeds_to_DTI_read;
  float* DTI_to_Seeds_read;
  Seeds_to_DTI_read=new float[16];
  DTI_to_Seeds_read=new float[16];
  Matrix MSeeds_to_DTI=IdentityMatrix(4);
  Matrix MDTI_to_Seeds=IdentityMatrix(4);
  if(opts.seeds_to_dti.value()!=""){
    if(!fsl_imageexists(opts.seeds_to_dti.value())){ //presumably ascii file provided
      MSeeds_to_DTI=read_ascii_matrix(opts.seeds_to_dti.value());
      MDTI_to_Seeds=MSeeds_to_DTI.i();
    }else{
      FnirtFileReader ffr(opts.seeds_to_dti.value());
      //MSeeds_to_DTI_warp=ffr.FieldAsNewimageVolume4D(true);
      if(opts.dti_to_seeds.value()==""){
        cerr << "Error: Seeds transform needed (DTI to Seeds space)" << endl;
        exit(1);
      }
      FnirtFileReader iffr(opts.dti_to_seeds.value());
      //MDTI_to_Seeds_warp=iffr.FieldAsNewimaheVolume4D(true);
    }
  }
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      Seeds_to_DTI_read[i*4+j]=MSeeds_to_DTI(i+1,j+1);
      DTI_to_Seeds_read[i*4+j]=MDTI_to_Seeds(i+1,j+1);
    }
  }
  tData.Seeds_to_DTI=new float[12];
  tData.DTI_to_Seeds=new float[12];

  float C = Seeds_to_DTI_read[12]+Seeds_to_DTI_read[13]+Seeds_to_DTI_read[14]+Seeds_to_DTI_read[15];
  float* div = new float[3];
  div[0]=C*tData.Ddims[0];
  div[1]=C*tData.Ddims[1];
  div[2]=C*tData.Ddims[2];
  tData.Seeds_to_DTI[0]=Seeds_to_DTI_read[0]*tData.Sdims[0]/div[0];
  tData.Seeds_to_DTI[1]=Seeds_to_DTI_read[1]*tData.Sdims[1]/div[0];
  tData.Seeds_to_DTI[2]=Seeds_to_DTI_read[2]*tData.Sdims[2]/div[0];
  tData.Seeds_to_DTI[3]=Seeds_to_DTI_read[3]/div[0];
  tData.Seeds_to_DTI[4]=Seeds_to_DTI_read[4]*tData.Sdims[0]/div[1];
  tData.Seeds_to_DTI[5]=Seeds_to_DTI_read[5]*tData.Sdims[1]/div[1];
  tData.Seeds_to_DTI[6]=Seeds_to_DTI_read[6]*tData.Sdims[2]/div[1];
  tData.Seeds_to_DTI[7]=Seeds_to_DTI_read[7]/div[1];
  tData.Seeds_to_DTI[8]=Seeds_to_DTI_read[8]*tData.Sdims[0]/div[2];
  tData.Seeds_to_DTI[9]=Seeds_to_DTI_read[9]*tData.Sdims[1]/div[2];
  tData.Seeds_to_DTI[10]=Seeds_to_DTI_read[10]*tData.Sdims[2]/div[2];
  tData.Seeds_to_DTI[11]=Seeds_to_DTI_read[11]/div[2];

  C = DTI_to_Seeds_read[12]+DTI_to_Seeds_read[13]+DTI_to_Seeds_read[14]+DTI_to_Seeds_read[15];
  div[0]=C*tData.Sdims[0];
  div[1]=C*tData.Sdims[1];
  div[2]=C*tData.Sdims[2];
  tData.DTI_to_Seeds[0]=DTI_to_Seeds_read[0]*tData.Ddims[0]/div[0];
  tData.DTI_to_Seeds[1]=DTI_to_Seeds_read[1]*tData.Ddims[1]/div[0];
  tData.DTI_to_Seeds[2]=DTI_to_Seeds_read[2]*tData.Ddims[2]/div[0];
  tData.DTI_to_Seeds[3]=DTI_to_Seeds_read[3]/div[0];
  tData.DTI_to_Seeds[4]=DTI_to_Seeds_read[4]*tData.Ddims[0]/div[1];
  tData.DTI_to_Seeds[5]=DTI_to_Seeds_read[5]*tData.Ddims[1]/div[1];
  tData.DTI_to_Seeds[6]=DTI_to_Seeds_read[6]*tData.Ddims[2]/div[1];
  tData.DTI_to_Seeds[7]=DTI_to_Seeds_read[7]/div[1];
  tData.DTI_to_Seeds[8]=DTI_to_Seeds_read[8]*tData.Ddims[0]/div[2];
  tData.DTI_to_Seeds[9]=DTI_to_Seeds_read[9]*tData.Ddims[1]/div[2];
  tData.DTI_to_Seeds[10]=DTI_to_Seeds_read[10]*tData.Ddims[2]/div[2];
  tData.DTI_to_Seeds[11]=DTI_to_Seeds_read[11]/div[2];

  ///////////////////////////////
  ////// NON LINEAR warps ///////
  ///////////////////////////////
  tData.IsNonlinXfm = false;
  tData.Warp_S2D_sizes=new int[3];
  tData.Warp_D2S_sizes=new int[3];
  tData.Warp_S2D_sizes[0]=0;
  tData.Warp_S2D_sizes[1]=0;
  tData.Warp_S2D_sizes[2]=0;
  tData.Warp_D2S_sizes[0]=0;
  tData.Warp_D2S_sizes[1]=0;
  tData.Warp_D2S_sizes[2]=0;
  tData.SsamplingI=new float[3];
  tData.DsamplingI=new float[3];
  tData.Wsampling_S2D_I=new float[3];
  tData.Wsampling_D2S_I=new float[3];
  if(opts.seeds_to_dti.value()!=""){
    if(!fsl_imageexists(opts.seeds_to_dti.value())){  // presumably ascii file provided
      //m _Seeds_to_DTI = read_ascii_matrix(opts.seeds_to_dti.value());
      // m_DTI_to_Seeds = m_Seeds_to_DTI.i();
      // m_rotdir       = m_Seeds_to_DTI.SubMatrix(1,3,1,3);
    }else{
      tData.IsNonlinXfm = true;
      FnirtFileReader ffr(opts.seeds_to_dti.value());
      volume4D<float> SeedDTIwarp4D = ffr.FieldAsNewimageVolume4D(true);
      int size=SeedDTIwarp4D.xsize()*SeedDTIwarp4D.ysize()*SeedDTIwarp4D.zsize();
      tData.Warp_S2D_sizes[0]=SeedDTIwarp4D.xsize();
      tData.Warp_S2D_sizes[1]=SeedDTIwarp4D.ysize();
      tData.Warp_S2D_sizes[2]=SeedDTIwarp4D.zsize();
      tData.SeedDTIwarp= new float[3*size];
      for(int v=0;v<3;v++){
        for(int z=0;z<tData.Warp_S2D_sizes[2];z++){
          for(int y=0;y<tData.Warp_S2D_sizes[1];y++){
            for(int x=0;x<tData.Warp_S2D_sizes[0];x++){
              tData.SeedDTIwarp[v*size+z*tData.Warp_S2D_sizes[0]*tData.Warp_S2D_sizes[1]+y*tData.Warp_S2D_sizes[0]+x]=SeedDTIwarp4D[v](x,y,z);
            }
          }
        }
      }
      if(opts.dti_to_seeds.value()==""){
        cerr << "TRACT::Streamliner:: DTI -> Seeds transform needed" << endl;
        exit(1);
      }
      FnirtFileReader iffr(opts.dti_to_seeds.value());
      volume4D<float> DTISeedwarp4D = iffr.FieldAsNewimageVolume4D(true);
      size=DTISeedwarp4D.xsize()*DTISeedwarp4D.ysize()*DTISeedwarp4D.zsize();
      tData.Warp_D2S_sizes[0]=DTISeedwarp4D.xsize();
      tData.Warp_D2S_sizes[1]=DTISeedwarp4D.ysize();
      tData.Warp_D2S_sizes[2]=DTISeedwarp4D.zsize();
      tData.DTISeedwarp = new float[3*size];
      for(int v=0;v<3;v++){
        for(int z=0;z<tData.Warp_D2S_sizes[2];z++){
          for(int y=0;y<tData.Warp_D2S_sizes[1];y++){
            for(int x=0;x<tData.Warp_D2S_sizes[0];x++){
              tData.DTISeedwarp[v*size+z*tData.Warp_D2S_sizes[0]*tData.Warp_D2S_sizes[1]+y*tData.Warp_D2S_sizes[0]+x]=DTISeedwarp4D[v](x,y,z);
            }
          }
        }
      }
      Matrix samp=IdentityMatrix(4);
      samp(1,1) = tData.Sdims[0];
      samp(2,2) = tData.Sdims[1];
      samp(3,3) = tData.Sdims[2];
      samp=samp.i();
      tData.SsamplingI[0]=samp(1,1);
      tData.SsamplingI[1]=samp(2,2);
      tData.SsamplingI[2]=samp(3,3);
      samp=IdentityMatrix(4);
      samp(1,1) = tData.Ddims[0];
      samp(2,2) = tData.Ddims[1];
      samp(3,3) = tData.Ddims[2];
      samp=samp.i();
      tData.DsamplingI[0]=samp(1,1);
      tData.DsamplingI[1]=samp(2,2);
      tData.DsamplingI[2]=samp(3,3);
      samp=IdentityMatrix(4);
      samp(1,1) = SeedDTIwarp4D.xdim();
      samp(2,2) = SeedDTIwarp4D.ydim();
      samp(3,3) = SeedDTIwarp4D.zdim();
      samp=samp.i();
      tData.Wsampling_S2D_I[0]=samp(1,1);
      tData.Wsampling_S2D_I[1]=samp(2,2);
      tData.Wsampling_S2D_I[2]=samp(3,3);
      samp=IdentityMatrix(4);
      samp(1,1) = DTISeedwarp4D.xdim();
      samp(2,2) = DTISeedwarp4D.ydim();
      samp(3,3) = DTISeedwarp4D.zdim();
      samp=samp.i();
      tData.Wsampling_D2S_I[0]=samp(1,1);
      tData.Wsampling_D2S_I[1]=samp(2,2);
      tData.Wsampling_D2S_I[2]=samp(3,3);
    }
  }


  ///////////////////////////////
  ///////// AVOID MASKs /////////
  ///////////////////////////////
  tData.avoid.nlocs=0;
  tData.avoid.NVols=0;
  tData.avoid.NSurfs=0;
  if(opts.rubbishfile.value()!=""){
    load_rois_mixed(opts.rubbishfile.value(),mm2vox,tData.Sdims,tData.Ssizes,tData.avoid);
  }

  ///////////////////////////////
  ///////// STOP MASKs //////////
  ///////////////////////////////
  tData.forcefirststep=opts.forcefirststep.value();
  tData.stop.nlocs=0;
  tData.stop.NVols=0;
  tData.stop.NSurfs=0;
  if(opts.stopfile.value()!=""){
    load_rois_mixed(opts.stopfile.value(),mm2vox,tData.Sdims,tData.Ssizes,tData.stop);
  }

  ///////////////////////////////
  //////// WTSTOP MASKs /////////
  ///////////////////////////////
  tData.wtstop.nlocs=0;
  tData.wtstop.NVols=0;
  tData.wtstop.NSurfs=0;
  Matrix null;
  if(opts.wtstopfiles.set()){
    load_rois(opts.wtstopfiles.value(),mm2vox,tData.Sdims,tData.Ssizes,0,refVol,tData.wtstop,null);
  }

  /////////////////////////////////
  //////// WAYPOINT MASKs /////////
  /////////////////////////////////
  tData.oneway=opts.onewaycondition.value();
  tData.wayorder=opts.wayorder.value();
  tData.waycond=false; // OR
  if(opts.waycond.value()=="AND") tData.waycond=1;
  tData.waypoint.nlocs=0;
  tData.waypoint.NVols=0;
  tData.waypoint.NSurfs=0;
  if(opts.waypoints.set()){
    load_rois(opts.waypoints.value(),mm2vox,tData.Sdims,tData.Ssizes,0,refVol,tData.waypoint,null);
  }

  /////////////////////////////////
  ///////// TARGET MASKs //////////
  /////////////////////////////////
  tData.targets.nlocs=0;
  tData.targets.NVols=0;
  tData.targets.NSurfs=0;
  tData.targetsREF.nlocs=0;
  tData.targetsREF.NVols=0;
  tData.targetsREF.NSurfs=0;
  if(opts.s2tout.value()){
    load_rois(opts.targetfile.value(),mm2vox,tData.Sdims,tData.Ssizes,0,refVol,tData.targets,null);
    long total_s2targets=tData.nseeds*(tData.targets.NVols+tData.targets.NSurfs);
    m_s2targets=new float[total_s2targets];
    for(long i=0;i<total_s2targets;i++){
      m_s2targets[i]=0;
    }
    if(opts.omeanpathlength.value()){
      m_s2targetsb=new float[total_s2targets];
      for(long i=0;i<total_s2targets;i++){
        m_s2targetsb[i]=0;
      }
    }

    // gather the target mask file names
    // and extract their base names to
    // use as names for each target.
    vector<string> targetnames;

    // target is single volume/mesh
    if (fsl_imageexists(opts.targetfile.value()) ||
        meshExists(opts.targetfile.value())) {
      targetnames.push_back(opts.targetfile.value());
    }
    // text file containing paths to multiple targets
    else {
      ifstream fs(opts.targetfile.value().c_str());

      if (!fs.good()) {
        cerr << "Could not open " << opts.targetfile.value() << endl;
        exit(1);
      }
      copy(istream_iterator<string>(fs),
           istream_iterator<string>(),
           back_inserter(targetnames));
    }

    // extract base name (folder/suffix
    // stripped) of each target file
    for (auto name : targetnames) {

      size_t start = name.find_last_of("/");
      size_t sz    = name.size();
      if (start != string::npos) {
        name = name.substr(start + 1, sz);
        sz   = name.size();
      }

      if (sz > 4 && ((name.rfind(".nii") == (sz - 4)) ||
                     (name.rfind(".asc") == (sz - 4)))) {
        name = name.substr(0, sz - 4);
      }
      else if (sz > 7 && (name.rfind(".nii.gz") == (sz - 7))) {
        name = name.substr(0, sz - 7);
      }
      tData.targetnames.push_back(name);
    }

    /////// TARGET REF MASK ////////
    load_rois_mixed(opts.targetfile.value(),mm2vox,tData.Sdims,tData.Ssizes,tData.targetsREF);
  }

  ///////////////////////////////////
  ///////// MATRIX 1 MASKs //////////
  ///////////////////////////////////
  tData.Seeds_to_M2=new float[12];
  tData.lrmatrix1.distthresh=opts.distthresh1.value();
  tData.lrmatrix1.nlocs=0;
  tData.lrmatrix1.NVols=0;
  tData.lrmatrix1.NSurfs=0;
  Matrix m1_coords;

  if(opts.matrix1out.value()||opts.matrix2out.value()){

    load_rois_matrix1(tData,opts.seedfile.value(),mm2vox,tData.Sdims,tData.Ssizes,2,refVol,tData.lrmatrix1,m1_coords);

    if(opts.matrix1out.value())
      write_ascii_matrix(m1_coords,logger.appendDir("coords_for_fdt_matrix1"));
    else
      write_ascii_matrix(m1_coords,logger.appendDir("coords_for_fdt_matrix2"));
    ///////////////////////////////////
    ////////// MATRIX 2 MASKs /////////
    ///////////////////////////////////
    Matrix lrm1_coords;
    if(opts.matrix2out.value()){
      volume<float> tmpvolM2;
      read_volume(tmpvolM2,opts.lrmask.value());
      // lrmask / --target2 must be defined in seed space,
      // but may be downsampled, so the seed -> target2
      // affine is just a scaling matrix (identity if
      // it is the same resolution as the seed)
      memset(tData.Seeds_to_M2, 0, 12 * sizeof(float));
      tData.Seeds_to_M2[0]  = tData.Sdims[0] / tmpvolM2.xdim();
      tData.Seeds_to_M2[5]  = tData.Sdims[1] / tmpvolM2.ydim();
      tData.Seeds_to_M2[10] = tData.Sdims[2] / tmpvolM2.zdim();
      tData.M2sizes = new int[3];
      tData.M2sizes[0]=tmpvolM2.xsize();
      tData.M2sizes[1]=tmpvolM2.ysize();
      tData.M2sizes[2]=tmpvolM2.zsize();

      tData.lrmatrix1.nlocs=0;
      tData.lrmatrix1.NVols=0;
      tData.lrmatrix1.NSurfs=0;
      load_rois(opts.lrmask.value(),mm2vox,tData.Sdims,tData.M2sizes,1,refVol,tData.lrmatrix1,lrm1_coords);
      write_ascii_matrix(lrm1_coords,logger.appendDir("tract_space_coords_for_fdt_matrix2"));
      nColsMat1=tData.lrmatrix1.nlocs;

      volume<int> m_lookup2;
      volume<int> m_lrmask;
      read_volume(m_lrmask,opts.lrmask.value());
      m_lookup2.reinitialize(m_lrmask.xsize(),m_lrmask.ysize(),m_lrmask.zsize());
      copybasicproperties(m_lrmask,m_lookup2);
      m_lookup2=0;
      int numnz=0;
      for(int Wz=m_lrmask.minz();Wz<=m_lrmask.maxz();Wz++){
        for(int Wy=m_lrmask.miny();Wy<=m_lrmask.maxy();Wy++){
          for(int Wx=m_lrmask.minx();Wx<=m_lrmask.maxx();Wx++){
            if(m_lrmask.value(Wx,Wy,Wz)!=0){
              numnz++;
              m_lookup2(Wx,Wy,Wz)=numnz;
            }
          }
        }
      }
      save_volume(m_lookup2,logger.appendDir("lookup_tractspace_fdt_matrix2"));
    }else{
      nColsMat1=tData.lrmatrix1.nlocs;
    }

    ConMat1 = new float*[tData.nseeds];
    nRowsMat1=tData.nseeds;
    for(int i=0;i<nRowsMat1;i++){
      ConMat1[i]=new float[nColsMat1];
      for(int j=0;j<nColsMat1;j++)
        ConMat1[i][j]=0;
    }
    if(opts.omeanpathlength.value()){
      ConMat1b = new float*[tData.nseeds];
      for(int i=0;i<nRowsMat1;i++){
        ConMat1b[i]=new float[nColsMat1];
        for(int j=0;j<nColsMat1;j++)
          ConMat1b[i][j]=0;
      }
    }
    if(opts.matrix2out.value()){
      printf("Dimensions Matrix2: %i x %i\n",nRowsMat1,nColsMat1);
    }else{
      printf("Dimensions Matrix1: %i x %i\n",nRowsMat1,nColsMat1);
    }
  }

  ///////////////////////////////////
  ///////// MATRIX 3 MASKs //////////
  ///////////////////////////////////
  tData.matrix3.distthresh=opts.distthresh3.value();
  tData.matrix3.nlocs=0;
  tData.matrix3.NVols=0;
  tData.matrix3.NSurfs=0;

  tData.lrmatrix3.distthresh=opts.distthresh3.value();
  tData.lrmatrix3.nlocs=0;
  tData.lrmatrix3.NVols=0;
  tData.lrmatrix3.NSurfs=0;
  Matrix m3_coords;

  if(opts.matrix3out.value()){

    load_rois(opts.mask3.value(),mm2vox,tData.Sdims,tData.Ssizes,2,refVol,tData.matrix3,m3_coords);

    write_ascii_matrix(m3_coords,logger.appendDir("coords_for_fdt_matrix3"));

    ConMat3 = new float*[tData.matrix3.nlocs];
    nRowsMat3=tData.matrix3.nlocs;

    ///////////////////////////////////
    //////// LRMATRIX 3 MASKs /////////
    ///////////////////////////////////

    if(opts.lrmask3.value()!=""){
      Matrix lrm3_coords;
      load_rois(opts.lrmask3.value(),mm2vox,tData.Sdims,tData.Ssizes,2,refVol,tData.lrmatrix3,lrm3_coords);

      write_ascii_matrix(lrm3_coords,logger.appendDir("tract_space_coords_for_fdt_matrix3"));
      nColsMat3=tData.lrmatrix3.nlocs;
    }else{
      nColsMat3=tData.matrix3.nlocs;
    }
    for(int i=0;i<nRowsMat3;i++){
      ConMat3[i]=new float[nColsMat3];
      for(int j=0;j<nColsMat3;j++)
        ConMat3[i][j]=0;
    }
    if(opts.omeanpathlength.value()){
      ConMat3b = new float*[tData.matrix3.nlocs];
      for(int i=0;i<nRowsMat3;i++){
        ConMat3b[i]=new float[nColsMat3];
        for(int j=0;j<nColsMat3;j++)
          ConMat3b[i][j]=0;
      }
    }

    printf("Dimensions Matrix3 %i x %i\n",nRowsMat3,nColsMat3);
  }

  ////////////////////////////////
  //////// NETWORK MASKs /////////
  ////////////////////////////////
  tData.network.nlocs=0;
  tData.network.NVols=0;
  tData.network.NSurfs=0;
  tData.networkREF.nlocs=0;
  tData.networkREF.NVols=0;
  tData.networkREF.NSurfs=0;

  if(opts.network.value()){
    load_rois(opts.seedfile.value(),mm2vox,tData.Sdims,tData.Ssizes,0,refVol,tData.network,null);

    nRowsNet=tData.network.NVols+tData.network.NSurfs;
    nColsNet=tData.network.NVols+tData.network.NSurfs;
    ConNet = new float*;
    ConNet[0] = new float[nRowsNet*nColsNet];
    printf("Dimensions Network Matrix: %i x %i\n",nRowsNet,nColsNet);

    if(opts.omeanpathlength.value()){
      ConNetb = new float*;
      ConNetb[0] = new float[nRowsNet*nColsNet];
    }

    /////// NETWORK REF MASK ////////
    load_rois_mixed(opts.seedfile.value(),mm2vox,tData.Sdims,tData.Ssizes,tData.networkREF);
  }
}
