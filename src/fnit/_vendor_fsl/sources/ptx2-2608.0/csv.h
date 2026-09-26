/*  Combined Surfaces and Volumes Class

    Saad Jbabdi  - FMRIB Image Analysis Group

    Copyright (C) 2010 University of Oxford  */

/*  CCOPYRIGHT  */


#if !defined (CSV_H)
#define CSV_H

#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define VOLUME  0
#define SURFACE 1

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include "warpfns/warpfns.h"
#include "warpfns/fnirt_file_reader.h"
#include "newimage/newimageio.h"
#include "meshclass/triangle.h"
#include "miscmaths/miscmaths.h"
#include "fslvtkio/fslvtkio.h"
#include "utils/tracer_plus.h"
#include "csv_mesh.h"

// useful routines for collision detection
bool triBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3]);
bool rayBoxIntersection(float origin[3],float direction[3],float vminmax[2][3]);
bool segTriangleIntersection(float seg[2][3],float tri[3][3]);
float triDistPoint(const mesh::Triangle& t,const NEWMAT::ColumnVector pos);

// This is the main class that can handle an ROI made of a bunch of voxels and surface vertices
// Its main role is to tell you whether a point is in the ROI (voxels) or whether a line segment
// crossed its surface
// This class is quite tailored to fit with probtrackx requirements :)
// For now: it can only detect collisions with triangle meshes
// For now: it uses meshclass for I/O. Eventually port into Brian's I/O to support GIFTI

class CSV {
private:
  NEWIMAGE::volume<float>                          roivol;     // volume-like ROIs (all in the same volume for memory efficiency)
  std::vector<CsvMesh>                             roimesh;    // surface-like ROIs

  std::vector<std::string>                         roinames;   // file names
                                                               // useful for seed_to_target storage in probtrackx

  NEWMAT::Matrix                                   hitvol;     // hit counter (voxels)
  NEWMAT::Matrix                                   isinroi;    // which voxel in which ROI
  NEWMAT::Matrix                                   mat2vol;    // lookup volume coordinates
  NEWIMAGE::volume<int>                            vol2mat;    // lookup matrix row

  NEWIMAGE::volume<int>                            surfvol;    // this is the main thing that speeds up surface collision detection
                                                               // where each voxel knows whether the surface goes through or not
                                                               // the values inicate which triangle list crosses the voxel
                                                               // this volume can be higher resolution than refvol
  std::vector< std::vector< std::pair<int,int> > > triangles;  // triangles crossed by voxels as pair<mesh,triangle>

  NEWMAT::Matrix                                   mm2vox;     // for surfaces, transform coords into voxel space
                                                               // this will depend on various software conventions (argh)
  NEWMAT::Matrix                                   vox2mm;     // the inverse transform
  std::string                                      convention; // FREESURFER/FIRST/CARET/etc.?


  float                                            _xdim;      // voxel sizes
  float                                            _ydim;
  float                                            _zdim;
  NEWMAT::ColumnVector                             _dims;

  NEWMAT::Matrix                                   _identity;  // useful for highres<->lowres

  int                                              nvols;      // # volume-like ROIs
  int                                              nsurfs;     // # surface-like ROIs
  int                                              nvoxels;    // # voxels (total across ROIs - unrepeated)
  int                                              nlocs;      // # ROI locations (eventually repeated)
  int                                              nrois;      // # ROIs in total (volumes+surfaces)

  // Sort out the transformations
  // from a given loc, I need to be able to tell
  // whether it is a surface or volume
  // + which surface or volume
  // + which sublocation within the surface or volume
  std::vector<int>                   loctype;            // SURFACE OR VOLUME
  std::vector<int>                   locroi;             // 0:NROIS-1
  std::vector<int>                   locsubroi;          // 0:NSURFS or 0:NVOLS
  std::vector<int>                   locsubloc;          // 0:n
  std::vector<NEWMAT::ColumnVector>  loccoords;
  std::vector< std::vector<int> >    mesh2loc;
  NEWIMAGE::volume4D<int>            vol2loc;
  std::vector<int>                   volind;             // vol is which roi index?
  std::vector<int>                   surfind;            // surf is which roi index?
  std::vector<int>                   roitype;
  std::vector<int>                   roisubind;          // roi is which subindex?

  NEWIMAGE::volume<short int>        refvol;             // reference volume (in case no volume-like ROI is used)
  bool                               refvolset;          // flag: do we have a refvol?

  std::vector<NEWMAT::ColumnVector>  maps;               // extra maps associated with CSV locations

public:
  CSV(){
    refvolset=false;
    init_dims();
  }
  CSV(const NEWIMAGE::volume<short int>& ref):refvol(ref){
    refvolset=true;
    init_dims();
    _xdim=refvol.xdim();
    _ydim=refvol.ydim();
    _zdim=refvol.zdim();
    _dims.ReSize(3);_dims<<_xdim<<_ydim<<_zdim;
    set_convention("caret");
    roivol.reinitialize(refvol.xsize(),refvol.ysize(),refvol.zsize());
    NEWIMAGE::copybasicproperties(refvol,roivol);
    roivol=0;
    vol2mat.reinitialize(refvol.xsize(),refvol.ysize(),refvol.zsize());
    vol2mat=0;
    NEWIMAGE::copybasicproperties(refvol,surfvol);
    surfvol.reinitialize(refvol.xsize(),
			 refvol.ysize(),
			 refvol.zsize());
    surfvol=0;
  }
  CSV(const CSV& csv){*this=csv;}
  ~CSV(){clear_all();}

  void init_dims(){
    nvols=0;
    nsurfs=0;
    nvoxels=0;
    nlocs=0;
    nrois=0;
    _identity=NEWMAT::IdentityMatrix(4);
  }
  void reinitialize(const NEWIMAGE::volume<short int>& vol){
    init_dims();
    _xdim=vol.xdim();
    _ydim=vol.ydim();
    _zdim=vol.zdim();
    _dims.ReSize(3);_dims<<_xdim<<_ydim<<_zdim;
    set_convention("caret");
    set_refvol(vol);
    clear_all();
  }

  void clear_all(){
    roimesh.clear();roinames.clear();
    loctype.clear();locroi.clear();locsubroi.clear();locsubloc.clear();
    mesh2loc.clear();maps.clear();
  }

  // get/set
  const NEWIMAGE::volume<short int>& get_refvol()const{return refvol;}
  void set_refvol(const NEWIMAGE::volume<short int>& vol){
    refvolset=true;
    refvol=vol;
    _xdim=refvol.xdim();
    _ydim=refvol.ydim();
    _zdim=refvol.zdim();
    _dims<<_xdim<<_ydim<<_zdim;
    roivol.reinitialize(refvol.xsize(),refvol.ysize(),refvol.zsize());
    NEWIMAGE::copybasicproperties(refvol,roivol);
    roivol=0;
    vol2mat.reinitialize(refvol.xsize(),refvol.ysize(),refvol.zsize());
    vol2mat=0;
    NEWIMAGE::copybasicproperties(refvol,surfvol);
    surfvol.reinitialize(refvol.xsize(),
			 refvol.ysize(),
			 refvol.zsize());
    surfvol=0;

  }
  void change_refvol(const std::vector<std::string>& fnames){
    for(unsigned int i=0;i<fnames.size();i++){
      if(NEWIMAGE::fsl_imageexists(fnames[i])){
	NEWIMAGE::volume<short int> tmpvol;
	read_volume(tmpvol,fnames[i]);
	set_refvol(tmpvol);
	return;
      }
    }
  }
  void init_vol2loc(const std::vector<std::string>& fnames){
    int cnt=0;
    for(unsigned int i=0;i<fnames.size();i++){
      if(NEWIMAGE::fsl_imageexists(fnames[i])){
	cnt++;
      }
    }
    vol2loc.reinitialize(vol2mat.xsize(),vol2mat.ysize(),vol2mat.zsize(),cnt);
    vol2loc=0;
  }
  inline const float xdim()const{return _xdim;}
  inline const float ydim()const{return _ydim;}
  inline const float zdim()const{return _zdim;}
  inline const int   xsize()const{return refvol.xsize();}
  inline const int   ysize()const{return refvol.ysize();}
  inline const int   zsize()const{return refvol.zsize();}

  int nRois ()const{return nrois;}
  int nSurfs()const{return nsurfs;}
  int nVols ()const{return nvols;}
  int nLocs ()const{return nlocs;}

  int nActVertices(const int& i)const{return (int)mesh2loc[i].size();}
  int nVertices(const int& i)const{return (int)roimesh[i]._points.size();}

  std::string get_name(const int& i){
    return roinames[i];
  }
  int get_roitype(const int& i){return roitype[i];}
  CsvMesh& get_mesh(const int i){return roimesh[i];}
  int get_volloc(int subroi,int x,int y,int z)const{
    return vol2loc(x,y,z,subroi);
  }
  int get_surfloc(int subroi,int subloc)const{
    return mesh2loc[subroi][subloc];
  }

  NEWMAT::ColumnVector get_loc_coord(const int& loc){
    return loccoords[loc];
  }
  NEWMAT::ColumnVector get_loc_coord_as_vox(const int& loc){
    if(loctype[loc]==VOLUME){
      return loccoords[loc];
    }
    else{
      return get_vertex_as_vox(locsubroi[loc],locsubloc[loc]);
    }
  }
  std::vector<NEWMAT::ColumnVector> get_locs_coords()const{return loccoords;}
  std::vector<int> get_locs_roi_index()const{
    std::vector<int> ret;
    for(unsigned int i=0;i<locroi.size();i++){
      ret.push_back(locroi[i]);
    }
    return ret;
  }
  std::vector<int> get_locs_coord_index()const{
    std::vector<int> ret;
    for(unsigned int i=0;i<locsubloc.size();i++){
      ret.push_back(locsubloc[i]);
    }
    return ret;
  }


  // indices start at 0
  NEWMAT::ColumnVector get_vertex(const int& surfind,const int& vertind){
    CsvMpoint P(roimesh[surfind].get_point(vertind));
    NEWMAT::ColumnVector p(3);
    p << P.get_coord().X
      << P.get_coord().Y
      << P.get_coord().Z;
    return p;
  }
  NEWMAT::ColumnVector get_vertex_as_vox(const int& surfind,const int& vertind){
    CsvMpoint P(roimesh[surfind].get_point(vertind));
    NEWMAT::ColumnVector p(4);
    p << P.get_coord().X
      << P.get_coord().Y
      << P.get_coord().Z
      << 1;
    p = mm2vox*p;
    p = p.SubMatrix(1,3,1,1);
    return p;
  }
   NEWMAT::ColumnVector get_normal(const int& surfind,const int& vertind){
     mesh::Vec n = roimesh[surfind].local_normal(vertind);
     NEWMAT::ColumnVector ret(3);
     ret<<n.X<<n.Y<<n.Z;
     return ret;
   }
   NEWMAT::ColumnVector get_normal_as_vox(const int& surfind,const int& vertind){
     mesh::Vec n = roimesh[surfind].local_normal(vertind);
     NEWMAT::ColumnVector ret(3);
     ret<<n.X<<n.Y<<n.Z;
     ret=mm2vox.SubMatrix(1,3,1,3)*ret;
     if(ret.MaximumAbsoluteValue()>0)
       ret/=std::sqrt(ret.SumSquare());
     return ret;
   }


  // routines
  void load_volume  (const std::string& filename);
  void load_surface (const std::string& filename);
  void load_rois    (const std::string& filename,bool do_change_refvol=true);
  void reload_rois  (const std::string& filename);
  void save_roi     (const int& roiind,const std::string& prefix);
  void save_rois    (const std::string& prefix);
  void divide_rois  (CSV csv2);
  void divide_maps  (CSV csv2);
  //void save_as_volume (const string& prefix);
  void fill_volume  (NEWIMAGE::volume<float>& vol,const int& ind);
  void cleanup();

  void set_convention(const std::string& conv);
  void switch_convention(const std::string& new_convention,const NEWMAT::Matrix& vox2vox,const NEWMAT::ColumnVector& old_dims);
  void switch_convention(const std::string& new_convention,const NEWIMAGE::volume4D<float>& new2old_warp,
			 const NEWIMAGE::volume<short int>& oldref,const NEWIMAGE::volume<short int>& newref);


  void init_surfvol();
  void update_surfvol(const std::vector<NEWMAT::ColumnVector>& v,const int& id,const int& meshid);
  void save_surfvol(const std::string& filename,const bool& binarise=true)const;
  void save_normalsAsVol(const int& meshind,const std::string& filename)const;
  void init_hitvol(const std::vector<std::string>& fnames);


  void  add_value(const int& loc,const float& val);
  void  set_value(const int& loc,const float& val);
  void  set_vol_values(const float& val);
  float get_value(const int& loc)const;
  void  reset_values();
  void  reset_values(const std::vector<int>& locs);
  NEWMAT::ReturnMatrix  get_all_values()const;
  void  set_all_values(const NEWMAT::ColumnVector& vals);
  void  set_all_values(const float& val);
  void  save_values(const int& roi);
  void  loc_info(const int& loc)const;

  void  add_map_value(const int& loc,const float& val,const int& map);
  void  set_map_value(const int& loc,const float& val,const int& map);
  void  reset_maps(){maps.clear();}
  void  add_map(){
    NEWMAT::ColumnVector map(nlocs);
    map=0.0;
    maps.push_back(map);
  }
  void  add_map(const NEWMAT::ColumnVector& map){
    if(map.Nrows()!=nlocs){
      std::cerr<<"CSV::add_map: map does not contain the correct number of entries"<<std::endl;
      exit(1);
    }
    else{
      maps.push_back(map);
    }
  }
  void save_map(const int& roiind,const int& mapind,const std::string& fname);

  bool isInRoi(int x,int y,int z)const{return (roivol(x,y,z)!=0);}
  bool isInRoi(int x,int y,int z,int roi)const{
    if(!isInRoi(x,y,z))return false;
    return (isinroi(vol2mat(x,y,z),roi));
  }

  bool has_crossed(const NEWMAT::ColumnVector& x1,const NEWMAT::ColumnVector& x2,
		   const std::vector<NEWMAT::ColumnVector>& crossedvox,bool docount=false,bool docontinue=false,const float& val=1.0);
  bool has_crossed_roi(const NEWMAT::ColumnVector& x1,const NEWMAT::ColumnVector& x2,
		       const std::vector<NEWMAT::ColumnVector>& crossedvox,std::vector<int>& crossedrois)const;
  bool has_crossed_roi_vols(const NEWMAT::ColumnVector& x1,std::vector<int>& crossedrois)const;
  bool has_crossed_roi(const NEWMAT::ColumnVector& x1,const NEWMAT::ColumnVector& x2,
		       const std::vector<NEWMAT::ColumnVector>& crossedvox,std::vector<int>& crossedrois,std::vector<int>& crossedlocs,
		       std::vector< std::pair<int,int> >& surf_Triangle,bool closestvertex)const;


  int step_sign(const int& loc,const mesh::Vec& step)const;
  int coord_sign(const int& loc,const NEWMAT::ColumnVector& x2)const;

  // "near-collision" detection
  // returns true if one of the surfaces (meshind) is closer than dist.
  // also returns (in dir) the direction towards the nearest surface vertex that is within dist
  // as well as the index for that closest surface
  bool is_near_surface(const NEWMAT::ColumnVector& pos,const float& dist,NEWMAT::ColumnVector& dir,float& mindist);

  void find_crossed_voxels(const NEWMAT::ColumnVector& x1,const NEWMAT::ColumnVector& x2,std::vector<NEWMAT::ColumnVector>& crossed);
  void tri_crossed_voxels(float tri[3][3],std::vector<NEWMAT::ColumnVector>& crossed);
  void line_crossed_voxels(float line[2][3],std::vector<NEWMAT::ColumnVector>& crossed)const;


  // Operators
  const CSV& operator=(const float& val){
    this->set_all_values(val);
    return *this;
  }
  inline float operator()(int loc){
    return get_value(loc);
  }

  // COPY
  CSV& operator=(const CSV& rhs){
    roivol=rhs.roivol;
    roimesh=rhs.roimesh;
    roinames=rhs.roinames;
    hitvol=rhs.hitvol;
    isinroi=rhs.isinroi;
    mat2vol=rhs.mat2vol;
    vol2mat=rhs.vol2mat;
    surfvol=rhs.surfvol;
    triangles=rhs.triangles;
    mm2vox=rhs.mm2vox;
    vox2mm=rhs.vox2mm;
    convention=rhs.convention;
    _xdim=rhs._xdim;
    _ydim=rhs._ydim;
    _zdim=rhs._zdim;
    _dims=rhs._dims;
    _identity=rhs._identity;
    nvols=rhs.nvols;
    nsurfs=rhs.nsurfs;
    nvoxels=rhs.nvoxels;
    nlocs=rhs.nlocs;
    nrois=rhs.nrois;
    loctype=rhs.loctype;
    locroi=rhs.locroi;
    locsubroi=rhs.locsubroi;
    locsubloc=rhs.locsubloc;
    loccoords=rhs.loccoords;
    mesh2loc=rhs.mesh2loc;
    vol2loc=rhs.vol2loc;
    volind=rhs.volind;
    surfind=rhs.surfind;
    roitype=rhs.roitype;
    roisubind=rhs.roisubind;
    refvol=rhs.refvol;
    refvolset=rhs.refvolset;
    maps=rhs.maps;
    return *this;
  }

};


#endif
