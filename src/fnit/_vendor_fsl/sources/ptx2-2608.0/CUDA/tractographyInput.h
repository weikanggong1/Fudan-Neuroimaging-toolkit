/*  tractographyInput.h

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <vector>

#include "armawrap/newmat.h"
#include "newimage/newimage.h"

#include "CUDA/tractographyData.h"
#include "CUDA/options/options.h"
#include "csv.h"

#define ASCII 5
#define VTK   6
#define GIFTI 7

/** \brief This class contain the methods to read the input files passsed to GPU tractography.*/
class tractographyInput{

public:

  tractographyInput();

  /// Method load all the necessary data from the input files to perform GPU Tractography
  void load_tractographyData( tractographyData&           tData,
                              NEWIMAGE::volume<float>*&   m_prob,
                              NEWIMAGE::volume<float>*&   m_prob2,
                              float**&                    ConNet,
                              float**&                    ConNetb,
                              int&                        nRowsNet,
                              int&                        nColsNet,
                              float**&                    ConMat1,
                              float**&                    ConMat1b,
                              int&                        nRowsMat1,
                              int&                        nColsMat1,
                              float**&                    ConMat3,
                              float**&                    ConMat3b,
                              int&                        nRowsMat3,
                              int&                        nColsMat3,
                              float*&                     m_s2targets,
                              float*&                     m_s2targetsb,
                              NEWIMAGE::volume4D<float>*& m_localdir);

  /// General Method to read a Surface file in ASCII, VTK or GIFTI format
  void load_mesh(   std::string&        filename,
                    std::vector<float>& vertices,   // all the vertices, same order than file
                    std::vector<int>&   faces,      // all the faces, same order than file
                    std::vector<int>&   locs,       // used to store the id of a vertex in the Matrix. If -1, then vertex is non-activated
                    int&                nlocs,      // number of ids(vertices) in the Matrix
                    bool                wcoords,    // save coordinates of the vertices in a file ?
                    int                 nroi,       // number of ROI to identify coordinates
                    std::vector<float>& coords);    // coordinates xyz of the vertices

  /// Method to read a surface file in ASCII format
  void load_mesh_ascii( std::string&        filename,
                        std::vector<float>& vertices,
                        std::vector<int>&   faces,
                        std::vector<int>&   locs,
                        int&                nlocs,
                        bool                wcoords,
                        int                 nroi,
                        std::vector<float>& coords);

  /// Method to read a surface file in VTK format
  void load_mesh_vtk(   std::string&        filename,
                        std::vector<float>& vertices,
                        std::vector<int>&   faces,
                        std::vector<int>&   locs,
                        int&                nlocs,
                        bool                wcoords,
                        int                 nroi,
                        std::vector<float>& coords);

  /// Method to read a surface file in GIFTI format
  void load_mesh_gifti( std::string&        filename,
                        std::vector<float>& vertices,
                        std::vector<int>&   faces,
                        std::vector<int>&   locs,
                        int&                nlocs,
                        bool                wcoords,
                        int                 nroi,
                        std::vector<float>& coords);

  /// Method to read a Volume
  void load_volume( std::string&        filename,
                    int*                Ssizes,
                    float*              Vout,
                    int&                nlocs,
                    bool                reset,
                    bool                wcoords,
                    int                 nroi,
                    std::vector<float>& coords);

  /// Method to initialise the realtionship between voxels and triangles for a Surface
  void init_surfvol(    int*                Ssizes,
                        NEWMAT::Matrix&     mm2vox,
                        std::vector<float>& vertices,
                        int*                faces,
                        int                 sizefaces,     // number of faces this time (maybe there are several ROIs for the same mask)
                        int                 initfaces,     // number of faces in previos times
                        std::vector<int>&   voxFaces,      // list of faces of all the voxels
                        int*                voxFacesIndex, // starting point of each voxel in the list
                        std::vector<int>&   locsV);

  /// Method to find out what voxels are crossed by a triangle
  void csv_tri_crossed_voxels(float                              tri[3][3],
                              std::vector<NEWMAT::ColumnVector>& crossed);

  /// Method to read all the ROIs of a mask in the same structure: for stop and avoid masks
  void load_rois_mixed(std::string    filename,
                       NEWMAT::Matrix mm2vox,
                       float*         Sdims,
                       int*           Ssizes,
                       // Output
                       MaskData&      matData);

  /// Method to read the ROIs of a mask in concatenated structures: for wtstop and waypoints masks
  void load_rois(    // Input
                 std::string              filename,
                 NEWMAT::Matrix           mm2vox,
                 float*                   Sdims,
                 int*                     Ssizes,
                 int                      wcoords,
                 NEWIMAGE::volume<float>& refVol,
                 // Output
                 MaskData&                matData,
                 NEWMAT::Matrix&          coords);

  /// Same than load_rois but it includes the initialisation of the rows (including triangles) of Matrix1
  void  load_rois_matrix1(  tractographyData&        tData,
                            // Input
                            std::string              filename,
                            NEWMAT::Matrix           mm2vox,
                            float*                   Sdims,
                            int*                     Ssizes,
                            bool                     wcoords,
                            NEWIMAGE::volume<float>& refVol,
                            // Output
                            MaskData&                data,
                            NEWMAT::Matrix&          coords);

  /// Method to load the seeds. Can be defined by volumes and/or by surfaces
  size_t load_seeds_rois(tractographyData&           tData,
                         std::string                 seeds_filename,
                         std::string                 ref_filename,
                         float*                      Sdims,
                         int*                        Ssizes,
                         int                         convention,
                         float*&                     seeds,
                         int*&                       seeds_ROI,
                         NEWMAT::Matrix&             mm2vox,
                         float*                      vox2mm,
                         NEWIMAGE::volume<float>*&   m_prob,
                         bool                        initialize_m_prob,
                         NEWIMAGE::volume<float>*&   m_prob2,
                         bool                        initialize_m_prob2,
                         NEWIMAGE::volume4D<float>*& m_localdir,
                         NEWIMAGE::volume<float>&    refVol);

  /// Method to set the transformation: voxel to milimeters
  void set_vox2mm(int                     convention,
                  float*                  Sdims,
                  int*                    Ssizes,
                  NEWIMAGE::volume<float> vol,
                  NEWMAT::Matrix&         mm2vox,   // 4x4
                  float*                  vox2mm);  // 4x4

};
