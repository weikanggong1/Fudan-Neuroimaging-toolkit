/*  tractographyData.h

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */


#ifndef Data_gpu_ptx_H
#define Data_gpu_ptx_H

#include <string>
#include <vector>

#include <probtrackxOptions.h>


struct MaskData{
	float 		distthresh;
	int* 		locs;
	int 		nlocs;
	float* 		volume;
	float* 		vertices;
	int* 		faces;
	int* 		VoxFaces;
	int* 		VoxFacesIndex;
	int* 		IndexRoi;	// index with the id of the roi: first the id of all the volumes, then the id of all the surfaces
	int 		NVols;		// 0 or 1 for mixed data
	int 		NSurfs;		// 0 or 1 for mixed data
	int* 		sizesStr;	// size structures loaded: [0] locs, [1] vertices, [2] faces, [3] voxFaces
};

/** \brief This class contain all necessary data to perform Tractography.
 * All the attributes are basic types in order to copy an object to the GPU*/
class tractographyData{

	public:

	int		nvoxels;
	int		nsamples;
	int		nfibres;
	size_t nseeds;
	int		nparticles;
	int		nsteps;

	float		steplength;
	float		distthresh;	// rejects paths shorter than this threshold (in mm)
	float		curv_thr;	// curvature Threshold
	float		fibthresh;	// anisotropy fraction Threshold
	float		sampvox;	// start with an offset of x random mm inside the voxel seed
	int			fibst;		// force a starting fibre for tracking
	bool		usef;		// use anisotropy to constrain tracking

	float*		Sdims;		// seed space dimensions
	float*		Ddims;		// diffusion space dimensions
	int*		Ssizes;		// seed space volume sizes
	int*		Dsizes;		// diffusion space volume sizes
	int*		M2sizes;	// Matrix2 volume sizes

	float*		seeds;
	std::vector<float>	seeds_vertices;	// for --os2t, store vertices if any mesh in the seeds list
	std::vector<int>	seeds_faces;	// for --os2t, store faces if any mesh in the seeds list
	std::vector<int>	seeds_act;	// for --os2t, sknow if activated or not
	std::vector<int>	seeds_mesh_info;// for --os2t and mesh. List with 2 numbers per mesh: number of vertices, number of faces
	int*		seeds_ROI;	// for network mode. Array with as many elms as seeds. Seed->ROI (sorted as text input file)

	float*		mask;
	float*		thsamples;
	float*		phsamples;
	float*		fsamples;
	int*		lut_vol2mat;
	// what is the position of a voxel?
	// lu_vol2mat tell us the index of the voxel inside an array,
	// array does not include data outside the mask, so cannot get voxel with coordinates xyz
	// diffusion space ... using mask

	float*		vox2mm;		// for surfaces

	float*		Seeds_to_DTI;	// precomputed some operations
	float*		DTI_to_Seeds;	// precomputed some operations
	float*		Seeds_to_M2;	// for Matrix2...precomputed some operations

	// non-linear transformations data (seed-diffusion spaces)
	bool		IsNonlinXfm;
	float*		SeedDTIwarp;
	float*		DTISeedwarp;
	int*		Warp_S2D_sizes;
	int*		Warp_D2S_sizes;
	float*		SsamplingI;
	float*		DsamplingI;
	float*		Wsampling_S2D_I;
	float*		Wsampling_D2S_I;

	bool		forcefirststep;

	MaskData	avoid;
	MaskData	stop;
	MaskData	wtstop;

	// waypoints
	bool		oneway;			// apply waypoint conditions to each half tract separately? default false
	bool		waycond;		// 0=OR 1=AND, default 1
	bool		wayorder;		// reject streamlines that do not hit waypoints in given order. Only valid if waycond=AND. 0=NO ORDER, 1=ORDER, default 0
	MaskData	waypoint;
	MaskData	targets;		// --os2t
	std::vector<std::string>	targetnames;// need it to write the files
	MaskData	targetsREF;		// --os2t reference
	// when there are a lot of ROIs,
	// the tool is not very efficient as it needs to check all of them every step.
	// So, I use a general MaskData as reference that adds all ROIs.
	// The methood first checks this mask, and only if it is crossed, checks the rest.

	MaskData	lrmatrix1;		// columns of Matrix 1. Used also in Matrix2

	// Info of the rows of Matrix 1:
	int*		matrix1_locs;		// locs of each seed in the Matrix. Only CPU
	int*		matrix1_idTri;		// Triangle of each element of matrix1. Only CPU
	// Value is always 0 because distance at seed point is 0
	int*		matrix1_Ntri;		// triangles of each seed: max 12 per vertex ? . Only CPU  !!!  WHY ??? ...correct this.

	MaskData	matrix3;
	MaskData	lrmatrix3;

	MaskData	network;
	MaskData    networkREF;
	// when there are a lot of ROIs in Network mode,
	// the tool is not very efficient as it needs to check all of them every step.
	// So, I use a general MaskData as reference that adds all ROIs.
	// The methood first checks this mask, and only if it is crossed, checks the rest.

	public:

	tractographyData();
};

#endif
