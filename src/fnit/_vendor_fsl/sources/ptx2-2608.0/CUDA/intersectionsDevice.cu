/*  intersectionsDevice.cu

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <CUDA/options/options.h>

__constant__ float C_vox2mm[12];

__device__ inline bool equal_coord(	const float3 	p1,
					const float3 	p2)
{
  return ((fabsf(p1.x-p2.x)<PRECISION_SAME_VERTEX) && (fabsf(p1.y-p2.y)<PRECISION_SAME_VERTEX) && (fabsf(p1.z-p2.z)<PRECISION_SAME_VERTEX));
}

__device__ inline float norm_point(float3 p)	// 3 coordinates
{
  return (sqrtf(p.x*p.x + p.y*p.y + p.z*p.z));
}

__device__ inline bool has_crossed_volume(	const float*	roivol, 
						float*		posx,	// shared ... segment coordinates
						float*		posy,
						float*		posz)
{
  int3 posI;
  posI.x=rintf(posx[0]);
  posI.y=rintf(posy[0]);
  posI.z=rintf(posz[0]);
  if(posI.x<0||posI.y<0||posI.z<0||posI.x>(C_Ssizes[0]-1)||posI.y>(C_Ssizes[1]-1)||posI.z>(C_Ssizes[2]-1)) return false;
  if(roivol[posI.z*C_Ssizes[1]*C_Ssizes[0]+posI.y*C_Ssizes[0]+posI.x]!=-1)
    return true;
  else
    return false;
}

// It writes in loc the id of the crossed voxel
template <bool M2>	// Matrix2
__device__ inline bool has_crossed_volume_loc(	const float*	roivol, 
						float*		posx,	//shared  ... segment.x
						float*		posy,	//shared  ... segment.y
						float*		posz,	//shared  ... segment.z
						int&		loc)
{
  int3 posI;
  posI.x=rintf(posx[0]);
  posI.y=rintf(posy[0]);
  posI.z=rintf(posz[0]);
  if(M2){
    if(posI.x<0||posI.y<0||posI.z<0||posI.x>(C_M2sizes[0]-1)||posI.y>(C_M2sizes[1]-1)||posI.z>(C_M2sizes[2]-1)) return false;
    loc=roivol[posI.z*C_M2sizes[0]*C_M2sizes[1]+posI.y*C_M2sizes[0]+posI.x];
  }else{
    if(posI.x<0||posI.y<0||posI.z<0||posI.x>(C_Ssizes[0]-1)||posI.y>(C_Ssizes[1]-1)||posI.z>(C_Ssizes[2]-1)) return false;
    loc=roivol[posI.z*C_Ssizes[0]*C_Ssizes[1]+posI.y*C_Ssizes[0]+posI.x];
  }
  if(loc!=-1)
    return true;
  else
    return false;
}

__device__ inline bool triangle_intersect(	const float3 	triangleA,
						const float3 	triangleB,
						const float3	triangleC,	
						const float3	segmentA,
						const float3	segmentB)		
{
  // check if point is one the vertices
  if(equal_coord(triangleA,segmentA))return true;
  if(equal_coord(triangleA,segmentB))return true;
  if(equal_coord(triangleB,segmentA))return true;
  if(equal_coord(triangleB,segmentB))return true;
  if(equal_coord(triangleC,segmentA))return true;
  if(equal_coord(triangleC,segmentB))return true;

  // get triangle edge vectors and plane normal
  //u = _vertice[1]-_vertice[0];
  float3 u;
  u.x= (triangleB.x-triangleA.x); 
  u.y= (triangleB.y-triangleA.y); 
  u.z= (triangleB.z-triangleA.z); 
    	
  //v = _vertice[2]-_vertice[0];
  float3 v;
  v.x= (triangleC.x-triangleA.x);
  v.y= (triangleC.y-triangleA.y);  
  v.z= (triangleC.z-triangleA.z);  

  //n = u*v; // cross product
  float3 n;
  n.x= u.y * v.z - u.z * v.y;
  n.y= v.x * u.z - v.z * u.x;
  n.z= u.x * v.y - v.x * u.y;

  float n_norm=norm_point(n);

  if(n_norm==0.0f) // triangle is degenerate
    return false;  
  //dir = p[1]-p[0];             // ray direction vector
  float3 dir;
  dir.x=segmentB.x-segmentA.x;
  dir.y=segmentB.y-segmentA.y;
  dir.z=segmentB.z-segmentA.z;
  //w0 = p[0]-_vertice[0];
  float3 w0;
  w0.x=segmentA.x-triangleA.x;
  w0.y=segmentA.y-triangleA.y;
  w0.z=segmentA.z-triangleA.z;
	
  //a = -(n|w0);
  float a= -(n.x*w0.x+ n.y*w0.y+ n.z*w0.z);
  //b = (n|dir);	
  float b= (n.x*dir.x+ n.y*dir.y+ n.z*dir.z);
   
  if (fabsf(b) < 0.00000001f) { 		// ray is parallel to triangle plane
    if (fabsf(a) < 0.00000001f)  	// ray lies in triangle plane
      return true;
    else return false;             	// ray disjoint from plane
  }

  // get intersect point of ray with triangle plane
  float r = a/ b;
  if (r < 0.0f)                   // ray goes away from triangle
    return false;                  // => no intersect

  // for a segment, also test if (r > 1.0) => no intersect
  if (r > 1.0f)
    return false;

  float3 I;
  //I = p[0] + r * dir;           // intersect point of ray and plane
  I.x= segmentA.x + r*dir.x;
  I.y= segmentA.y + r*dir.y;
  I.z= segmentA.z + r*dir.z;    

  // is I inside T?
  //double uu,uv,vv,wu,wv,D;
  //uu = (u|u);
  float uu = u.x*u.x+ u.y*u.y+ u.z*u.z;
  //uv = (u|v);
  float uv = u.x*v.x+ u.y*v.y+ u.z*v.z;
  //vv = (v|v);
  float vv = v.x*v.x+ v.y*v.y+ v.z*v.z;
  //w = I - _vertice[0];
  float3 w;
  w.x=I.x - triangleA.x;
  w.y=I.y - triangleA.y;
  w.z=I.z - triangleA.z;
  // wu = (w|u);
  float wu = w.x*u.x+ w.y*u.y+ w.z*u.z;
  //wv = (w|v);
  float wv = w.x*v.x+ w.y*v.y+ w.z*v.z;
  float D = uv * uv - uu * vv;
    
  // get and test parametric coords
  //double s,t;
  float s = (uv * wv - vv * wu) / D;
  if (s < 0.0f || s > 1.0f)        // I is outside T
    return false;
  float t = (uv * wu - uu * wv) / D;
  if (t < 0.0f || (s + t) > 1.0f)  // I is outside T
    return false; 
    
  return true;                      // I is in T
    
}

// return Closest Vertex (0,1,2) OR -1 if not intersected
__device__ inline int triangle_intersect_closest(	const float3 	triangleA,	// vertex 0
							const float3 	triangleB,	// vertex 1
							const float3	triangleC,	// vertex 2
							const float3	segmentA,
							const float3	segmentB)		
{
  // check if point is one the vertices
  if(equal_coord(triangleA,segmentA))return 0;
  if(equal_coord(triangleA,segmentB))return 0;
  if(equal_coord(triangleB,segmentA))return 1;
  if(equal_coord(triangleB,segmentB))return 1;
  if(equal_coord(triangleC,segmentA))return 2;
  if(equal_coord(triangleC,segmentB))return 2;

  // get triangle edge vectors and plane normal
  //u = _vertice[1]-_vertice[0];
  float3 u;
  u.x= (triangleB.x-triangleA.x); 
  u.y= (triangleB.y-triangleA.y); 
  u.z= (triangleB.z-triangleA.z); 
    	
  //v = _vertice[2]-_vertice[0];
  float3 v;
  v.x= (triangleC.x-triangleA.x);
  v.y= (triangleC.y-triangleA.y);  
  v.z= (triangleC.z-triangleA.z);  

  //n = u*v;             // cross product
  float3 n;
  n.x= u.y * v.z - u.z * v.y;
  n.y= v.x * u.z - v.z * u.x;
  n.z= u.x * v.y - v.x * u.y;

  float n_norm=norm_point(n);

  if(n_norm==0.0f) // triangle is degenerate
    return -1;  
  //dir = p[1]-p[0];             // ray direction vector
  float3 dir;
  dir.x=segmentB.x-segmentA.x;
  dir.y=segmentB.y-segmentA.y;
  dir.z=segmentB.z-segmentA.z;
  //w0 = p[0]-_vertice[0];
  float3 w0;
  w0.x=segmentA.x-triangleA.x;
  w0.y=segmentA.y-triangleA.y;
  w0.z=segmentA.z-triangleA.z;
	
  //a = -(n|w0);
  float a= -(n.x*w0.x+ n.y*w0.y+ n.z*w0.z);
  //b = (n|dir)/n.norm()/dir.norm();	
  float b= (n.x*dir.x+ n.y*dir.y+ n.z*dir.z);
   
  if (fabsf(b) < 0.00000001f){ 			// ray is parallel to triangle plane
    if (fabsf(a) < 0.00000001f)           	// ray lies in triangle plane
      return 0;
    else return -1;             		// ray disjoint from plane
  }

  // get intersect point of ray with triangle plane
  float r = a/ b;
  if (r < 0.0f)                   // ray goes away from triangle
    return -1;              // => no intersect

  // for a segment, also test if (r > 1.0) => no intersect
  if (r > 1.0f)
    return -1;

  float3 I;
  //I = p[0] + r * dir;           // intersect point of ray and plane
  I.x= segmentA.x + r*dir.x;
  I.y= segmentA.y + r*dir.y;
  I.z= segmentA.z + r*dir.z;    

  // is I inside T?
  //double uu,uv,vv,wu,wv,D;
  //uu = (u|u);
  float uu = u.x*u.x+ u.y*u.y+ u.z*u.z;
  //uv = (u|v);
  float uv = u.x*v.x+ u.y*v.y+ u.z*v.z;
  //vv = (v|v);
  float vv = v.x*v.x+ v.y*v.y+ v.z*v.z;
  //w = I - _vertice[0];
  float3 w;
  w.x=I.x - triangleA.x;
  w.y=I.y - triangleA.y;
  w.z=I.z - triangleA.z;
  // wu = (w|u);
  float wu = w.x*u.x+ w.y*u.y+ w.z*u.z;
  //wv = (w|v);
  float wv = w.x*v.x+ w.y*v.y+ w.z*v.z;
  float D = uv * uv - uu * vv;
    
  // get and test parametric coords
  //double s,t;
  float s = (uv * wv - vv * wu) / D;
  if (s < 0.0f || s > 1.0f)        // I is outside T
    return -1;
  float t = (uv * wu - uu * wv) / D;
  if (t < 0.0f || (s + t) > 1.0f)  // I is outside T
    return -1; 

  // which vertex is closest to where the segment intersects?
  float x=uu-2*wu;
  float y=vv-2*wv;
  if( x<0 ){
    if( x<y ) return 1;
    else return 2;
  }else{
    if( y<0 ) return 2;
    else return 0;
  }
}

__device__ inline bool rayBoxIntersection(	float* 		originx, 	// shared
						float* 		originy, 	// shared
						float* 		originz, 	// shared
						float3 		invdirection,
						int 		centerx,
						int		centery,
						int		centerz)	
{
  float tmin,tmax;
  float l1 = (centerx-0.5f - originx[0]) * invdirection.x;  
  float l2 = (centerx+0.5f - originx[0]) * invdirection.x;    
  tmin = min(l1, l2);  
  tmax = max(l1, l2);  

  l1 = (centery-0.5f - originy[0]) * invdirection.y;  
  l2 = (centery+0.5f - originy[0]) * invdirection.y;    
  tmin = max(min(l1, l2), tmin);  
  tmax = min(max(l1, l2), tmax);  
  
  l1   = (centerz-0.5f - originz[0]) * invdirection.z;  
  l2   = (centerz+0.5f - originz[0]) * invdirection.z;    
  tmin = max(min(l1, l2), tmin);  
  tmax = min(max(l1, l2), tmax);  
  
  return ((tmax >= tmin) && (tmax >= 0.0f));   
}

__device__ inline bool has_crossed_surface(	const float*		vertices,
						const int*		faces,
						const int*		VoxFaces,
						const int*		VoxFacesIndex,
						float*			segmentAx,  // in seed space
						float*			segmentAy,
						float*			segmentAz,
						float*			segmentBx,  // in seed space
						float*			segmentBy,
						float*			segmentBz)			
{   
  //SEGMENT
  //transform segment voxel coordinates into mm space
  float3 segmentAmm;
  float3 segmentBmm;
  segmentAmm.x=C_vox2mm[0]*segmentAx[0]+C_vox2mm[1]*segmentAy[0]+C_vox2mm[2]*segmentAz[0]+C_vox2mm[3];
  segmentAmm.y=C_vox2mm[4]*segmentAx[0]+C_vox2mm[5]*segmentAy[0]+C_vox2mm[6]*segmentAz[0]+C_vox2mm[7];
  segmentAmm.z=C_vox2mm[8]*segmentAx[0]+C_vox2mm[9]*segmentAy[0]+C_vox2mm[10]*segmentAz[0]+C_vox2mm[11];
  segmentBmm.x=C_vox2mm[0]*segmentBx[0]+C_vox2mm[1]*segmentBy[0]+C_vox2mm[2]*segmentBz[0]+C_vox2mm[3];
  segmentBmm.y=C_vox2mm[4]*segmentBx[0]+C_vox2mm[5]*segmentBy[0]+C_vox2mm[6]*segmentBz[0]+C_vox2mm[7];
  segmentBmm.z=C_vox2mm[8]*segmentBx[0]+C_vox2mm[9]*segmentBy[0]+C_vox2mm[10]*segmentBz[0]+C_vox2mm[11];

  //get voxels crossed
  if((int)rintf(segmentAx[0])==(int)rintf(segmentBx[0])&&
     (int)rintf(segmentAy[0])==(int)rintf(segmentBy[0])&&
     (int)rintf(segmentAz[0])==(int)rintf(segmentBz[0])){

    //only 1 voxel crossed (current position)
    if(segmentAz[0]<0||segmentAy[0]<0||segmentAx[0]<0||segmentAz[0]>(C_Ssizes[2]-1)||segmentAy[0]>(C_Ssizes[1]-1)||segmentAx[0]>(C_Ssizes[0]-1)) 
      return false;
    int pos=(int)rintf(segmentAz[0])*C_Ssizes[1]*C_Ssizes[0]+(int)rintf(segmentAy[0])*C_Ssizes[0]+(int)rintf(segmentAx[0]);
    int ntriangles=VoxFacesIndex[pos+1]-VoxFacesIndex[pos];	
    pos=VoxFacesIndex[pos];
    for(int j=0;j<ntriangles;j++){
      //each triangle of the voxel
      float3 triangleA;
      float3 triangleB;
      float3 triangleC;
      triangleA.x=vertices[faces[VoxFaces[pos+j]]]; 
      triangleA.y=vertices[faces[VoxFaces[pos+j]]+1];
      triangleA.z=vertices[faces[VoxFaces[pos+j]]+2];
      triangleB.x=vertices[faces[VoxFaces[pos+j]+1]];
      triangleB.y=vertices[faces[VoxFaces[pos+j]+1]+1];
      triangleB.z=vertices[faces[VoxFaces[pos+j]+1]+2];
      triangleC.x=vertices[faces[VoxFaces[pos+j]+2]];
      triangleC.y=vertices[faces[VoxFaces[pos+j]+2]+1];
      triangleC.z=vertices[faces[VoxFaces[pos+j]+2]+2];
      //intersect segment with triangle?
      if(triangle_intersect(triangleA,triangleB,triangleC,segmentAmm,segmentBmm))
	return true;
    }
    return false;
  }

  // several voxels crossed
  int x=rintf(segmentAx[0]);
  int y=rintf(segmentAy[0]);
  int z=rintf(segmentAz[0]);
  int3 max;
  max.x=x;
  max.y=y;
  max.z=z;

  int tmp;
  tmp=rintf(segmentBx[0]);
  if(tmp<x){
    x=tmp;
  }else{
    max.x=tmp;
  }
  tmp=rintf(segmentBy[0]);
  if(tmp<y){
    y=tmp;
  }else{
    max.y=tmp;
  }	
  tmp=rintf(segmentBz[0]);
  if(tmp<z){
    z=tmp;
  }else{
    max.z=tmp;
  }	
  float3 invdir;
  invdir.x=1.0f/(segmentBx[0]-segmentAx[0]);
  invdir.y=1.0f/(segmentBy[0]-segmentAy[0]);
  invdir.z=1.0f/(segmentBz[0]-segmentAz[0]);

  for(int ix=x;ix<=max.x;ix+=1){
    for(int iy=y;iy<=max.y;iy+=1){
      for(int iz=z;iz<=max.z;iz+=1){
	if(iz<0||iy<0||ix<0||iz>(C_Ssizes[2]-1)||iy>(C_Ssizes[1]-1)||ix>(C_Ssizes[0]-1)) continue;
	if(rayBoxIntersection(segmentAx,segmentAy,segmentAz,invdir,ix,iy,iz)){
	  int pos=iz*C_Ssizes[1]*C_Ssizes[0]+iy*C_Ssizes[0]+ix;
	  int ntriangles=VoxFacesIndex[pos+1]-VoxFacesIndex[pos];	
	  pos=VoxFacesIndex[pos];
	  for(int j=0;j<ntriangles;j++){
	    //TRIANGLE
	    float3 triangleA;
	    float3 triangleB;
	    float3 triangleC;
	    triangleA.x=vertices[faces[VoxFaces[pos+j]]]; 
	    triangleA.y=vertices[faces[VoxFaces[pos+j]]+1];
	    triangleA.z=vertices[faces[VoxFaces[pos+j]]+2]; 
	    triangleB.x=vertices[faces[VoxFaces[pos+j]+1]];
	    triangleB.y=vertices[faces[VoxFaces[pos+j]+1]+1];
	    triangleB.z=vertices[faces[VoxFaces[pos+j]+1]+2];
	    triangleC.x=vertices[faces[VoxFaces[pos+j]+2]];
	    triangleC.y=vertices[faces[VoxFaces[pos+j]+2]+1];
	    triangleC.z=vertices[faces[VoxFaces[pos+j]+2]+2]; 
	    if(triangle_intersect(triangleA,triangleB,triangleC,segmentAmm,segmentBmm))
	      return true;
	  }
	}	
      }
    }
  }
  return false;
}

// if closestV, only add to the list the closest vertex of each triangle crossed
template <bool M2>	// Matrix2
__device__ inline void has_crossed_surface_loc(	const float*		vertices,
						const int*		matlocs,
						const int*		faces,
						const int*		VoxFaces,
						const int*		VoxFacesIndex,
						float*			segmentAx,		//in seed space, shared
						float*			segmentAy,		//in seed space, shared
						float*			segmentAz,		//in seed space, shared
						float*			segmentBx,		//in seed space, shared
						float*			segmentBy,		//in seed space, shared
						float*			segmentBz,		//in seed space, shared			
						float3*			crossed,
						int&			numcrossed,
						float			value)
{   
  // SEGMENT
  // transform voxel segment coordinates into mm space
  float3 segmentAmm;
  float3 segmentBmm;
  segmentAmm.x=C_vox2mm[0]*segmentAx[0]+C_vox2mm[1]*segmentAy[0]+C_vox2mm[2]*segmentAz[0]+C_vox2mm[3];
  segmentAmm.y=C_vox2mm[4]*segmentAx[0]+C_vox2mm[5]*segmentAy[0]+C_vox2mm[6]*segmentAz[0]+C_vox2mm[7];
  segmentAmm.z=C_vox2mm[8]*segmentAx[0]+C_vox2mm[9]*segmentAy[0]+C_vox2mm[10]*segmentAz[0]+C_vox2mm[11];
  segmentBmm.x=C_vox2mm[0]*segmentBx[0]+C_vox2mm[1]*segmentBy[0]+C_vox2mm[2]*segmentBz[0]+C_vox2mm[3];
  segmentBmm.y=C_vox2mm[4]*segmentBx[0]+C_vox2mm[5]*segmentBy[0]+C_vox2mm[6]*segmentBz[0]+C_vox2mm[7];
  segmentBmm.z=C_vox2mm[8]*segmentBx[0]+C_vox2mm[9]*segmentBy[0]+C_vox2mm[10]*segmentBz[0]+C_vox2mm[11];

  // get voxels crossed
  if((int)rintf(segmentAx[0])==(int)rintf(segmentBx[0])&&
     (int)rintf(segmentAy[0])==(int)rintf(segmentBy[0])&&
     (int)rintf(segmentAz[0])==(int)rintf(segmentBz[0])){

    // only 1 voxel crossed (current position)
    int pos;
    if(M2){
      if(segmentAz[0]<0||segmentAy[0]<0||segmentAx[0]<0||segmentAz[0]>(C_M2sizes[2]-1)||segmentAy[0]>(C_M2sizes[1]-1)||segmentAx[0]>(C_M2sizes[0]-1))
	return;
      pos=(int)rintf(segmentAz[0])*C_M2sizes[1]*C_M2sizes[0]+(int)rintf(segmentAy[0])*C_M2sizes[0]+(int)rintf(segmentAx[0]);
    }else{
      if(segmentAz[0]<0||segmentAy[0]<0||segmentAx[0]<0||segmentAz[0]>(C_Ssizes[2]-1)||segmentAy[0]>(C_Ssizes[1]-1)||segmentAx[0]>(C_Ssizes[0]-1))
	return;
      pos=(int)rintf(segmentAz[0])*C_Ssizes[1]*C_Ssizes[0]+(int)rintf(segmentAy[0])*C_Ssizes[0]+(int)rintf(segmentAx[0]);
    }
    int ntriangles=VoxFacesIndex[pos+1]-VoxFacesIndex[pos];	
    pos=VoxFacesIndex[pos];

    for(int j=0;j<ntriangles;j++){

      // each triangle of the voxel
      float3 triangleA;
      float3 triangleB;
      float3 triangleC;
      triangleA.x=vertices[faces[VoxFaces[pos+j]]]; 
      triangleA.y=vertices[faces[VoxFaces[pos+j]]+1];
      triangleA.z=vertices[faces[VoxFaces[pos+j]]+2];
      triangleB.x=vertices[faces[VoxFaces[pos+j]+1]];
      triangleB.y=vertices[faces[VoxFaces[pos+j]+1]+1];
      triangleB.z=vertices[faces[VoxFaces[pos+j]+1]+2];
      triangleC.x=vertices[faces[VoxFaces[pos+j]+2]];
      triangleC.y=vertices[faces[VoxFaces[pos+j]+2]+1];
      triangleC.z=vertices[faces[VoxFaces[pos+j]+2]+2];

		
      // intersect segment with triangle?
      if(triangle_intersect(triangleA,triangleB,triangleC,segmentAmm,segmentBmm)){					
	int idTri=int(VoxFaces[pos+j]/3);
					
	int loc=matlocs[int(faces[VoxFaces[pos+j]]/3)];
	if(loc>=0){
	  crossed[numcrossed].x=loc;
	  crossed[numcrossed].z=value;
	  crossed[numcrossed].y=idTri;
	  numcrossed++;
	}
	loc=matlocs[int(faces[VoxFaces[pos+j]+1]/3)];
	if(loc>=0){
	  crossed[numcrossed].x=loc;
	  crossed[numcrossed].z=value;
	  crossed[numcrossed].y=idTri;
	  numcrossed++;	
	}
	loc=matlocs[int(faces[VoxFaces[pos+j]+2]/3)];
	if(loc>=0){
	  crossed[numcrossed].x=loc;
	  crossed[numcrossed].z=value;	
	  crossed[numcrossed].y=idTri;
	  numcrossed++;
	}
      }
    }
    return;
  }

  // several voxels crossed
  int x=rintf(segmentAx[0]);
  int y=rintf(segmentAy[0]);
  int z=rintf(segmentAz[0]);
  int3 max;
  max.x=x;
  max.y=y;
  max.z=z;

  int tmp;
  tmp=rintf(segmentBx[0]);
  if(tmp<x){
    x=tmp;
  }else{
    max.x=tmp;
  }
  tmp=rintf(segmentBy[0]);
  if(tmp<y){
    y=tmp;
  }else{
    max.y=tmp;
  }	
  tmp=rintf(segmentBz[0]);
  if(tmp<z){
    z=tmp;
  }else{
    max.z=tmp;
  }	
  float3 invdir;
  invdir.x=1.0f/(segmentBx[0]-segmentAx[0]);
  invdir.y=1.0f/(segmentBy[0]-segmentAy[0]);
  invdir.z=1.0f/(segmentBz[0]-segmentAz[0]);

  for(int ix=x;ix<=max.x;ix+=1){
    for(int iy=y;iy<=max.y;iy+=1){
      for(int iz=z;iz<=max.z;iz+=1){
	if(M2){
	  if(iz<0||iy<0||ix<0||iz>(C_M2sizes[2]-1)||iy>(C_M2sizes[1]-1)||ix>(C_M2sizes[0]-1)) continue;
	}else{
	  if(iz<0||iy<0||ix<0||iz>(C_Ssizes[2]-1)||iy>(C_Ssizes[1]-1)||ix>(C_Ssizes[0]-1)) continue;
	}				
	if(rayBoxIntersection(segmentAx,segmentAy,segmentAz,invdir,ix,iy,iz)){
	  int pos;
	  if(M2){
	    pos=iz*C_M2sizes[1]*C_M2sizes[0]+iy*C_M2sizes[0]+ix;
	  }else{
	    pos=iz*C_Ssizes[1]*C_Ssizes[0]+iy*C_Ssizes[0]+ix;
	  }
	  int ntriangles=VoxFacesIndex[pos+1]-VoxFacesIndex[pos];	
	  pos=VoxFacesIndex[pos];

	  for(int j=0;j<ntriangles;j++){

	    // TRIANGLE
	    float3 triangleA;
	    float3 triangleB;
	    float3 triangleC;
	    triangleA.x=vertices[faces[VoxFaces[pos+j]]];
	    triangleA.y=vertices[faces[VoxFaces[pos+j]]+1];
	    triangleA.z=vertices[faces[VoxFaces[pos+j]]+2];
	    triangleB.x=vertices[faces[VoxFaces[pos+j]+1]];
	    triangleB.y=vertices[faces[VoxFaces[pos+j]+1]+1];
	    triangleB.z=vertices[faces[VoxFaces[pos+j]+1]+2];
	    triangleC.x=vertices[faces[VoxFaces[pos+j]+2]];
	    triangleC.y=vertices[faces[VoxFaces[pos+j]+2]+1];
	    triangleC.z=vertices[faces[VoxFaces[pos+j]+2]+2];

	    if(triangle_intersect(triangleA,triangleB,triangleC,segmentAmm,segmentBmm)){
	      int idTri=int(VoxFaces[pos+j]/3);
			
	      int loc=matlocs[int(faces[VoxFaces[pos+j]]/3)];
	      if(loc>=0){
		crossed[numcrossed].x=loc;
		crossed[numcrossed].y=idTri;
		crossed[numcrossed].z=value;
		numcrossed++;
	      }
	      loc=matlocs[int(faces[VoxFaces[pos+j]+1]/3)];
	      if(loc>=0){
		crossed[numcrossed].x=loc;
		crossed[numcrossed].y=idTri;
		crossed[numcrossed].z=value;
		numcrossed++;
	      }
	      loc=matlocs[int(faces[VoxFaces[pos+j]+2]/3)];
	      if(loc>=0){
		crossed[numcrossed].x=loc;
		crossed[numcrossed].y=idTri;	
		crossed[numcrossed].z=value;
		numcrossed++;
	      }
	    }
	  }
	}	
      }
    }
  }
}

