/*  Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __PARTICLE_H_
#define __PARTICLE_H_



//////////////////////////////////////////////////////////////////
//      class Particle                                          //
//            tract particle..                                  //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//                                                              //
//           NB - Everything in this Class is in voxels!!       // 
//                                                              //
//////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

namespace PARTICLE{

  class Particle
    { 
      float m_x;
      float m_y;
      float m_z;
      float m_rx;
      float m_ry;
      float m_rz;
      float m_rx_init;
      float m_ry_init;
      float m_rz_init;
      float m_testx;
      float m_testy;
      float m_testz;
      float m_steplength;
      float m_xdim;
      float m_ydim;
      float m_zdim;
      bool m_has_jumped;
      bool m_simdiff;
      int m_jumpsign;
    public:
      //constructors::
      Particle(const float& xin,const float& yin,
	       const float& zin,const float& rxin,
	       const float& ryin,const float &rzin,
	       const float& steplengthin,
	       const float& xdimin,
	       const float& ydimin,
	       const float& zdimin,
	       const bool& hasjumpedin=false,
	       const bool& simdiffin=false) : 
	m_x(xin), m_y(yin), m_z(zin), m_rx(rxin),m_ry(ryin),m_rz(rzin),m_rx_init(rxin), 
	m_ry_init(ryin),m_rz_init(rzin),m_steplength(steplengthin),
	m_xdim(xdimin),m_ydim(ydimin),m_zdim(zdimin),
	m_has_jumped(hasjumpedin),m_simdiff(false){}
      Particle(){}
      ~Particle(){}
      
      //initialise
      void initialise(const float& xin=0,const float& yin=0,
		      const float& zin=0,const float& rxin=0,
		      const float& ryin=0,const float &rzin=0,
		 const float& steplengthin=0.5,
		 const float& xdimin=2,
		 const float& ydimin=2,
		 const float& zdimin=2,
		 const bool& hasjumpedin=false,
		 const bool& simdiffin=false){
       
	m_x=xin;
	m_y=yin;
	m_z=zin;
	m_rx=rxin; 
	m_ry=ryin;
	m_rz=rzin;
        m_rx_init=rxin;
	m_ry_init=ryin;
	m_rz_init=rzin;
	m_steplength=steplengthin;
	m_xdim=xdimin;
	m_ydim=ydimin;
	m_zdim=zdimin;
	m_has_jumped=hasjumpedin;
	m_simdiff=simdiffin;

      }
      
      
      //return values
      const float& x() const { return m_x; }
      float x() { return m_x; }
      
      const float& y() const { return m_y; }
      float y() { return m_y; }
  
      const float& z() const { return m_z; }
      float z() { return m_z; }
  
      const float& rx() const { return m_rx; }
      float rx() { return m_rx; }
  
      const float& ry() const { return m_ry; }
      float ry() { return m_ry; }
  
      const float& rz() const { return m_rz; }
      float rz() { return m_rz; }

      const float& testx() const { return m_testx; }
      float testx() { return m_testx; }

      const float& testy() const { return m_testy; }
      float testy() { return m_testy; }

      const float& testz() const { return m_testz; }
      float testz() { return m_testz; }
      
      const float& steplength() const { return m_steplength; }
      float steplength() { return m_steplength; }
      
      //change values
      void change_x (float new_x) { m_x=new_x; }
      void change_y (float new_y) { m_y=new_y; }
      void change_z  (float new_z) { m_z=new_z; }
      void change_xyz (float new_x,float new_y,float new_z){
	 m_x=new_x;
	 m_y=new_y;
	 m_z=new_z;
      } 
      void change_steplength (float new_sl) { m_steplength = new_sl; } 
      void reset(){
	m_x=0;m_y=0;m_z=0;m_rx=0;m_ry=0;m_rz=0;m_has_jumped=false;
      }
      //functions
      void jump(const float& theta,const float& phi){
	float rx_new=cos(phi)*sin(theta);
	float ry_new=sin(phi)*sin(theta);
	float rz_new=cos(theta);
	int sign; bool init=false;
	if(!m_simdiff){
	  if(m_has_jumped)
	    {sign=(rx_new*m_rx + ry_new*m_ry + rz_new*m_rz)>0 ? 1:-1;}
	  else{
	    float tmp=rand(); tmp/=RAND_MAX;
	    sign=tmp > 0.5 ? 1:-1;
	    m_jumpsign=sign;
	    m_has_jumped=true;
	    init=true;
	  }
	}
	else{
	  float tmp=rand(); tmp/=RAND_MAX;
	  sign=tmp > 0.5 ? 1:-1;
	}
	m_x += sign*m_steplength/m_xdim*rx_new;
	m_y += sign*m_steplength/m_ydim*ry_new;
	m_z += sign*m_steplength/m_zdim*rz_new;
	m_rx=sign*rx_new; m_ry=sign*ry_new;m_rz=sign*rz_new;
	
	if(init){
	  m_rx_init=m_rx;
	  m_ry_init=m_ry;
	  m_rz_init=m_rz;
	}
	
	
      }
     

      void testjump(const float& theta,const float& phi){
	float rx_new=cos(phi)*sin(theta);
	float ry_new=sin(phi)*sin(theta);
	float rz_new=cos(theta);
	int sign;bool init=false;
	if(!m_simdiff){
	  if(m_has_jumped)
	    {sign=(rx_new*m_rx + ry_new*m_ry + rz_new*m_rz)>0 ? 1:-1;}
	  else{
	    float tmp=rand(); tmp/=RAND_MAX;
	    sign=tmp > 0.5 ? 1:-1;
	    m_jumpsign=sign;
	    // m_has_jumped=true; // causes probtrackx to go in one direction!
	    // init=true;
	  }
	}
	else{
	  float tmp=rand(); tmp/=RAND_MAX;
	  sign=tmp > 0.5 ? 1:-1;
	}
	m_testx = m_x+sign*m_steplength/m_xdim*rx_new;
	m_testy = m_y+sign*m_steplength/m_ydim*ry_new;
	m_testz = m_z+sign*m_steplength/m_zdim*rz_new;
	
	if(init){
	  m_rx_init=m_rx;
	  m_ry_init=m_ry;
	  m_rz_init=m_rz;
	}
	
      }
      
      
      void restart_reverse(){
	if(m_has_jumped){
	  m_rx=-m_rx_init;
	  m_ry=-m_ry_init;
	  m_rz=-m_rz_init;
	}
	
      }

      void set_dir(const float& rx,const float& ry,const float& rz){
	m_rx=rx;m_ry=ry;m_rz=rz;m_has_jumped=true;
      }
      
      
      bool check_dir(const float& theta,const float& phi, const float& thr){
	if(m_has_jumped){
	  float rx_new=cos(phi)*sin(theta);
	  float ry_new=sin(phi)*sin(theta);
	  float rz_new=cos(theta);
	  return fabs(rx_new*m_rx + ry_new*m_ry + rz_new*m_rz)>thr;
	}
	else return true;
      }

      // function added by Saad to choose a direction during deterministic streamlining
      // the choosed direction is the one that is closest to current direction
      unsigned int choose_dir(const vector<float>& th,const vector<float>& ph){
	float ps,tmpps=0;
	unsigned int r=0;

	for(unsigned int i=0;i<th.size();i++){
	  ps=tmpps;
	  tmpps=fabs((sin(th[i])*(cos(ph[i])*m_rx+sin(ph[i])*m_ry)+cos(th[i])*m_rz));
	  r = tmpps > ps ? i : r;
	}
	
	return r;
	
      }


      friend ostream& operator<<(ostream& ostr,const Particle& p);
  

    };

  //overload <<
  inline ostream& operator<<(ostream& ostr,const Particle& p){
    ostr<<p.m_x<<" "<<p.m_y<<" "<<p.m_z<<endl;
    return ostr;
  }

  


}

#endif











