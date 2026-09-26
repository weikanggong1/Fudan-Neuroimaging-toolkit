/*    Copyright (C) 2012 University of Oxford  */

/*  CCOPYRIGHT  */
#include <fstream>
#include <string>
#include <vector>

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif
#include "utils/log.h"
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/SpMat.h"
#include "meshclass/meshclass.h"
#include "probtrackxOptions.h"
#include "particle.h"
#include "tractvolsx.h"



namespace TRACT{
  void read_masks(std::vector<std::string>& masks, const std::string& filename);


  class Streamliner{
    //Everything in DTI space is done INSIDE this class and lower level classes (particle and tractvolsx)
    //This class communicates with higher level classes in Seed voxels.
    //
    probtrackxOptions& opts;
    Utilities::Log& logger;
    PARTICLE::Particle m_part;
    std::vector<NEWMAT::ColumnVector> m_path;
    NEWIMAGE::volume<int> m_mask;


    NEWIMAGE::volume<int> m_skipmask;
    NEWIMAGE::volume<int> m_rubbish;
    NEWIMAGE::volume<int> m_stop;
    NEWIMAGE::volume4D<float> m_prefdir;
    NEWIMAGE::volume4D<float> m_loopcheck;
    std::vector<NEWIMAGE::volume<float>* > m_waymasks;
    std::vector<bool> m_passed_flags;
    std::vector<bool> m_own_waymasks;

    NEWMAT::Matrix m_Seeds_to_DTI;
    NEWMAT::Matrix m_DTI_to_Seeds;
    NEWIMAGE::volume4D<float> m_Seeds_to_DTI_warp;
    NEWIMAGE::volume4D<float> m_DTI_to_Seeds_warp;
    NEWIMAGE::volume4D<float> m_jacx;
    NEWIMAGE::volume4D<float> m_jacy;
    NEWIMAGE::volume4D<float> m_jacz;
    bool m_IsNonlinXfm;
    NEWMAT::Matrix m_rotdir;

    TRACTVOLSX::Tractvolsx vols;
    float m_lcrat;
    float m_x_s_init;
    float m_y_s_init;
    float m_z_s_init;

    // Streamliner needs to know about matrix3
    NEWIMAGE::volume<int>  m_mask3;
    NEWIMAGE::volume<int>  m_beenhere3;
    std::vector<NEWMAT::ColumnVector> m_inmask3;

    // we need this class to know about seed space
    const NEWIMAGE::volume<float>& m_seeds;

  public:
    //Constructors
    Streamliner(const NEWIMAGE::volume<float>&);
    ~Streamliner(){
      for(unsigned int i=0;i<m_waymasks.size();i++)
	if(m_own_waymasks[i]) delete m_waymasks[i];
    }
    void add_waymask(NEWIMAGE::volume<float>& myway,const bool& ownership=false){
      //make sure that the waymask to add will not be deleted before
      //this streamliner goes out of scope!!
      m_waymasks.push_back(&myway);
      m_own_waymasks.push_back(ownership);
      m_passed_flags.push_back(false);
    }
    void pop_waymasks(){
      NEWIMAGE::volume<float>* tmpptr=m_waymasks[m_waymasks.size()-1];
      m_waymasks.pop_back();
      m_passed_flags.pop_back();
      if(m_own_waymasks[m_own_waymasks.size()-1]){
	delete tmpptr;
      }
      m_own_waymasks.pop_back();

    }
    void clear_waymasks(){
      // clear all waymasks
      for(unsigned int i=0;i<m_waymasks.size();i++)
	pop_waymasks();
    }

    const TRACTVOLSX::Tractvolsx& get_vols() const {return vols;}
    inline int nfibres() const {return vols.nfibres();}
    inline const float get_x_seed() const {return m_x_s_init;}
    inline const float get_y_seed() const {return m_y_s_init;}
    inline const float get_z_seed() const {return m_z_s_init;}
    inline const std::vector<NEWMAT::ColumnVector>& get_path_ref() const{return m_path;}
    inline std::vector<NEWMAT::ColumnVector> get_path() const{return m_path;}
    inline void reset(){
      m_part.reset();
      vols.reset(opts.fibst.value());
      for(unsigned int i=0;i<m_passed_flags.size();i++)
	m_passed_flags[i]=false;
    }
    inline void reverse(){
      m_part.restart_reverse();
    }
    int streamline(const float& x_init,const float& y_init, const float& z_init,const NEWMAT::ColumnVector& dim_seeds,const int& fibst,const NEWMAT::ColumnVector& dir);

    void rotdir(const NEWMAT::ColumnVector& dir,NEWMAT::ColumnVector& rotdir,const float& x,const float& y,const float& z);

    const NEWIMAGE::volume<int>& get_stop()const{return m_stop;}


    // matrix3 methods
    void clear_beenhere3(){
      for(unsigned int i=0;i<m_inmask3.size();i++){
	m_beenhere3((int)MISCMATHS::round(float(m_inmask3[i](1))),
		    (int)MISCMATHS::round(float(m_inmask3[i](2))),
		    (int)MISCMATHS::round(float(m_inmask3[i](3))))=0;
      }
    }
    void                  clear_inmask3(){m_inmask3.clear();}
    std::vector<NEWMAT::ColumnVector>& get_inmask3(){return m_inmask3;}
    NEWIMAGE::volume<int>&          get_mask3(){return m_mask3;}
    NEWIMAGE::volume<int>&          get_beenhere3(){return m_beenhere3;}
  };


  class Counter{
    probtrackxOptions& opts;
    Utilities::Log& logger;
    NEWIMAGE::volume<int> m_prob;
    NEWIMAGE::volume<int> m_beenhere;
    NEWMAT::Matrix m_I;
    std::vector<NEWMAT::ColumnVector> m_path;

    std::vector<NEWMAT::ColumnVector> m_seedcounts;
    NEWMAT::Matrix m_SeedCountMat;
    int    m_SeedRow;
    int    m_numseeds;

    NEWMAT::Matrix m_targetmasks;
    std::vector<std::string> m_targetmasknames;
    std::vector<int> m_targflags;


    NEWIMAGE::volume<int> m_seeds_vol2mat;
    NEWMAT::Matrix      m_seeds_mat2vol;
    NEWIMAGE::volume<int> m_targets_vol2mat;
    NEWMAT::Matrix      m_targets_mat2vol;


    NEWIMAGE::volume<int> m_ConMat;
    NEWIMAGE::volume<int> m_CoordMat;
    int m_Conrow;

    NEWIMAGE::volume<int> m_ConMat2;
    NEWIMAGE::volume<int> m_CoordMat2;
    NEWIMAGE::volume<int> m_CoordMat_tract2;
    NEWIMAGE::volume<int> m_lrmask;
    NEWIMAGE::volume4D<int> m_lookup2;
    NEWIMAGE::volume<int> m_beenhere2;
    int m_Conrow2;
    NEWMAT::ColumnVector m_lrdim;

    //volume<int>  m_ConMat3;
    MISCMATHS::SpMat<int>  *m_ConMat3;    // Use sparse representation to allow for big NxN matrices
    NEWIMAGE::volume<int>  m_Lookup3;
    NEWMAT::Matrix       m_CoordMat3;

    const NEWIMAGE::volume<float>& m_seeds;
    NEWMAT::ColumnVector m_seedsdim;
    Streamliner& m_stline;
    Streamliner& m_nonconst_stline;

    NEWMAT::Matrix m_pathlengths;
    //int m_nsamp;

  public:
    Counter(const NEWIMAGE::volume<float>& seeds,Streamliner& stline,int numseeds):opts(probtrackxOptions::getInstance()),
									 logger(Utilities::LogSingleton::getInstance()),
									 m_numseeds(numseeds),
									 m_seeds(seeds),m_stline(stline),
									 m_nonconst_stline(stline){
      //are they initialised to zero?
      m_beenhere.reinitialize(m_seeds.xsize(),m_seeds.ysize(),m_seeds.zsize());
      m_seedsdim.ReSize(3);
      m_seedsdim << m_seeds.xdim() <<m_seeds.ydim() <<m_seeds.zdim();
      m_I=NEWMAT::IdentityMatrix(4);

    }

    void initialise();

    void initialise_path_dist(){
      m_prob.reinitialize(m_seeds.xsize(),m_seeds.ysize(),m_seeds.zsize());
      NEWIMAGE::copybasicproperties(m_seeds,m_prob);
      m_prob=0;
    }
    void initialise_seedcounts();

    void initialise_matrix1(); //Need to make sure that initialise_path_dist is run first
    void initialise_matrix2();
    void initialise_matrix3();

    void initialise_maskmatrix(){} //not written yet

    void store_path(){ m_path=m_stline.get_path();}
    void append_path(){
      for(unsigned int i=0;i<m_stline.get_path_ref().size();i++)
	m_path.push_back(m_stline.get_path_ref()[i]);
    }
    void clear_path(){ m_path.clear(); };

    void count_streamline();
    void count_seed();
    void clear_streamline();


    void update_pathdist();
    void reset_beenhere();

    void reset_prob(){m_prob=0;}
    void update_seedcounts();
    void reset_targetflags(){
      for(unsigned int i=0;i<m_targflags.size();i++) m_targflags[i]=0;
    }


    void update_matrix1(); //update path_dist after each streamline, only run this after each voxel!!

    void update_matrix2_row(); //but run this one every streamline as with the others
    void next_matrix2_row(){m_Conrow2++;}//and then run this after each voxel..
    void reset_beenhere2();

    void update_matrix3();
    void reset_beenhere3();

    void update_maskmatrix(){} //not written yet

    void save_total(const int& keeptotal);
    void save_total(const std::vector<int>& keeptotal);
    void save();
    void save_pathdist();
    void save_pathdist(string add);
    void save_seedcounts();
    void save_matrix1();
    void save_matrix2();
    void save_matrix3();
    void save_maskmatrix(){}//not written yet


    inline const Streamliner& get_streamline() const {return m_stline;}
    inline Streamliner& get_nonconst_streamline() const {return m_nonconst_stline;}
    inline const NEWIMAGE::volume<float>& get_seeds() const {return m_seeds;}


  };

  class Seedmanager{
    probtrackxOptions& opts;
    Utilities::Log& logger;
    Counter& m_counter;
    Streamliner& m_stline;
    const NEWIMAGE::volume<float>& m_seeds;
    NEWMAT::ColumnVector m_seeddims;
  public:
    Seedmanager(Counter& counter):opts(probtrackxOptions::getInstance()),
				  logger(Utilities::LogSingleton::getInstance()),
				  m_counter(counter),
				  m_stline(m_counter.get_nonconst_streamline()),
				  m_seeds(m_counter.get_seeds()){
      m_seeddims.ReSize(3);
      m_seeddims<<m_seeds.xdim()<<m_seeds.ydim()<<m_seeds.zdim();
    }
    int run(const float& x,const float& y,const float& z,bool onewayonly=false, int fibst=-1);
    int run(const float& x,const float& y,const float& z,bool onewayonly, int fibst,const NEWMAT::ColumnVector& dir);
  };

}
