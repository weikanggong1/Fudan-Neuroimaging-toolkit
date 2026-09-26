/*  samples.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "xfibresoptions.h"


////////////////////////////////////////////
//       MCMC SAMPLE STORAGE
////////////////////////////////////////////

class Samples{
    Xfibres::xfibresOptions& opts;
  	NEWMAT::Matrix m_dsamples;
  	NEWMAT::Matrix m_d_stdsamples;
	NEWMAT::Matrix m_Rsamples;
  	NEWMAT::Matrix m_S0samples;
  	NEWMAT::Matrix m_f0samples;

	//   // storing signal
	//   NEWMAT::Matrix m_mean_sig;
	//   NEWMAT::Matrix m_std_sig;
	//   NEWMAT::Matrix m_sig2;

  	std::vector<NEWMAT::Matrix> m_thsamples;
  	std::vector<NEWMAT::Matrix> m_phsamples;
  	std::vector<NEWMAT::Matrix> m_fsamples;
  	std::vector<NEWMAT::Matrix> m_lamsamples;

  	//for storing means
  	NEWMAT::RowVector m_mean_dsamples;
  	NEWMAT::RowVector m_mean_d_stdsamples;
	NEWMAT::RowVector m_mean_Rsamples;
  	NEWMAT::RowVector m_mean_S0samples;
  	NEWMAT::RowVector m_mean_f0samples;
  	NEWMAT::RowVector m_mean_tausamples;
  	std::vector<NEWMAT::Matrix> m_dyadic_vectors;
  	std::vector<NEWMAT::RowVector> m_mean_fsamples;
  	std::vector<NEWMAT::RowVector> m_mean_lamsamples;

  	//float m_sum_d;  changed GPU version
  	//float m_sum_d_std;  changed GPU version
  	//float m_sum_S0;  changed GPU version
  	//float m_sum_f0;  changed GPU version
  	//float m_sum_tau;  changed GPU version
  	//std::vector<SymmetricMatrix> m_dyad;  changed GPU version
  	//std::vector<float> m_sum_f;  changed GPU version
  	//std::vector<float> m_sum_lam;  changed GPU version
  	//ColumnVector m_vec;  changed GPU version

  	/////////////// GPU version /////////////////////
  	float *m_sum_d;
  	float *m_sum_S0;
  	float *m_sum_d_std;
	float *m_sum_R;
  	float *m_sum_f0;
  	float *m_sum_tau;

    std::vector<NEWMAT::SymmetricMatrix> *m_dyad;
  	std::vector<float>  *m_sum_f;
  	std::vector<float> *m_sum_lam;
  	NEWMAT::ColumnVector *m_vec;
  	////////////////////////////////////////////////

  	int m_nsamps;

	public:

  	Samples(int nvoxels,int nmeasures);

	void record(float rd,float rf0,float rtau,float rdstd,float rR,float rs0,float *rth,float *rph, float *rf, int vox, int samp);

  	void finish_voxel(int vox);

  	void save(int idpart);
};
