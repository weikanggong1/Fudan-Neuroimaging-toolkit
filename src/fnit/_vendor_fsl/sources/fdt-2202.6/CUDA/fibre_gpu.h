/*  fibre_gpu.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __FIBRE_GPU_
#define __FIBRE_GPU_

class FibreGPU{
public:  
    	float m_th;
	float m_th_prop;
	float m_th_prior;
	int m_th_acc; 
    	int m_th_rej;

    	float m_ph;
	float m_ph_prop;
	float m_ph_prior;
	int m_ph_acc;
    	int m_ph_rej; 

    	float m_f;
	float m_f_prop;
	float m_f_prior;
	int m_f_acc;
    	int m_f_rej;

    	//float m_lam;
    	//float m_lam_prop;
    	//float m_lam_prior;
    
    	float m_prior_en;
    	bool m_lam_jump;
    	//float m_d;
};

class MultifibreGPU{
public:    
    
    	float m_f0;
    	float m_f0_prop;
    	float m_f0_prior;
	int m_f0_acc;
    	int m_f0_rej;

    	float m_tau;
    	float m_tau_prop;
    	float m_tau_prior;
	int m_tau_acc;
    	int m_tau_rej;

	float m_S0;
    	float m_S0_prop;
    	float m_S0_prior;
    	int m_S0_acc;
    	int m_S0_rej;

    	float m_d; 			
    	float m_d_prop;
    	float m_d_prior; 
    	int m_d_acc;
    	int m_d_rej;

    	float m_dstd;
    	float m_dstd_prop;
    	float m_dstd_prior;	 
    	int m_dstd_acc;
    	int m_dstd_rej;

	float m_R;
    	float m_R_prop;
    	float m_R_prior;	 
    	int m_R_acc;
    	int m_R_rej;

    	float m_prior_en;		
    	float m_likelihood_en;
    	float m_energy;
};

#endif
