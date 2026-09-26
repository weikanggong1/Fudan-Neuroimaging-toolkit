/*  samples.cu

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>

#include "utils/log.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "xfibresoptions.h"
#include "samples.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
//using namespace MISCMATHS;
using namespace Xfibres;

////////////////////////////////////////////
//       MCMC SAMPLE STORAGE
////////////////////////////////////////////

Samples::Samples(int nvoxels,int nsamples):
opts(xfibresOptions::getInstance()){

	/////////////// GPU version /////////////////////
    	m_sum_d=new float[nvoxels];
    	m_sum_S0=new float[nvoxels];
    	for(int i=0;i<nvoxels;i++){
    		m_sum_d[i]=0;
     		m_sum_S0[i]=0;
    	}
    	m_vec=new ColumnVector[nvoxels];
    	m_dyad=new vector<SymmetricMatrix>[nvoxels];
    	m_sum_f=new vector<float> [nvoxels];
    	m_sum_lam=new vector<float> [nvoxels];
    	////////////////////////////////////////////////

    	m_dsamples.ReSize(nsamples,nvoxels);
    	m_dsamples=0;
    	m_S0samples.ReSize(nsamples,nvoxels);
    	m_S0samples=0;

    	m_mean_dsamples.ReSize(nvoxels);
    	m_mean_dsamples=0;
    	m_mean_S0samples.ReSize(nvoxels);
    	m_mean_S0samples=0;
    	Matrix tmpvecs(3,nvoxels);
    	tmpvecs=0;
    	//m_sum_d=0;  changed GPU version
    	//m_sum_S0=0;  changed GPU version

    	if(opts.modelnum.value()>=2){
      		m_d_stdsamples.ReSize(nsamples,nvoxels);
      		m_d_stdsamples=0;
      		m_mean_d_stdsamples.ReSize(nvoxels);
      		m_mean_d_stdsamples=0;
      		//m_sum_d_std=0;  changed GPU version

      		/////////////// GPU version /////////////////////
      		m_sum_d_std=new float[nvoxels];
      		for(int i=0;i<nvoxels;i++){
      			m_sum_d_std[i]=0;
      		}
      		////////////////////////////////////////////////
		if(opts.modelnum.value()==3){
			m_Rsamples.ReSize(nsamples,nvoxels);
			m_Rsamples=0;
			m_mean_Rsamples.ReSize(nvoxels);
			m_mean_Rsamples=0;
			m_sum_R=new float[nvoxels];
			for(int i=0;i<nvoxels;i++){
      				m_sum_R[i]=0;
      			}
		}
    	}

    	if (opts.f0.value()){
      		m_f0samples.ReSize(nsamples,nvoxels);
      		m_f0samples=0;
      		m_mean_f0samples.ReSize(nvoxels);
      		m_mean_f0samples=0;
      		//m_sum_f0=0;  changed GPU version

     	 	/////////////// GPU version /////////////////////
      		m_sum_f0=new float[nvoxels];
      		for(int i=0;i<nvoxels;i++)
      			m_sum_f0[i]=0;
      		////////////////////////////////////////////////
    	}

    	if (opts.rician.value()){
      		m_mean_tausamples.ReSize(nvoxels);
      		m_mean_tausamples=0;
      		//m_sum_tau=0;  changed GPU version

      		/////////////// GPU version /////////////////////
      		m_sum_tau=new float[nvoxels];
      		for(int i=0;i<nvoxels;i++)
      			m_sum_tau[i]=0;
      		////////////////////////////////////////////////
    	}

    	SymmetricMatrix tmpdyad(3);
    	tmpdyad=0;
    	m_nsamps=nsamples;
    	//m_vec.ReSize(3);  changed GPU version

    	/////////////// GPU version /////////////////////
    	for(int i=0;i<nvoxels;i++){
        	m_vec[i].ReSize(3);
		for(int f=0;f<opts.nfibres.value();f++){
			m_dyad[i].push_back(tmpdyad);
                	m_sum_f[i].push_back(0);
                	m_sum_lam[i].push_back(0);
        	}
    	}
    	////////////////////////////////////////////////

    	for(int f=0;f<opts.nfibres.value();f++){
      		m_thsamples.push_back(m_S0samples);
      		m_phsamples.push_back(m_S0samples);
      		m_fsamples.push_back(m_S0samples);
      		m_lamsamples.push_back(m_S0samples);

      		m_dyadic_vectors.push_back(tmpvecs);
      		m_mean_fsamples.push_back(m_mean_S0samples);
      		m_mean_lamsamples.push_back(m_mean_S0samples);

      		//m_sum_lam.push_back(0);  changed GPU version
      		//m_sum_f.push_back(0);  changed GPU version
      		//m_dyad.push_back(tmpdyad);  changed GPU version
    	}
}

	//new version for GPU
void Samples::record(float rd,float rf0,float rtau,float rdstd,float rR,float rs0,float *rth,float *rph, float *rf, int vox, int samp){
    	m_dsamples(samp,vox)=rd;
    	m_sum_d[vox-1]+=rd;

    	if(opts.modelnum.value()>=2){
		m_d_stdsamples(samp,vox)=rdstd;
      		m_sum_d_std[vox-1]+=rdstd;
    	}
	if(opts.modelnum.value()==3){
		m_Rsamples(samp,vox)=rR;
		m_sum_R[vox-1]+=rR;
    	}
    	if (opts.f0.value()){
     		m_f0samples(samp,vox)=rf0;
      		m_sum_f0[vox-1]+=rf0;
    	}
    	if (opts.rician.value()){
      		m_sum_tau[vox-1]+=rtau;
    	}

    	m_S0samples(samp,vox)=rs0;
    	m_sum_S0[vox-1]+=rs0;

    	for(int f=0;f<opts.nfibres.value();f++){
      		float th=rth[f];
      		float ph=rph[f];
      		m_thsamples[f](samp,vox)=th;
      		m_phsamples[f](samp,vox)=ph;
      		m_fsamples[f](samp,vox)=rf[f];
     	 	//for means
      		m_vec[vox-1] << sin(th)*cos(ph) << sin(th)*sin(ph)<<cos(th) ;

      		m_dyad[vox-1][f] << m_dyad[vox-1][f]+m_vec[vox-1]*m_vec[vox-1].t();
      		m_sum_f[vox-1][f]+=rf[f];
      		m_sum_lam[vox-1][f]+=0;
    	}
}

//new version for GPU
 void Samples::finish_voxel(int vox){
    	m_mean_dsamples(vox)=m_sum_d[vox-1]/m_nsamps;

    	if(opts.modelnum.value()>=2)
      		m_mean_d_stdsamples(vox)=m_sum_d_std[vox-1]/m_nsamps;
	if(opts.modelnum.value()==3)
      		m_mean_Rsamples(vox)=m_sum_R[vox-1]/m_nsamps;
    	if(opts.f0.value())
      		m_mean_f0samples(vox)=m_sum_f0[vox-1]/m_nsamps;
    	if(opts.rician.value())
      		m_mean_tausamples(vox)=m_sum_tau[vox-1]/m_nsamps;

    	m_mean_S0samples(vox)=m_sum_S0[vox-1]/m_nsamps;

    	m_sum_d[vox-1]=0;
    	m_sum_S0[vox-1]=0;

    	if(opts.rician.value())
    		m_sum_tau[vox-1]=0;

    	if(opts.modelnum.value()>=2)
      		m_sum_d_std[vox-1]=0;
	if(opts.modelnum.value()==3)
		m_sum_R[vox-1]=0;
    	if (opts.f0.value())
      		m_sum_f0[vox-1]=0;

    	DiagonalMatrix dyad_D; //eigenvalues
    	Matrix dyad_V; //eigenvectors
    	int nfibs=0;
    	for(int f=0;f<opts.nfibres.value();f++){
      		EigenValues(m_dyad[vox-1][f],dyad_D,dyad_V);
      		int maxeig;
      		if(dyad_D(1)>dyad_D(2)){
			if(dyad_D(1)>dyad_D(3)) maxeig=1;
			else maxeig=3;
      		}
      		else{
			if(dyad_D(2)>dyad_D(3)) maxeig=2;
			else maxeig=3;
      		}
      		m_dyadic_vectors[f](1,vox)=dyad_V(1,maxeig);
      		m_dyadic_vectors[f](2,vox)=dyad_V(2,maxeig);
      		m_dyadic_vectors[f](3,vox)=dyad_V(3,maxeig);

      		if((m_sum_f[vox-1][f]/m_nsamps)>0.05){
			nfibs++;
      		}
      		m_mean_fsamples[f](vox)=m_sum_f[vox-1][f]/m_nsamps;
      		m_mean_lamsamples[f](vox)=m_sum_lam[vox-1][f]/m_nsamps;

      		m_dyad[vox-1][f]=0;
      		m_sum_f[vox-1][f]=0;
      		m_sum_lam[vox-1][f]=0;
    	}
    	//m_beenhere(int(m_matrix2volkey(vox,1)),int(m_matrix2volkey(vox,2)),int(m_matrix2volkey(vox,3)))=nfibs;
}

void save_part(RowVector data, string name, int idpart){
	Log& logger = LogSingleton::getInstance();
    	xfibresOptions& opts = xfibresOptions::getInstance();

	int nvox = data.Ncols();

	string file_name;

	file_name = logger.appendDir(name+"_"+MISCMATHS::num2str(idpart));
	ofstream out;
	out.open(file_name.data(), ios::out | ios::binary);
	out.write((char*)&data(1),nvox*sizeof(Real));
	out.close();
}

void save_part(Matrix data, string name, int idpart){
	Log& logger = LogSingleton::getInstance();
    	xfibresOptions& opts = xfibresOptions::getInstance();

	int nvox = data.Ncols();
	int nsamples = data.Nrows();

	string file_name;

	file_name = logger.appendDir(name+"_"+MISCMATHS::num2str(idpart));
	ofstream out;
	out.open(file_name.data(), ios::out | ios::binary);
	out.write((char*)&data(1,1),nvox*nsamples*sizeof(Real));
	out.close();
}

void Samples::save(int idpart){

	vector<Matrix> thsamples_out=m_thsamples;
	vector<Matrix> phsamples_out=m_phsamples;
	vector<Matrix> fsamples_out=m_fsamples;
	vector<Matrix> lamsamples_out=m_lamsamples;

    	vector<Matrix> dyadic_vectors_out=m_dyadic_vectors;
    	vector<Matrix> mean_fsamples_out;
    	for(unsigned int f=0;f<m_mean_fsamples.size();f++)
      		mean_fsamples_out.push_back(m_mean_fsamples[f]);

    	Log& logger = LogSingleton::getInstance();
    	if(opts.modelnum.value()==1){
		save_part(m_mean_dsamples,"mean_dsamples",idpart);
    	}
    	else if(opts.modelnum.value()>=2){
		save_part(m_mean_dsamples,"mean_dsamples",idpart);
		save_part(m_mean_d_stdsamples,"mean_d_stdsamples",idpart);
		//save_part(m_dsamples,"m_d_stdsamples",idpart);
		//save_part(m_d_stdsamples,"d_stdsamples",idpart);
		if(opts.modelnum.value()==3){
			save_part(m_mean_Rsamples,"mean_Rsamples",idpart);
		}
    	}
    	if (opts.f0.value()){
		save_part(m_mean_f0samples,"mean_f0samples",idpart);
		//save_part(m_f0samples,"f0samples",idpart);
    	}
    	if (opts.rician.value()){
		save_part(m_mean_tausamples,"mean_tausamples",idpart);
    	}

	save_part(m_mean_S0samples,"mean_S0samples",idpart);

    	//Sort the output based on mean_fsamples
    	//
    	vector<Matrix> sumf;
    	for(int f=0;f<opts.nfibres.value();f++){
      		Matrix tmp=MISCMATHS::sum(m_fsamples[f],1);
      		sumf.push_back(tmp);
    	}
    	for(int vox=1;vox<=m_dsamples.Ncols();vox++){
      		vector<pair<float,int> > sfs;
      		pair<float,int> ftmp;

      		for(int f=0;f<opts.nfibres.value();f++){
			ftmp.first=sumf[f](1,vox);
			ftmp.second=f;
			sfs.push_back(ftmp);
      		}
      		sort(sfs.begin(),sfs.end());

      		for(int samp=1;samp<=m_dsamples.Nrows();samp++){
			for(int f=0;f<opts.nfibres.value();f++){;
	  			thsamples_out[f](samp,vox)=m_thsamples[sfs[(sfs.size()-1)-f].second](samp,vox);
	  			phsamples_out[f](samp,vox)=m_phsamples[sfs[(sfs.size()-1)-f].second](samp,vox);
	  			fsamples_out[f](samp,vox)=m_fsamples[sfs[(sfs.size()-1)-f].second](samp,vox);
	  			lamsamples_out[f](samp,vox)=m_lamsamples[sfs[(sfs.size()-1)-f].second](samp,vox);
			}
      		}

      		for(int f=0;f<opts.nfibres.value();f++){
			mean_fsamples_out[f](1,vox)=m_mean_fsamples[sfs[(sfs.size()-1)-f].second](vox);
			dyadic_vectors_out[f](1,vox)=m_dyadic_vectors[sfs[(sfs.size()-1)-f].second](1,vox);
			dyadic_vectors_out[f](2,vox)=m_dyadic_vectors[sfs[(sfs.size()-1)-f].second](2,vox);
			dyadic_vectors_out[f](3,vox)=m_dyadic_vectors[sfs[(sfs.size()-1)-f].second](3,vox);
      		}
    	}
    	// save the sorted fibres
    	for(int f=0;f<opts.nfibres.value();f++){
      		//      element_mod_n(thsamples_out[f],M_PI);
      		//      element_mod_n(phsamples_out[f],2*M_PI);

		save_part(thsamples_out[f],"th"+MISCMATHS::num2str(f+1)+"samples",idpart);

		save_part(phsamples_out[f],"ph"+MISCMATHS::num2str(f+1)+"samples",idpart);

		save_part(fsamples_out[f],"f"+MISCMATHS::num2str(f+1)+"samples",idpart);

		//save_part(mean_fsamples_out[f],"mean_f"+MISCMATHS::num2str(f+1)+"samples",idpart);
		//save_part(dyadic_vectors_out[f],"dyads"+MISCMATHS::num2str(f+1),idpart);


    	}
}
