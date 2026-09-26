/*  xfibres_gpu.cc

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "utils/log.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

#include "CUDA/xfibres_gpu.h"
#include "xfibresoptions.h"

#include "CUDA/init_gpu.h"
#include "CUDA/options/options.h"


using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace Xfibres;

//////////////////////////////////////////////////////////
//       XFIBRES CPU PART. IT CALLS TO GPU PART
//////////////////////////////////////////////////////////
// last 4 parameters are subjdir, idPart, nParts, num_total_voxels

int main(int argc, char *argv[]){

	struct timeval t1,t2;
	double time;
    	gettimeofday(&t1,NULL);

	init_gpu();

	// Setup logging:
    	Log& logger = LogSingleton::getInstance();
    	xfibresOptions& opts = xfibresOptions::getInstance();
	opts.parse_command_line(argc-4,argv,logger);
	srand(opts.seed.value());  //randoms seed

	Matrix bvals,bvecs;
    	bvals=read_ascii_matrix(opts.bvalsfile.value());
    	bvecs=read_ascii_matrix(opts.bvecsfile.value());
    	if(bvecs.Nrows()>3) bvecs=bvecs.t();
    	if(bvals.Nrows()>1) bvals=bvals.t();
    	for(int i=1;i<=bvecs.Ncols();i++){
      		float tmpsum=sqrt(bvecs(1,i)*bvecs(1,i)+bvecs(2,i)*bvecs(2,i)+bvecs(3,i)*bvecs(3,i));
      		if(tmpsum!=0){
			bvecs(1,i)=bvecs(1,i)/tmpsum;
			bvecs(2,i)=bvecs(2,i)/tmpsum;
			bvecs(3,i)=bvecs(3,i)/tmpsum;
      		}
    	}
	int ndirections = bvals.Ncols();
	int dirs_grad = 9;  // always ???
	if(ndirections<=0){
		cerr << "The number of gradient directions must be greater than 0" << endl;
    		exit (EXIT_FAILURE);
	}
	if(bvecs.Ncols()!=ndirections){
		cerr << "xfibres_gpu error: The number of elements in bvals and number of vectors in bvecs must be the same\n" << endl;
    		exit (EXIT_FAILURE);
    	}


	///////////////////////////////////////////
	///////////// Check Arguments /////////////
	///////////////////////////////////////////
	string subjdir=argv[argc-4];
	struct stat sb;
	if (stat(subjdir.data(), &sb) != 0 || !S_ISDIR(sb.st_mode)){
		cerr << "xfibres_gpu. The last 4 arguments must be:\n\tSubject-directory \n\tNumThisPart \n\tTotalNumParts \n\tTotalNumVoxels(all parts)" << endl;
		cerr << "\nSubject-directory: "<< subjdir << " is not a directory" << endl;
    		exit (EXIT_FAILURE);
	}

	istringstream ss_idPart(argv[argc-3]);
	int idPart;
	if (!(ss_idPart >> idPart)){
		cerr << "xfibres_gpu. The last 4 arguments must be:\n\tSubject-directory \n\tNumThisPart \n\tTotalNumParts \n\tTotalNumVoxels(all parts)" << endl;
		cerr << "\nNumThisPart: " << argv[argc-3] << " is not a valid number" << endl;
    		exit (EXIT_FAILURE);
	}

	istringstream ss_nParts(argv[argc-2]);
	int nParts;
	if (!(ss_nParts >> nParts)){
		cerr << "xfibres_gpu. The last 4 arguments must be:\n\tSubject-directory \n\tNumThisPart \n\tTotalNumParts \n\tTotalNumVoxels(all parts)" << endl;
		cerr << "\nTotalNumParts: " << argv[argc-2] << " is not a valid number" << endl;
    		exit (EXIT_FAILURE);
	}

	istringstream ss_totalNvox(argv[argc-1]);
	int totalNvox;
	if (!(ss_totalNvox >> totalNvox)|| totalNvox<=0){
		cerr << "xfibres_gpu. The last 4 arguments must be:\n\tSubject-directory \n\tNumThisPart \n\tTotalNumParts \n\tTotalNumVoxels(all parts)" << endl;
		cerr << "\nThe number of voxels must be greater than 0" << endl;
    		exit (EXIT_FAILURE);
	}

	///////////////////////////////////////////
	////////// Read my part of data ///////////
	///////////////////////////////////////////
	int size_part = totalNvox / nParts;
	// if last part
	if(idPart==(nParts-1)) size_part = totalNvox-(size_part*(nParts-1));

	Matrix mydatam;
	mydatam.ReSize(ndirections,size_part);
	ifstream in;
	in.open(opts.datafile.value().data(), ios::in | ios::binary);
	in.read((char*)&mydatam(1,1), size_part*ndirections*sizeof(Real));
	in.close();

	Matrix mygradm;
	mygradm.ReSize(dirs_grad,size_part);
	if (opts.grad_file.set()){
		ifstream in;
		in.open(opts.grad_file.value().data(), ios::in | ios::binary);
		in.read((char*)&mygradm(1,1), size_part*dirs_grad*sizeof(Real));
		in.close();
	}

	cout << "Number of Voxels to compute in this part: " << size_part << endl;
	cout << "Number of Directions: " << ndirections << endl;

    	if(opts.rician.value() && !opts.nonlin.value())
      		cout<<"Rician noise model requested. Non-linear parameter initialization will be performed, overriding other initialization options!"<<endl<<endl;

	//////////////////////////////////////////////////////////////
	////////// Divide the process in Subparts ////////////////////
	//////////////////////////////////////////////////////////////

	int nsubparts = (size_part/SIZE_SUB_PART); 			//SIZE_SUB_PART voxels per iteration
	int size_sub_part = SIZE_SUB_PART;
	if(size_part%SIZE_SUB_PART) nsubparts++;
	int last_sub_part = size_part - ((nsubparts-1)*SIZE_SUB_PART);

	if(last_sub_part<(SIZE_SUB_PART*0.5)){ 	//if is too small the last part we distribute it between the others
		if(nsubparts-1){
			size_sub_part = size_sub_part + last_sub_part/(nsubparts-1);
			nsubparts--;
		}else{
			size_sub_part = 0;
		}
		last_sub_part = size_part - ((nsubparts-1)*size_sub_part);
	}

	Matrix mydatam_part;
	Matrix mygradm_part;

	for(int i=0;i<nsubparts-1;i++){
		cout << "SubPart " << i+1 << " of " << nsubparts << ": processing " << size_sub_part << " voxels" <<  endl;
		mydatam_part = mydatam.SubMatrix(1,ndirections,i*size_sub_part+1,(i+1)*size_sub_part);
		if (opts.grad_file.set()) mygradm_part = mygradm.SubMatrix(1,dirs_grad,i*size_sub_part+1,(i+1)*size_sub_part);
		xfibres_gpu(mydatam_part,bvecs,bvals,mygradm_part,idPart,i,rand(),subjdir);
		//for the monitor
		if(nParts==1){
			std::string file_name;
			file_name.assign(subjdir);
			file_name += ".bedpostX/logs/monitor/";
			char n[4];
			sprintf(n,"%d",i);
			file_name += n;
			ofstream out;
			out.open(file_name.data(), ios::out | ios::binary);
			out.write("done",4*sizeof(char));
			out.close();
		}
	}

	cout << "SubPart " << nsubparts << " of " << nsubparts << ": processing " << last_sub_part << " voxels" <<  endl;
	mydatam_part = mydatam.SubMatrix(1,ndirections,(nsubparts-1)*size_sub_part+1,size_part);
	if (opts.grad_file.set()) mygradm_part = mygradm.SubMatrix(1,dirs_grad,(nsubparts-1)*size_sub_part+1,size_part);
	xfibres_gpu(mydatam_part,bvecs,bvals,mygradm_part,idPart,nsubparts-1,rand(),subjdir);
	//for the monitor
	if(nParts==1){
		std::string file_name;
		file_name.assign(subjdir);
		file_name += ".bedpostX/logs/monitor/";
		char n[4];
		sprintf(n,"%d",(nsubparts-1));
		file_name += n;
		ofstream out;
		out.open(file_name.data(), ios::out | ios::binary);
		out.write("done",4*sizeof(char));
		out.close();
	}

	//////////////////////////////////////////////////////////////
	////////// JOIN Results of the Subparts //////////////////////
	//////////////////////////////////////////////////////////////

	if(opts.modelnum.value()==1){
		join_subParts("mean_dsamples",size_part,nsubparts,size_sub_part,last_sub_part,true);
	}else if(opts.modelnum.value()>=2){
		join_subParts("mean_dsamples",size_part,nsubparts,size_sub_part,last_sub_part,true);
		join_subParts("mean_d_stdsamples",size_part,nsubparts,size_sub_part,last_sub_part,true);
		//join_subParts("dsamples",size_part,nsubparts,size_sub_part,last_sub_part,false);
		//join_subParts("d_stdsamples",size_part,nsubparts,size_sub_part,last_sub_part,false);
		if(opts.modelnum.value()==3){
			join_subParts("mean_Rsamples",size_part,nsubparts,size_sub_part,last_sub_part,true);
		}
	}
	if (opts.f0.value()){
      		join_subParts("mean_f0samples",size_part,nsubparts,size_sub_part,last_sub_part,true);
		//join_subParts("f0samples",size_part,nsubparts,size_sub_part,last_sub_part,false);
    	}
    	if (opts.rician.value()){
		join_subParts("mean_tausamples",size_part,nsubparts,size_sub_part,last_sub_part,true);
    	}

	for(int f=0;f<opts.nfibres.value();f++){
		join_subParts("th"+num2str(f+1)+"samples",size_part,nsubparts,size_sub_part,last_sub_part,false);
		join_subParts("ph"+num2str(f+1)+"samples",size_part,nsubparts,size_sub_part,last_sub_part,false);
		join_subParts("f"+num2str(f+1)+"samples",size_part,nsubparts,size_sub_part,last_sub_part,false);

		//join_subParts("mean_f"+num2str(f+1)+"samples",size_part,nsubparts,size_sub_part,last_sub_part,true);
		//join_subParts("dyads"+num2str(f+1),size_part,nsubparts,size_sub_part,last_sub_part,false);
	}

	join_subParts("mean_S0samples",size_part,nsubparts,size_sub_part,last_sub_part,true); //the last one to control with monitor

	gettimeofday(&t2,NULL);
    	time=timeval_diff(&t2,&t1);
	cout << endl << "Part processed in: " << time << " seconds" << endl;

	//for the monitor
	if(nParts>1){
		std::string file_name;
		file_name.assign(subjdir);
		file_name += ".bedpostX/logs/monitor/";
		char n[4];
		sprintf(n,"%d",idPart);
		file_name += n;
		ofstream out;
		out.open(file_name.data(), ios::out | ios::binary);
		out.write("done",4*sizeof(char));
		out.close();
	}

  	return 0;
}

void join_subParts(string name, int size_part, int nsubparts, int size_sub_part, int last_sub_part, bool mean){
	Log& logger = LogSingleton::getInstance();
    	xfibresOptions& opts = xfibresOptions::getInstance();

	int nsamples = opts.njumps.value()/opts.sampleevery.value();
	if(mean) nsamples=1;

	Matrix tmp(nsamples,0);
	Matrix part;
	ifstream in;
	ofstream out;

	string file_name;

	for(int i=0;i<nsubparts-1;i++){
		part.ReSize(nsamples,size_sub_part);
		file_name = logger.appendDir(name+"_"+num2str(i));
		in.open(file_name.data(), ios::in | ios::binary);
		in.read((char*)&part(1,1), size_sub_part*nsamples*sizeof(Real));
		in.close();
		remove (file_name.data());
		tmp |=  part;
	}
	part.ReSize(nsamples,last_sub_part);
	file_name = logger.appendDir(name+"_"+num2str(nsubparts-1));
	in.open (file_name.data(), ios::in | ios::binary);
	in.read((char*)&part(1,1), last_sub_part*nsamples*sizeof(Real));
	in.close();
	remove (file_name.data());
	tmp |= part;

	file_name = logger.appendDir(name+"J");
	out.open(file_name.data(), ios::out | ios::binary);
	out.write((char*)&tmp(1,1),size_part*nsamples*sizeof(Real));
	out.close();

}
