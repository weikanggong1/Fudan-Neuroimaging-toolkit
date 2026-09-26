/*  merge_parts_gpu.cc

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <sys/stat.h>
#include <string>
#include <iostream>

#include "utils/log.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

#include "xfibresoptions.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace Xfibres;

void join_Parts(NEWIMAGE::volume<float> mask, string name_in, string name_out, string subjdir, int nvox, int nsamples, int nParts, float max, float min){
		int size_parts = nvox/nParts;
		int last_part = nvox - ((nParts-1)*size_parts);

		//if(mean) nsamples=1;

		Matrix result(nsamples,0);
		Matrix part;

		for(int i=0;i<(nParts-1);i++){
			part.ReSize(nsamples,size_parts);
			std::ostringstream num;
			num << i;
			std::string part_number;
			part_number.assign(num.str());
			std::string aux;
			while(part_number.size()<4){
				aux = "0" + part_number;
				part_number.assign(aux);
			}
			std::string file_name;
			file_name.assign(subjdir);
			file_name += ".bedpostX/diff_parts/data_part_";
			file_name += part_number;
			file_name += "/";
			file_name += name_in;
			file_name += "J";

			ifstream in;
			in.open (file_name.data(), ios::in | ios::binary);
			in.read((char*)&part(1,1), size_parts*nsamples*sizeof(Real));
			in.close();

			result |= part;
		}
		part.ReSize(nsamples,last_part);
		std::ostringstream num;
		num << nParts-1;;
		std::string part_number;
		part_number.assign(num.str());
		std::string aux;
		while(part_number.size()<4){
			aux = "0" + part_number;
			part_number.assign(aux);
		}
		std::string file_name;
		file_name.assign(subjdir);
		file_name += ".bedpostX/diff_parts/data_part_";
		file_name += part_number;
		file_name += "/";
		file_name += name_in;
		file_name += "J";

		ifstream in;
		in.open (file_name.data(), ios::in | ios::binary);
		in.read((char*)&part(1,1), last_part*nsamples*sizeof(Real));
		in.close();
		result |= part;

		NEWIMAGE::volume4D<float> tmp;
      		tmp.setmatrix(result,mask);
		if(max==-10) max=tmp.max();
		if(min==-10) min=tmp.min();
		tmp.setDisplayMaximumMinimum(max,min);
     		save_volume4D(tmp,subjdir+".bedpostX/"+name_out);
}

//////////////////////////////////////////////////////////
//       MERGE THE OUTPUTS FILES OF BEDPOSTX
//////////////////////////////////////////////////////////
//parameters:
// argc - 3 nvox
// argc - 2 nParts
// argc - 1 subjdir

int main(int argc, char *argv[])
{
	// Setup logging:
    	Log& logger = LogSingleton::getInstance();
    	xfibresOptions& opts = xfibresOptions::getInstance();
	opts.parse_command_line(argc-3,argv,logger);

    	NEWIMAGE::volume<float> mask;
    	read_volume(mask,opts.maskfile.value());

	cout << opts.maskfile.value() << endl;

	///////////////////////////////////////////
	///////////// Check Arguments /////////////
	///////////////////////////////////////////
	istringstream ss_nvox(argv[argc-3]);
	int nvox;
	if (!(ss_nvox >> nvox)|| nvox<=0){
		cerr << "merge_parts_gpu. The last 3 arguments must be:\n\tTotalNumVoxels(all parts)\n\tTotalNumParts \n\tSubject-directory" << endl;
		cerr << "\nThe number of voxels must be greater than 0" << endl;
    		exit (EXIT_FAILURE);
	}

	istringstream ss_nParts(argv[argc-2]);
	int nParts;
	if (!(ss_nParts >> nParts)){
		cerr << "merge_parts_gpu. The last 3 arguments must be:\n\tTotalNumVoxels(all parts)\n\tTotalNumParts \n\tSubject-directory" << endl;
		cerr << "\nTotalNumParts: " << argv[argc-2] << " is not a valid number" << endl;
    		exit (EXIT_FAILURE);
	}

	string subjdir=argv[argc-1];
	struct stat sb;
	if (stat(subjdir.data(), &sb) != 0 || !S_ISDIR(sb.st_mode)){
		cerr << "merge_parts_gpu. The last 3 arguments must be:\n\tTotalNumVoxels(all parts)\n\tTotalNumParts \n\tSubject-directory" << endl;
		cerr << "\nSubject-directory: "<< subjdir << " is not a directory" << endl;
    		exit (EXIT_FAILURE);
	}

	int nsamples = opts.njumps.value()/opts.sampleevery.value();
	if(nsamples<=0){
		cerr << "The number of samples must be greater than 0" << endl;
    		exit (EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////
	////////// JOIN Results of the Parts //////////////////////
	//////////////////////////////////////////////////////////////

	if(opts.modelnum.value()==1){
		join_Parts(mask,"mean_dsamples","mean_dsamples",subjdir, nvox, 1, nParts, -10, 0);
	}else if(opts.modelnum.value()>=2){
		join_Parts(mask,"mean_dsamples","mean_dsamples",subjdir, nvox, 1, nParts, -10, 0);
		join_Parts(mask,"mean_d_stdsamples","mean_d_stdsamples",subjdir, nvox, 1, nParts, -10, 0);
		//join_Parts(mask,"dsamples","dsamples",subjdir, nvox, nsamples, nParts, -10, 0);
		//join_Parts(mask,"d_stdsamples","d_stdsamples",subjdir, nvox, nsamples, nParts, -10, 0);
		if(opts.modelnum.value()==3){
			join_Parts(mask,"mean_Rsamples","mean_Rsamples",subjdir, nvox, 1, nParts, 1, 0);
		}
	}
	if (opts.f0.value()){
      		join_Parts(mask,"mean_f0samples","mean_f0samples",subjdir, nvox, 1, nParts, 1, 0);
		//join_Parts(mask,"f0samples","f0samples",subjdir, nvox, nsamples, nParts, 1, 0);
    	}
    	if (opts.rician.value()){
		join_Parts(mask,"mean_tausamples","mean_tausamples",subjdir, nvox, 1, nParts, -10, 0);
    	}
	join_Parts(mask,"mean_S0samples","mean_S0samples",subjdir, nvox, 1, nParts, -10, 0);

	for(int f=0;f<opts.nfibres.value();f++){
		join_Parts(mask,"th"+num2str(f+1)+"samples","merged_th"+num2str(f+1)+"samples",subjdir, nvox, nsamples, nParts, -10, -10);
		join_Parts(mask,"ph"+num2str(f+1)+"samples","merged_ph"+num2str(f+1)+"samples",subjdir, nvox, nsamples, nParts, -10, -10);
		join_Parts(mask,"f"+num2str(f+1)+"samples","merged_f"+num2str(f+1)+"samples",subjdir, nvox, nsamples, nParts, 1, 0);

		//join_Parts(mask,"mean_f"+num2str(f+1)+"samples",subjdir, nvox, 1, nParts, 1, 0);
		//join_Parts(mask,"dyads"+num2str(f+1),subjdir, nvox, nsamples, nParts, 1, -1);
	}

  	return 0;
}
