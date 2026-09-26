/*  saveResults_ptxGPU.cc

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <vector>
#include <iostream>

#include "utils/log.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimage.h"

#include "probtrackx.h"
#include "CUDA/tractographyInput.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace NEWIMAGE;
using namespace MISCMATHS;
using namespace TRACT;

void counter_save_total(int*& keeptotal, int numKeeptotals){
  // save total number of particles that made it through the streamlining
  Log& logger = LogSingleton::getInstance();
  ColumnVector keeptotvec(numKeeptotals);
  for (int i=1;i<=numKeeptotals;i++)
    keeptotvec(i)=keeptotal[i-1];
  write_ascii_matrix(keeptotvec,logger.appendDir("waytotal"));
}

void counter_save_pathdist(volume<float>& m_prob,volume<float>& m_prob2){
  Log& logger = LogSingleton::getInstance();
  probtrackxOptions& opts =probtrackxOptions::getInstance();

  m_prob.setDisplayMaximumMinimum(m_prob.max(),m_prob.min());
  save_volume(m_prob,logger.appendDir(opts.outfile.value()));

  if(opts.omeanpathlength.value()){
    if(!opts.pathdist.value()){
      for (int z=0; z<m_prob.zsize(); z++) {
	for (int y=0; y<m_prob.ysize(); y++) {
	  for (int x=0; x<m_prob.xsize(); x++) {
	    if(m_prob(x,y,z)){
	      m_prob(x,y,z)=m_prob2(x,y,z)/m_prob(x,y,z);
	    }else{
	      m_prob(x,y,z)=0;
	    }
	  }
	}
      }
    }else{
      for (int z=0; z<m_prob.zsize(); z++) {
	for (int y=0; y<m_prob.ysize(); y++) {
	  for (int x=0; x<m_prob.xsize(); x++) {
	    if(m_prob2(x,y,z)){
	      m_prob(x,y,z)=m_prob(x,y,z)/m_prob2(x,y,z);
	    }else{
	      m_prob(x,y,z)=0;
	    }
	  }
	}
      }
    }
    m_prob.setDisplayMaximumMinimum(m_prob.max(),m_prob.min());
    save_volume(m_prob,logger.appendDir(opts.outfile.value())+"_lengths");
  }

  /*if(opts.pathfile.set()){
    m_prob_alt.save_rois(logger.appendDir(opts.outfile.value())+"_alt");
    //m_prob_alt.save_as_volume(logger.appendDir(opts.outfile.value())+"_alt_vol");
    //m_beenhere_alt.save_rois(logger.appendDir(opts.outfile.value())+"_beenhere");
    }*/

}

void counter_save(
    	tractographyData& 	data_host,
	volume<float> 		*m_prob, 	// spatial histogram of tract location within brain mask (in seed space)
	volume<float> 		*m_prob2,	// omeanpathlength
	float**			ConNet,		// Network mode
	float**			ConNetb,	// Network mode
	int 			nRowsNet,	// Network mode
	int 			nColsNet,	// Network mode
	float**			ConMat1,
	float**			ConMat1b,	// omeanpathlength
	int 			nRowsMat1,
	int 			nColsMat1,
	float**			ConMat3,
	float**			ConMat3b,
	int 			nRowsMat3,
	int 			nColsMat3,
	float*			m_s2targets,
	float*			m_s2targetsb,
	vector< vector<float> >& m_save_paths,
	volume4D<float> 	*m_localdir)
{
  Log& logger = LogSingleton::getInstance();
  probtrackxOptions& opts =probtrackxOptions::getInstance();
  if(opts.simpleout.value()){
    counter_save_pathdist(*m_prob,*m_prob2);
  }

  if(opts.network.value()){
    string file(logger.appendDir("fdt_network_matrix"));
    ostream* out=0;
    out= new ofstream(file.c_str());
    //(*out) << setprecision(8);

    if(!opts.omeanpathlength.value()){
      int pos=0;
      for(int i=0;i<nRowsNet;i++){
	for(int j=0;j<nColsNet;j++){
	  (*out) << ConNet[0][pos] <<"  ";
	  pos++;
				}
	(*out) << endl;
      }
      delete out;
    }else{
      // ConNet pathlengths, ConNetb 1 hits
      if(!opts.pathdist.value()){
	int pos=0;
	for(int i=0;i<nRowsNet;i++){
	  for(int j=0;j<nColsNet;j++){
	    (*out) << ConNetb[0][pos] <<"  ";
	    pos++;
	  }
	  (*out) << endl;
	}
	delete out;
      }else{
	int pos=0;
	for(int i=0;i<nRowsNet;i++){
	  for(int j=0;j<nColsNet;j++){
	    (*out) << ConNet[0][pos] <<"  ";
	    pos++;
	  }
	  (*out) << endl;
	}
	delete out;
      }
      string file(logger.appendDir("fdt_network_matrix_lengths"));
      ostream* out2=0;
      out2= new ofstream(file.c_str());
      int pos=0;
      for(int i=0;i<nRowsNet;i++){
	for(int j=0;j<nColsNet;j++){
	  if(ConNetb[0][pos]!=0)
	    (*out2) << (ConNet[0][pos])/(ConNetb[0][pos]) <<"  ";
	  else
	    (*out2) << 0 <<"  ";
	  pos++;
	}
	(*out2) << endl;
      }
      delete out2;
    }

  }
  if(opts.matrix1out.value()||opts.matrix2out.value()){
    ostream* out;
    if(opts.matrix2out.value()){
      string file(logger.appendDir("fdt_matrix2.dot"));
      out=0;
      out= new ofstream(file.c_str());
    }else{
      string file(logger.appendDir("fdt_matrix1.dot"));
      out=0;
      out= new ofstream(file.c_str());
    }
    //(*out) << setprecision(8);
    if(!opts.omeanpathlength.value()){
      for(int i=0;i<nRowsMat1;i++){
	for(int j=0;j<nColsMat1;j++){
	  if(ConMat1[i][j])
	    (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat1[i][j] << endl;
	}
      }
      (*out) << nRowsMat1 <<"  "<< nColsMat1 <<"  " << 0 << endl;
      delete out;
    }else{
      if(!opts.pathdist.value()){
	for(int i=0;i<nRowsMat1;i++){
	  for(int j=0;j<nColsMat1;j++){
	    if(ConMat1b[i][j])
	      (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat1b[i][j] << endl;
	  }
	}
	(*out) << nRowsMat1 <<"  "<< nColsMat1 <<"  " << 0 << endl;
	delete out;
      }else{
	for(int i=0;i<nRowsMat1;i++){
	  for(int j=0;j<nColsMat1;j++){
	    if(ConMat1[i][j])
	      (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat1[i][j] << endl;
	  }
	}
	(*out) << nRowsMat1 <<"  "<< nColsMat1 <<"  " << 0 << endl;
	delete out;

      }
    }
    if(opts.omeanpathlength.value()){
      ostream* out2;
      if(opts.matrix2out.value()){
	string file2(logger.appendDir("fdt_matrix2_lengths.dot"));
	out2=0;
	out2= new ofstream(file2.c_str());
      }else{
	string file2(logger.appendDir("fdt_matrix1_lengths.dot"));
	out2=0;
	out2= new ofstream(file2.c_str());
      }
      for(int i=0;i<nRowsMat1;i++){
	for(int j=0;j<nColsMat1;j++){
	  if(ConMat1b[i][j])
	    (*out2) << i+1 <<"  "<< j+1 <<"  "<< (ConMat1[i][j]/ConMat1b[i][j]) << endl;
	}
      }
      (*out2) << nRowsMat1 <<"  "<< nColsMat1 <<"  " << 0 << endl;
      delete out2;
    }
  }

  if(opts.matrix3out.value()){
    string file(logger.appendDir("fdt_matrix3.dot"));
    ostream* out=0;
    out= new ofstream(file.c_str());
    //(*out) << setprecision(8);
    if(!opts.omeanpathlength.value()){
      for(int i=0;i<nRowsMat3;i++){
	for(int j=0;j<nColsMat3;j++){
	  if(ConMat3[i][j])
	    (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat3[i][j] << endl;
	}
      }
      (*out) << nRowsMat3 <<"  "<< nColsMat3 <<"  " << 0 << endl;
      delete out;
    }else{
      if(!opts.pathdist.value()){
	for(int i=0;i<nRowsMat3;i++){
	  for(int j=0;j<nColsMat3;j++){
	    if(ConMat3b[i][j])
	      (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat3b[i][j] << endl;
	  }
	}
	(*out) << nRowsMat3 <<"  "<< nColsMat3 <<"  " << 0 << endl;
	delete out;
      }else{
	for(int i=0;i<nRowsMat3;i++){
	  for(int j=0;j<nColsMat3;j++){
	    if(ConMat3[i][j])
	      (*out) << i+1 <<"  "<< j+1 <<"  "<< ConMat3[i][j] << endl;
	  }
	}
	(*out) << nRowsMat3 <<"  "<< nColsMat3 <<"  " << 0 << endl;
	delete out;
      }
    }
    if(opts.omeanpathlength.value()){
      string file2(logger.appendDir("fdt_matrix3_lengths.dot"));
      ostream* out2=0;
      out2= new ofstream(file2.c_str());
      for(int i=0;i<nRowsMat3;i++){
	for(int j=0;j<nColsMat3;j++){
	  if(ConMat3b[i][j])
	    (*out2) << i+1 <<"  "<< j+1 <<"  "<< (ConMat3[i][j]/ConMat3b[i][j]) << endl;
	}
      }
      (*out2) << nRowsMat3 <<"  "<< nColsMat3 <<"  " << 0 << endl;
      delete out2;
    }
  }

  ///////////////////////////////////
  ////// save seeds to targets //////
  ///////////////////////////////////
  if(opts.s2tout.value()){
    size_t pos=0;
    int ntargets=data_host.targets.NVols+data_host.targets.NSurfs;
    if (fsl_imageexists(opts.seedfile.value())){
      volume<float> tmp;
      read_volume(tmp,opts.seedfile.value());

      for(int targ=0;targ<ntargets;targ++){
	volume<float> tmp2;
	volume<float> tmp3;
	tmp2.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize());
	copybasicproperties(tmp,tmp2);
	tmp2=0;
	if(opts.omeanpathlength.value()){
	  tmp3.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize());
	  copybasicproperties(tmp,tmp3);
	  tmp3=0;
	}
	for (int z=0;z<tmp.zsize();z++){
	  for (int y=0;y<tmp.ysize();y++){
	    for (int x=0;x<tmp.xsize();x++){
	      if (tmp(x,y,z)){
		if(!opts.omeanpathlength.value()){
		  tmp2(x,y,z)=m_s2targets[pos];
		}else{
		  if(opts.pathdist.value()){
		    tmp2(x,y,z)=m_s2targets[pos];
		  }else{
		    tmp2(x,y,z)=m_s2targetsb[pos];
		  }
		  if(m_s2targetsb[pos])
		    tmp3(x,y,z)=m_s2targets[pos]/m_s2targetsb[pos];
		  else
		    tmp3(x,y,z)=0;
		}
		pos++;
	      }
	    }
	  }
	}
	string fname;
	fname=logger.appendDir("seeds_to_"+data_host.targetnames[targ]);
	tmp2.setDisplayMaximumMinimum(tmp2.max(),tmp2.min());
	save_volume(tmp2,fname);
	if(opts.omeanpathlength.value()){
	  string fname3;
	  fname3=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+"_lengths");
	  tmp3.setDisplayMaximumMinimum(tmp3.max(),tmp3.min());
	  save_volume(tmp3,fname3);
	}
      }
    }else if(meshExists(opts.seedfile.value())){
      CSV seeds;//(refvol);
      seeds.set_convention(opts.meshspace.value());
      seeds.load_rois(opts.seedfile.value());

      for(int targ=0;targ<ntargets;targ++){
	//save ascii
	string fname;
	fname=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+".asc");
	ofstream f(fname.c_str());
	stringstream flot;
	stringstream flot2;
	int nvertices=data_host.seeds_mesh_info[0];
	int nfaces=data_host.seeds_mesh_info[1];

	if(f.is_open()){
	  size_t pos2=0;
	  for(int i=0;i<nvertices;i++){
	    flot<<data_host.seeds_vertices[pos2]<<" "
		<<data_host.seeds_vertices[pos2+1]<<" "
		<<data_host.seeds_vertices[pos2+2]<<" ";
	    if(data_host.seeds_act[pos2/3]){
	      if(!opts.omeanpathlength.value()){
		flot<<m_s2targets[pos]<<endl;
	      }else{
		if(opts.pathdist.value()){
		  flot<<m_s2targets[pos]<<endl;
		}else{
		  flot<<m_s2targetsb[pos]<<endl;
		}
		flot2<<data_host.seeds_vertices[pos2]<<" "
		     <<data_host.seeds_vertices[pos2+1]<<" "
		     <<data_host.seeds_vertices[pos2+2]<<" ";
		if(m_s2targetsb[pos])
		  flot2<<(m_s2targets[pos]/m_s2targetsb[pos])<<endl;
		else
		  flot2<<0<<endl;
	      }
	      pos++;
	    }else{
	      flot<<0<<endl;
	      if(opts.omeanpathlength.value()){
		flot2<<data_host.seeds_vertices[pos2]<<" "
		     <<data_host.seeds_vertices[pos2+1]<<" "
		     <<data_host.seeds_vertices[pos2+2]<<" "<<0<<endl;
	      }
	    }
	    pos2=pos2+3;
	  }
	  pos2=0;
	  for(int i=0;i<nfaces;i++){
	    flot<<data_host.seeds_faces[pos2]<<" "
		<<data_host.seeds_faces[pos2+1]<<" "
		<<data_host.seeds_faces[pos2+2]<<" "<<0<<endl;
	    if(opts.omeanpathlength.value()){
	      flot2<<data_host.seeds_faces[pos2]<<" "
		   <<data_host.seeds_faces[pos2+1]<<" "
		   <<data_host.seeds_faces[pos2+2]<<" "<<0<<endl;
	    }
	    pos2=pos2+3;
	  }
	  f<<"#!ascii file"<<endl;
	  f<<nvertices<<" "<<nfaces<<endl<<flot.str();
	  f.close();
	  if(opts.omeanpathlength.value()){
	    string fname2;
	    fname2=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+"_lengths.asc");
	    ofstream f2(fname2.c_str());
	    if(f2.is_open()){
	      f2<<"#!ascii file"<<endl;
	      f2<<nvertices<<" "<<nfaces<<endl<<flot2.str();
	      f2.close();
	    }else cerr<<"Save_ascii:error opening file for writing: "<<fname2<<endl;
	  }
	}else cerr<<"Save_ascii:error opening file for writing: "<<fname<<endl;
      }
    }else{
      // list of ROIS
      vector<string> fnames;
      ifstream fs(opts.seedfile.value().c_str());
      string tmp;
      if (fs){
	fs>>tmp;
	do{
	  fnames.push_back(tmp);
	  fs>>tmp;
	}while (!fs.eof());
      }
      for(int targ=0;targ<ntargets;targ++){
	int id_mesh=0;
	for(unsigned int R=0;R<fnames.size();R++){
	  if (fsl_imageexists(fnames[R])){
	    volume<float> tmp;
	    read_volume(tmp,fnames[R]);
	    volume<float> tmp2;
	    volume<float> tmp3;
	    tmp2.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize());
	    copybasicproperties(tmp,tmp2);
	    tmp2=0;
	    if(opts.omeanpathlength.value()){
	      tmp3.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize());
	      copybasicproperties(tmp,tmp3);
	      tmp3=0;
	    }
	    for (int z=0;z<tmp.zsize();z++){
	      for (int y=0;y<tmp.ysize();y++){
		for (int x=0;x<tmp.xsize();x++){
		  if (tmp(x,y,z)){
		    if(!opts.omeanpathlength.value()){
		      tmp2(x,y,z)=m_s2targets[pos];
		    }else{
		      if(opts.pathdist.value()){
			tmp2(x,y,z)=m_s2targets[pos];
		      }else{
			tmp2(x,y,z)=m_s2targetsb[pos];
		      }
		      if(m_s2targetsb[pos])
			tmp3(x,y,z)=m_s2targets[pos]/m_s2targetsb[pos];
		      else
			tmp3(x,y,z)=0;
		    }
		    pos++;
		  }
		}
	      }
	    }
	    string fname;
	    if(ntargets>1)
	      fname=logger.appendDir("seeds_"+num2str(R)+"_to_"+data_host.targetnames[targ]);
	    else
	      fname=logger.appendDir("seeds_to_"+data_host.targetnames[targ]);
	    tmp2.setDisplayMaximumMinimum(tmp2.max(),tmp2.min());
	    save_volume(tmp2,fname);
	    if(opts.omeanpathlength.value()){
	      string fname3;
	      if(ntargets>1)
		fname3=logger.appendDir("seeds_"+num2str(R)+"_to_"+data_host.targetnames[targ]+"_lengths");
	      else
		fname3=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+"_lengths");
	      tmp3.setDisplayMaximumMinimum(tmp3.max(),tmp3.min());
	      save_volume(tmp3,fname3);
	    }
	  }else if(meshExists(fnames[R])){
	    //save ascii
	    string fname;
	    if(ntargets>1)
	      fname=logger.appendDir("seeds_"+num2str(R)+"_to_"+data_host.targetnames[targ]+".asc");
	    else
	      fname=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+".asc");
	    ofstream f(fname.c_str());
	    stringstream flot;
	    stringstream flot2;
	    int nvertices=data_host.seeds_mesh_info[id_mesh*2];
	    int nfaces=data_host.seeds_mesh_info[id_mesh*2+1];
	    if (f.is_open()){
	      size_t pos2=0;
	      for(int i=0;i<id_mesh;i++) pos2+= (data_host.seeds_mesh_info[i*2]*3);
	      for(int i=0;i<nvertices;i++){
		flot<<data_host.seeds_vertices[pos2]<<" "
		    <<data_host.seeds_vertices[pos2+1]<<" "
		    <<data_host.seeds_vertices[pos2+2]<<" ";
		if(data_host.seeds_act[pos2/3]){
		  if(!opts.omeanpathlength.value()){
		    flot<<m_s2targets[pos]<<endl;
		  }else{
		    if(opts.pathdist.value()){
		      flot<<m_s2targets[pos]<<endl;
		    }else{
		      flot<<m_s2targetsb[pos]<<endl;
		    }
		    flot2<<data_host.seeds_vertices[pos2]<<" "
			 <<data_host.seeds_vertices[pos2+1]<<" "
			 <<data_host.seeds_vertices[pos2+2]<<" ";
		    if(m_s2targetsb[pos])
		      flot2<<(m_s2targets[pos]/m_s2targetsb[pos])<<endl;
		    else
		      flot2<<0<<endl;
		  }
		  pos++;
		}else{
		  flot<<0<<endl;
		  if(opts.omeanpathlength.value()){
		    flot2<<data_host.seeds_vertices[pos2]<<" "
			 <<data_host.seeds_vertices[pos2+1]<<" "
			 <<data_host.seeds_vertices[pos2+2]<<" "<<0<<endl;
		  }
		}
		pos2=pos2+3;
	      }
	      pos2=0;
	      for(int i=0;i<id_mesh;i++) pos2+= (data_host.seeds_mesh_info[i*2+1]*3);
	      for(int i=0;i<nfaces;i++){
		flot<<data_host.seeds_faces[pos2]<<" "
		    <<data_host.seeds_faces[pos2+1]<<" "
		    <<data_host.seeds_faces[pos2+2]<<" "<<0<<endl;
		if(opts.omeanpathlength.value()){
		  flot2<<data_host.seeds_faces[pos2]<<" "
		       <<data_host.seeds_faces[pos2+1]<<" "
		       <<data_host.seeds_faces[pos2+2]<<" "<<0<<endl;
		}
		pos2=pos2+3;
	      }
	      f<<"#!ascii file"<<endl;
	      f<<nvertices<<" "<<nfaces<<endl<<flot.str();
	      f.close();
	      if(opts.omeanpathlength.value()){
		string fname2;
		if(ntargets>1)
		  fname2=logger.appendDir("seeds_"+num2str(R)+"_to_"+data_host.targetnames[targ]+"_lengths.asc");
		else
		  fname2=logger.appendDir("seeds_to_"+data_host.targetnames[targ]+"_lengths.asc");
		ofstream f2(fname2.c_str());
		if (f2.is_open()){
		  f2<<"#!ascii file"<<endl;
		  f2<<nvertices<<" "<<nfaces<<endl<<flot2.str();
		  f2.close();
		}else cerr<<"Save_ascii:error opening file for writing: "<<fname2<<endl;
	      }
	    }else cerr<<"Save_ascii:error opening file for writing: "<<fname<<endl;
	    id_mesh++;
	  }
	}
      }
    }
    if(opts.s2tastext.value()){
      pos=0;
      string file(logger.appendDir("matrix_seeds_to_all_targets"));
      string file2(logger.appendDir("matrix_seeds_to_all_targets_lengths"));
      ostream* out=0;
      ostream* out2=0;
      out= new ofstream(file.c_str());
      if(opts.omeanpathlength.value()){
	out2= new ofstream(file2.c_str());
      }
      for(size_t i=0;i<data_host.nseeds;i++){
	pos=i;
	for(int j=0;j<ntargets;j++){
	  if(!opts.omeanpathlength.value()){
	    (*out) << m_s2targets[pos] <<"  ";
	  }else{
	    if(opts.pathdist.value()){
	      (*out) << m_s2targets[pos] <<"  ";
	    }else{
	      (*out) << m_s2targetsb[pos] <<"  ";
	    }
	    if(m_s2targetsb[pos])
	      (*out2) << (m_s2targets[pos]/m_s2targetsb[pos]) <<"  ";
	    else
	      (*out2) << 0 <<"  ";
	  }
	  pos=pos+data_host.nseeds;
	}
	(*out) << endl;
	if(opts.omeanpathlength.value()) (*out2) << endl;
      }
      delete out;
      if(opts.omeanpathlength.value()) delete out2;
    }
  }

  //////// SAVE PATHS /////////////
  if(opts.save_paths.value()){
    string filename=logger.appendDir("saved_paths.txt");
    ofstream of(filename.c_str());
    if (of.is_open()){
      for(unsigned int i=0;i<m_save_paths.size();i++){
	stringstream flot;
	flot << "# " << (m_save_paths[i].size()/3)<<endl;
	for(unsigned int j=0;j<m_save_paths[i].size();j=j+3){
	  flot << m_save_paths[i][j] << " " << m_save_paths[i][j+1] << " " << m_save_paths[i][j+2] << endl;
	}
	of<<flot.str();
      }
      of.close();
    }else{
      cerr<<"Counter::save_paths:error opening file for writing: "<<filename<<endl;
    }
  }
  // PATH DIRECTION //
  if(opts.opathdir.value()){
    volume4D<float> tmplocdir(m_prob->xsize(),m_prob->ysize(),m_prob->zsize(),3);
    copybasicproperties(*m_prob,tmplocdir);
    tmplocdir=0;
    SymmetricMatrix Tens(3);
    DiagonalMatrix D;Matrix V;
    for(int z=0;z<m_prob->zsize();z++){
      for(int y=0;y<m_prob->ysize();y++){
	for(int x=0;x<m_prob->xsize();x++){
	  if(m_prob[0](x,y,z)==0)continue;
	  //printf("xyz: %i %i %i: %f,%f,%f,%f,%f,%f \n",x,y,z,m_localdir[0](x,y,z,0),m_localdir[0](x,y,z,1),m_localdir[0](x,y,z,2),m_localdir[0](x,y,z,3),m_localdir[0](x,y,z,4),m_localdir[0](x,y,z,5));
	  Tens<<m_localdir[0](x,y,z,0)
	      <<m_localdir[0](x,y,z,1)
	      <<m_localdir[0](x,y,z,2)
	      <<m_localdir[0](x,y,z,3)
	      <<m_localdir[0](x,y,z,4)
	      <<m_localdir[0](x,y,z,5);
	  if(m_prob[0](x,y,z)!=0)
	    Tens=Tens/m_prob[0](x,y,z);
	  EigenValues(Tens,D,V);
	  for(int t=0;t<3;t++)
	    tmplocdir(x,y,z,t)=V(t+1,3);
	}
      }
    }
    tmplocdir.setDisplayMaximumMinimum(1,-1);
    save_volume4D(tmplocdir,logger.appendDir(opts.outfile.value()+"_localdir"));
  }
}
