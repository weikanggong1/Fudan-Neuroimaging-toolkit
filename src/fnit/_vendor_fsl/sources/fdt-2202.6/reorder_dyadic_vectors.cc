/*  reorder_dyadic_vectors.cc

    Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2006 University of Oxford */

/*  CCOPYRIGHT */

#include "utils/options.h"
#define QUEUEHDR <queue>
#if __GNUC__
  #define GCC_VERSION (__GNUC__ * 100000 \
                       + __GNUC_MINOR__ * 100 \
		       + __GNUC_PATCHLEVEL__)
  #if GCC_VERSION < 40300
    #define QUEUEHDR <heap.h>
  #endif
#endif
#include QUEUEHDR
#include "newimage/newimageall.h"
#include "miscmaths/miscprob.h"
#include "stdlib.h"

#define AP 2
#define NB 1
#define FA 0


using namespace Utilities;
using namespace std;
using namespace NEWIMAGE;
using namespace MISCMATHS;


string title="reorder_dyadic_vectors \nReordering of the dyadic vectors with direction preservation";
string examples="reorder_dyadic_vectors -b <dirname> [-m mask]\n<dirname> is supposed to contain mean_fisamples and dyadsi for i=1:n";

Option<bool> verbose(string("-v,--verbose"),false,
		       string("switch on diagnostic messages"),
		       false,no_argument);
Option<bool> help(string("-h,--help"),false,
		       string("display this message"),
		       false,no_argument);
Option<string> basename(string("-b,--basename"),string(""),
		       string("output basename"),
		       true,requires_argument);
Option<string> mask(string("-m,--mask"),string(""),
		       string("filename of brain mask"),
		       false,requires_argument);
////////////////////////////////////////////////////////

class FM{
 private:
  volume<float> mask,tmap,state;
  vector< volume4D<float> > dyads;
  vector< volume<float> >  meanf;
  int nx,ny,nz;
  Matrix P;
  Matrix neighbours;

  Option<string>& obasename;
  Option<string>& omask;

 public:
  FM(Option<string>& optbasename,Option<string>& optmask):obasename(optbasename),omask(optmask){

    int fib=1;
    bool fib_existed=true;

    cout<<"reading the data"<<endl;
    while(fib_existed){
      if(fsl_imageexists(obasename.value()+"/mean_f"+num2str(fib)+"samples")){
	cout<<"found something to read"<<endl;
	volume4D<float> *tmpdptr=new volume4D<float>;
	volume<float> *tmpfptr=new volume<float>;
	read_volume4D(*tmpdptr,obasename.value()+"/dyads"+num2str(fib));
	dyads.push_back(*tmpdptr);
	read_volume(*tmpfptr,obasename.value()+"/mean_f"+num2str(fib)+"samples");
	meanf.push_back(*tmpfptr);
	fib++;
      }
      else
	fib_existed=false;
    }
    cout<<"data read"<<endl;

    // read input volumes
    nx=dyads[0].xsize();
    ny=dyads[0].ysize();
    nz=dyads[0].zsize();

    //OUT(nx);
    //OUT(ny);
    //OUT(nz);

    // create other volumes
    if(omask.set()){
      read_volume(mask,omask.value());
    }
    else{
      mask.reinitialize(nx,ny,nz);
      copybasicproperties(dyads[0],mask);
      mask=1;
    }

    cout<<"state and mask"<<endl;
    state.reinitialize(nx,ny,nz);
    tmap.reinitialize(nx,ny,nz);

    P=MISCMATHS::perms(dyads.size());
    //    OUT(P);

    neighbours.ReSize(3,26);
    neighbours << 1 << 0 << 0 << -1 << 0 << 0 << 1 << 1 <<-1 <<-1 << 1 <<-1 << 1 <<-1 << 0 << 0 << 0 << 0 << 1 <<-1 << 1 << 1 <<-1 <<-1 << 1 <<-1 
	       << 0 << 1 << 0 << 0  <<-1 << 0 << 1 <<-1 << 1 <<-1 << 0 << 0 << 0 << 0 << 1 <<-1 << 1 <<-1 << 1 << 1 <<-1 << 1 <<-1 << 1 <<-1 <<-1
	       << 0 << 0 << 1 << 0  << 0 <<-1 << 0 << 0 << 0 << 0 << 1 << 1 <<-1 <<-1 << 1 << 1 <<-1 <<-1 << 1 << 1 << 1 <<-1 << 1 <<-1 <<-1 <<-1;
  }
  
  ////////////////////// functions
  void do_fast_marching(){
    int i0,j0,k0;                      /* current point coords */
    float tval; 
    
    /********************************/
    /*** performing fmt algorithm ***/
    /********************************/
    
    /*** initialization ***/
    cout<<"initialise"<<endl;
    /* look for bigger f1+f2 point as a seed */
    float maxf=0,curf;
    //OUT(nx);OUT(ny);OUT(nz);
    for(int z=0;z<nz;z++)
      for(int y=0;y<ny;y++)
	for(int x=0;x<nx;x++){
	  if(mask(x,y,z)==0)
	    continue;
	  curf=0;
	  for(unsigned int f=0;f<meanf.size();f++)
	    curf += meanf[f](x,y,z);
	  if(curf>maxf){
	    i0=x;j0=y;k0=z;
	    maxf=curf;
	  }
	  state(x,y,z)=FA;
	  tmap(x,y,z)=1;
	}
    state(i0,j0,k0)=AP;
    tmap(i0,j0,k0)=0;
    
     /* create heap sort structure */
    Heap h(nx,ny,nz);

    //cout<<"nbvalue1"<<endl;
    /* and all points of the ROIs as Alive Points */
    updateNBvalue(i0,j0,k0,h);
    //h.print();

    //return;
    /*--------------------------------------------------------*/
    /*** big loop ***/
    cout<<"start FM"<<endl;
    int STOP = 0;
    int counter = 0;
    while(STOP==0){
      /*break;*/
      counter++;
      //OUT(counter);
      
      /*** uses the heap sort structure to find the NB point with the smallest T-value ***/
      h.heapRemove(tval,i0,j0,k0);
      //cout << i0 << " " << j0 << " " << k0 << endl;
      
      /*** add this point to the set of alive points ***/
      state(i0,j0,k0)=AP;
      
      if(h.get_N()==0)
	break;
      
      /*** update narrow band's T-value ***/
      updateNBvalue(i0,j0,k0,h);
      
    }
    
  }
  bool isInside(int i,int j,int k){
    //cout<<"je suis dans isInside"<<endl;
    //OUT(i);OUT(j);OUT(k);
    //OUT(mask(i,j,k));
    return((i>=0) && (i<nx)
	   && (j>=0) && (j<ny) 
	   && (k>=0) && (k<nz) 
	   && (mask(i,j,k)!=0));
  }
  void computeT(int x,int y,int z){
    Matrix V(P.Ncols(),3);
    Matrix nV(P.Ncols(),3);
    ColumnVector mF(P.Ncols());

    V=get_vector(x,y,z);
    mF=get_f(x,y,z);

    //cout << "NB voxel: ";
    //cout << x << " " << y << " " << z << endl;

    int nbx,nby,nbz;
    int opt_perm=1;
    float opt_sperm=0;
    for(int per=1;per<=P.Nrows();per++){
      //OUT(per);
      float opt_nb=0;int nnb=0;
      for(int nb=1;nb<=neighbours.Ncols();nb++){
	//OUT(nb);
	nbx=x+neighbours(1,nb);nby=y+neighbours(2,nb);nbz=z+neighbours(3,nb);
	if(!isInside(nbx,nby,nbz))
	  continue;
	if(state(nbx,nby,nbz)==AP){
	  //cout<<"this neighbour is an AP"<<endl;
	  nV=get_vector(nbx,nby,nbz);
	  
	  float opt_s=0,s;
	  for(int f=1;f<=V.Nrows();f++){
	    s=abs(dot(nV.Row(f).t(),V.Row(P(per,f)).t()));
	    if(s>opt_s){
	      opt_s=s;
	    }
	  }//f
	  opt_nb+=opt_s;
	  nnb++;
	}//endif
	
      }//nb

      opt_nb/=nnb;
      //OUT(opt_nb);
	
      if(opt_nb>opt_sperm){
	opt_sperm=opt_nb;
	opt_perm=per;
      }

    }//perm
    tmap(x,y,z)=1-opt_sperm; // store optimal mean scalar product

    // reorder dyads
    for(unsigned int f=0;f<dyads.size();f++){
      dyads[f](x,y,z,0)=V(P(opt_perm,f+1),1);
      dyads[f](x,y,z,1)=V(P(opt_perm,f+1),2);
      dyads[f](x,y,z,2)=V(P(opt_perm,f+1),3);

      meanf[f](x,y,z)=mF(P(opt_perm,f+1));
    }
  }

  ReturnMatrix get_vector(int x,int y,int z){
    Matrix V(P.Ncols(),3);
    for(unsigned int f=0;f<dyads.size();f++)
      for(int i=1;i<=3;i++)
	V(f+1,i)=dyads[f](x,y,z,i-1);

    V.Release();
    return V;
  }

  ReturnMatrix get_f(int x,int y,int z){
    ColumnVector V(P.Ncols());
    for(unsigned int f=0;f<dyads.size();f++)
      V(f+1)=meanf[f](x,y,z);

    V.Release();
    return V;
  }

  void updateNBvalue(int i,int j,int k,Heap& h){    
    int ni,nj,nk;
    int pos;
    double val;
   
    for(int nb=1;nb<=neighbours.Ncols();nb++){
      ni=i+neighbours(1,nb);nj=j+neighbours(2,nb);nk=k+neighbours(3,nb);
      if(isInside(ni,nj,nk)){
	if(state(ni,nj,nk)==AP)continue;
	computeT(ni,nj,nk);
	val=tmap(ni,nj,nk);
	/* update value if in NB */
	if(state(ni,nj,nk)==NB){
	  pos=h.get_bpr(ni,nj,nk);
	  h.set_val(pos,MIN(h.get_val(pos),val));
	  h.heapUp(pos);
	}
	/* insert value if in FA */
	else{
	  state(ni,nj,nk)=NB;
	  h.heapInsert(val,ni,nj,nk);
	}
      }
      
    }
  }
  
  void save_results(){
    int fib=1;
    for(unsigned int f=0;f<dyads.size();f++){
      save_volume4D(dyads[f],obasename.value()+"/reordered_dyads"+num2str(fib));
      save_volume(meanf[f],obasename.value()+"/reordered_meanf"+num2str(fib)+"samples");
      fib++;
    }
  }

};




int main(int argc,char *argv[]){

  Tracer tr("main");
  OptionParser options(title,examples);

  try{
    options.add(verbose);
    options.add(help);
    options.add(basename);
    options.add(mask);

    options.parse_command_line(argc,argv);

    
    if ( (help.value()) || (!options.check_compulsory_arguments(true)) ){
      options.usage();
      exit(EXIT_FAILURE);
    }

    if(verbose.value())
      cout<<"Call for fast marching"<<endl;
    FM fm(basename,mask);
  
    if(verbose.value())
      cout<<"perform fast marching"<<endl;
    fm.do_fast_marching();

    if(verbose.value())
      cout<<"save results"<<endl;
    fm.save_results();



  }
  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } 
  catch(std::exception &e) {
    cerr << e.what() << endl;
  } 

  return 0;
  
  
}
