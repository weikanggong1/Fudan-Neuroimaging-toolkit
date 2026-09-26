/*  xfibresOptions.h

    Saad Jbabdi, Tim Behrens, Stam Sotiropoulos, FMRIB Image Analysis Group

    Copyright (C) 1999-2010 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(xfibresOptions_h)
#define xfibresOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "utils/log.h"
#include "utils/tracer_plus.h"

namespace Xfibres {

class xfibresOptions {
 public:
  static xfibresOptions& getInstance();
  ~xfibresOptions() { delete gopt; }

  Utilities::Option<bool> verbose;
  Utilities::Option<bool> help;
  Utilities::Option<std::string> logdir;
  Utilities::Option<bool> forcedir;
  Utilities::Option<std::string> datafile;
  Utilities::Option<std::string> maskfile;
  Utilities::Option<std::string> bvecsfile;
  Utilities::Option<std::string> bvalsfile;
  Utilities::Option<int> nfibres;
  Utilities::Option<int> modelnum;
  Utilities::Option<float> fudge;
  Utilities::Option<int> njumps;
  Utilities::Option<int> nburn;
  Utilities::Option<int> nburn_noard;
  Utilities::Option<int> sampleevery;
  Utilities::Option<int> updateproposalevery;
  Utilities::Option<int> seed;
  Utilities::Option<bool> no_ard;
  Utilities::Option<bool> all_ard;
  Utilities::Option<bool> localinit;
  Utilities::Option<bool> nonlin;
  Utilities::Option<bool> cnonlin;
  Utilities::Option<bool> rician;
  Utilities::Option<bool> f0;
  Utilities::Option<bool> ardf0;
  Utilities::Option<float> R_prior_mean;  //setting the prior for model's 3 ratio of perp. to parallel diffusivity
  Utilities::Option<float> R_prior_std;
  Utilities::FmribOption<float> R_prior_fudge; //if used (set to a positive number), an ARD for R is used for the high diffusivity regions with the requested fudge factor
  Utilities::FmribOption<std::string> grad_file;


  void parse_command_line(int argc, char** argv,  Utilities::Log& logger);

 private:
  xfibresOptions();
  const xfibresOptions& operator=(xfibresOptions&);
  xfibresOptions(xfibresOptions&);

  Utilities::OptionParser options;

  static xfibresOptions* gopt;

};

 inline xfibresOptions& xfibresOptions::getInstance(){
   if(gopt == NULL)
     gopt = new xfibresOptions();

   return *gopt;
 }

 inline xfibresOptions::xfibresOptions() :
   verbose(std::string("-V,--verbose"), false,
	   std::string("switch on diagnostic messages"),
	   false, Utilities::no_argument),
   help(std::string("-h,--help"), false,
	std::string("display this message"),
	false, Utilities::no_argument),
   logdir(std::string("--ld,--logdir"), std::string("logdir"),
	  std::string("log directory (default is logdir)"),
	  false, Utilities::requires_argument),
   forcedir(std::string("--forcedir"),false,std::string("Use the actual directory name given - i.e. don't add + to make a new directory"),false,Utilities::no_argument),
   datafile(std::string("-k,--data,--datafile"), std::string("data"),
	    std::string("data file"),
	    true, Utilities::requires_argument),
   maskfile(std::string("-m,--mask, --maskfile"), std::string("nodif_brain_mask"),
	    std::string("mask file"),
	    true, Utilities::requires_argument),
   bvecsfile(std::string("-r,--bvecs"), std::string("bvecs"),
	     std::string("b vectors file"),
	     true, Utilities::requires_argument),
   bvalsfile(std::string("-b,--bvals"), std::string("bvals"),
	     std::string("b values file"),
	     true, Utilities::requires_argument),
   nfibres(std::string("--nf,--nfibres"),1,
	   std::string("Maximum number of fibres to fit in each voxel (default 1)"),
	   false,Utilities::requires_argument),
   modelnum(std::string("--model"),1,
	    std::string("Which model to use. 1=deconv. with sticks (default). 2=deconv. with sticks and a range of diffusivities. 3=deconv. with zeppelins"),
	    false,Utilities::requires_argument),
   fudge(std::string("--fudge"),1,
	 std::string("ARD fudge factor"),
	 false,Utilities::requires_argument),
   njumps(std::string("--nj,--njumps"),1250,
	  std::string("Num of jumps to be made by MCMC (default is 1250)"),
	  false,Utilities::requires_argument),
   nburn(std::string("--bi,--burnin"),1000,
	 std::string("Total num of jumps at start of MCMC to be discarded (default is 1000)"),
	 false,Utilities::requires_argument),
   nburn_noard(std::string("--bn,--burnin_noard"),0,
	       std::string("num of burnin jumps before the ard is imposed (default is 0)"),
	       false,Utilities::requires_argument),
   sampleevery(std::string("--se,--sampleevery"),25,
	       std::string("Num of jumps for each sample (MCMC) (default is 25)"),
	       false,Utilities::requires_argument),
   updateproposalevery(std::string("--upe,--updateproposalevery"),40,
		       std::string("Num of jumps for each update to the proposal density std (MCMC) (default is 40)"),
		       false,Utilities::requires_argument),
   seed(std::string("--seed"),8665904,std::string("seed for pseudo random number generator"),
	false,Utilities::requires_argument),
   no_ard(std::string("--noard"),false,std::string("Turn ARD off on all fibres"),
	  false,Utilities::no_argument),
   all_ard(std::string("--allard"),false,std::string("Turn ARD on on all fibres"),
	   false,Utilities::no_argument),
   localinit(std::string("--nospat"),false,std::string("Initialise with tensor, not spatially"),
	     false,Utilities::no_argument),
   nonlin(std::string("--nonlinear"),false,std::string("Initialise with nonlinear fitting"),
	  false,Utilities::no_argument),
   cnonlin(std::string("--cnonlinear"),false,std::string("Initialise with constrained nonlinear fitting"),
	  false,Utilities::no_argument),
   rician(std::string("--rician"),false,std::string("Use Rician noise modelling"),false,Utilities::no_argument),
   f0(std::string("--f0"),false,std::string("Add to the model an unattenuated signal compartment"),false,Utilities::no_argument),
   ardf0(std::string("--ardf0"),false,std::string("Use ard on f0"),false,Utilities::no_argument),
   R_prior_mean(std::string("--Rmean"),0.13,std::string("Set the prior mean for R of model 3 (default:0.13- Must be<0.5)"),false, Utilities::requires_argument),
   R_prior_std(std::string("--Rstd"),0.03,std::string("Set the prior standard deviation for R of model 3 (default:0.03)"),false, Utilities::requires_argument),
   R_prior_fudge(std::string("--Rfudge"),0,std::string("If set(>0), an ARD prior is used for R with the requested fudge factor"),false, Utilities::requires_argument),
   grad_file(std::string("--gradnonlin"), std::string("gradnonlin"),
	     std::string("Gradient Nonlinearity Tensor file"),
	     false, Utilities::requires_argument),
   options("xfibres","xfibres --help (for list of options)\n")
     {
       try {
       options.add(verbose);
       options.add(help);
       options.add(logdir);
       options.add(forcedir);
       options.add(datafile);
       options.add(maskfile);
       options.add(bvecsfile);
       options.add(bvalsfile);
       options.add(nfibres);
       options.add(modelnum);
       options.add(fudge);
       options.add(njumps);
       options.add(nburn);
       options.add(nburn_noard);
       options.add(sampleevery);
       options.add(updateproposalevery);
       options.add(seed);
       options.add(no_ard);
       options.add(all_ard);
       options.add(localinit);
       options.add(nonlin);
       options.add(cnonlin);
       options.add(rician);
       options.add(f0);
       options.add(ardf0);
       options.add(R_prior_mean);
       options.add(R_prior_std);
       options.add(R_prior_fudge);
       options.add(grad_file);
     }
     catch(Utilities::X_OptionError& e) {
       options.usage();
       std::cerr << std::endl << e.what() << std::endl;
     }
     catch(std::exception &e) {
       std::cerr << e.what() << std::endl;
     }

   }
}

#endif
