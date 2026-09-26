/*  probtrackxOptions.h

    Tim Behrens, Saad Jbabdi, FMRIB Image Analysis Group

    Copyright (C) 2010 University of Oxford  */

/*  CCOPYRIGHT */

#if !defined(probtrackxOptions_h)
#define probtrackxOptions_h

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "utils/log.h"

#include "commonopts.h"

namespace TRACT {

  class probtrackxOptions {
  public:
    static probtrackxOptions& getInstance();
    ~probtrackxOptions() { delete gopt; }

    Utilities::Option<int>              verbose;
    Utilities::Option<bool>             help;

    Utilities::Option<std::string>      basename;
    Utilities::Option<std::string>      outfile;
    Utilities::Option<std::string>      logdir;
    Utilities::Option<bool>             forcedir;

    Utilities::Option<std::string>      maskfile;
    Utilities::Option<std::string>      seedfile;

    Utilities::Option<bool>             simple;
    Utilities::Option<bool>             network;
    Utilities::Option<bool>             simpleout;
    Utilities::Option<bool>             pathdist;
    Utilities::Option<bool>             omeanpathlength;
    Utilities::Option<std::string>      pathfile;
    Utilities::Option<bool>             s2tout;
    Utilities::Option<bool>             s2tastext;
    Utilities::Option<bool>             opathdir;         // like fdt_paths but with average local tract orientation

    Utilities::Option<bool>             closestvertex;
    Utilities::Option<std::string>      targetfile;
    Utilities::Option<std::string>      waypoints;
    Utilities::Option<std::string>      waycond;
    Utilities::Option<bool>             wayorder;
    Utilities::Option<bool>             onewaycondition;  // apply waypoint conditions to half the tract
    Utilities::Option<std::string>      rubbishfile;
    Utilities::Option<std::string>      stopfile;
    Utilities::Option<std::string>      wtstopfiles;

    Utilities::Option<bool>             matrix1out;
    Utilities::Option<float>            distthresh1;
    Utilities::Option<bool>             matrix2out;
    Utilities::Option<std::string>      lrmask;
    Utilities::Option<bool>             matrix3out;
    Utilities::Option<std::string>      mask3;
    Utilities::Option<std::string>      lrmask3;
    Utilities::Option<float>            distthresh3;
    Utilities::Option<bool>             matrix4out;
    Utilities::Option<std::string>      mask4;
    Utilities::Option<std::string>      dtimask;

    Utilities::Option<std::string>      seeds_to_dti;
    Utilities::Option<std::string>      dti_to_seeds;
    Utilities::Option<std::string>      seedref;
    Utilities::Option<std::string>      meshspace;

    Utilities::Option<int>              nparticles;
    Utilities::Option<int>              nsteps;
    Utilities::Option<float>            steplength;


    Utilities::Option<float>            distthresh;
    Utilities::Option<float>            c_thr;
    Utilities::Option<float>            fibthresh;
    Utilities::Option<bool>             loopcheck;
    Utilities::Option<bool>             usef;
    Utilities::Option<bool>             modeuler;

    Utilities::Option<float>            sampvox;
    Utilities::Option<int>              randfib;
    Utilities::Option<int>              fibst;
    Utilities::Option<int>              rseed;

    // hidden options
    Utilities::FmribOption<std::string> prefdirfile;      // inside this mask, pick orientation closest to whatever is in here
    Utilities::FmribOption<std::string> skipmask;         // inside this mask, ignore data (inertia)
    Utilities::FmribOption<bool>        forcefirststep;   // always take at least one step
    Utilities::FmribOption<bool>        osampfib;         // not yet
    Utilities::FmribOption<bool>        onewayonly;       // in surface mode, track towards the brain (assumes surface normal points towards the brain)
    Utilities::FmribOption<bool>        save_paths;       // save paths to ascii file
    Utilities::FmribOption<std::string> locfibchoice;     // inside this mask, define local rules for fibre picking
    Utilities::FmribOption<std::string> loccurvthresh;    // inside this mask, define local curvature threshold
    Utilities::FmribOption<bool>        targetpaths;      // output separate fdt_paths for each target
    Utilities::FmribOption<bool>        noprobinterpol;   // turn off probabilistic interpolation

    void parse_command_line(int argc, char** argv,Utilities::Log& logger);
    void modecheck();
    void modehelp();
    void matrixmodehelp();
    void status();
  private:
    probtrackxOptions();
    const probtrackxOptions& operator=(probtrackxOptions&);
    probtrackxOptions(probtrackxOptions&);

    Utilities::OptionParser options;

    static probtrackxOptions* gopt;

  };


  inline probtrackxOptions& probtrackxOptions::getInstance(){
    if(gopt == NULL)
      gopt = new probtrackxOptions();

    return *gopt;
  }

  inline probtrackxOptions::probtrackxOptions() :
    verbose(std::string("-V,--verbose"), 0,
            std::string("Verbose level, [0-2]"),
            false, Utilities::requires_argument),
    help(std::string("-h,--help"), false,
         std::string("Display this message\n\n"),
         false, Utilities::no_argument),

    basename(std::string("-s,--samples"),"",
             std::string("Basename for samples files - e.g. 'merged'"),
             true, Utilities::requires_argument),

    outfile(std::string("-o,--out"), std::string("fdt_paths"),
            std::string("Output file (default='fdt_paths')"),
            false, Utilities::requires_argument),
    logdir(std::string("--dir"), std::string("logdir"),
           std::string("\tDirectory to put the final volumes in - code makes this directory - default='logdir'"),
           false, Utilities::requires_argument),
    forcedir(std::string("--forcedir"), false,
             std::string("Use the actual directory name given - i.e. don't add + to make a new directory\n\n"),
             false, Utilities::no_argument),

    maskfile(std::string("-m,--mask"),"",
             std::string("Bet binary mask file in diffusion space"),
             true, Utilities::requires_argument),
    seedfile(std::string("-x,--seed"),"",
             std::string("Seed volume or list (ascii text file) of volumes and/or surfaces"),
             true, Utilities::requires_argument),

    simple(std::string("--simple"),false,
           std::string("Track from a list of voxels (seed must be a ASCII list of coordinates)"),
           false, Utilities::no_argument),
    network(std::string("--network"), false,
            std::string("Activate network mode - only keep paths going through at least one of the other seed masks"),
            false, Utilities::no_argument),
    simpleout(std::string("--opd"), false,
              std::string("\tOutput path distribution"),
              false, Utilities::no_argument),
    pathdist(std::string("--pd"), false,
             std::string("\tCorrect path distribution for the length of the pathways"),
             false, Utilities::no_argument),
    omeanpathlength(std::string("--ompl"), false,
                    std::string("\tOutput mean path length from seed"),
                    false, Utilities::no_argument),
    pathfile(std::string("--fopd"), "",
             std::string("\tOther mask for binning tract distribution"),
             false, Utilities::requires_argument),
    s2tout(std::string("--os2t"), false,
           std::string("\tOutput seeds to targets"),
           false, Utilities::no_argument),
    s2tastext(std::string("--s2tastext"), false,
              std::string("Output seed-to-target counts as a text file (default in simple mode)"),
              false, Utilities::no_argument),
    opathdir(std::string("--opathdir"),false,
             std::string("Output average local tract orientation (tangent)\n\n"),
             false, Utilities::no_argument),

    closestvertex(std::string("--closestvertex"), false,
                  std::string("Count only nearest neighbour vertex when the face of a surface is crossed."),
                  false, Utilities::no_argument),
    targetfile(std::string("--targetmasks"),"",
               std::string("File containing a list of target masks - for seeds_to_targets classification"),
               false, Utilities::requires_argument),
    waypoints(std::string("--waypoints"), std::string(""),
              std::string("Waypoint mask or ascii list of waypoint masks - only keep paths going through ALL the masks"),
              false, Utilities::requires_argument),
    waycond(std::string("--waycond"),"AND",
            std::string("Waypoint condition. Either 'AND' (default) or 'OR'"),
            false, Utilities::requires_argument),
    wayorder(std::string("--wayorder"),false,
             std::string("Reject streamlines that do not hit waypoints in given order. Only valid if waycond=AND"),
             false,Utilities::no_argument),
    onewaycondition(std::string("--onewaycondition"),false,
                    std::string("Apply waypoint conditions to each half tract separately"),
                    false, Utilities::no_argument),
    rubbishfile(std::string("--avoid"), std::string(""),
                std::string("\tReject pathways passing through locations given by this mask"),
                false, Utilities::requires_argument),
    stopfile(std::string("--stop"), std::string(""),
             std::string("\tStop tracking at locations given by this mask file"),
             false, Utilities::requires_argument),
    wtstopfiles(std::string("--wtstop"), std::string(""),
                std::string("One mask or text file with mask names. Allow propagation within mask but terminate on exit. If multiple masks, non-overlapping volumes expected\n\n"),
                false, Utilities::requires_argument),

    matrix1out(std::string("--omatrix1"), false,
               std::string("Output matrix1 - SeedToSeed Connectivity"),
               false, Utilities::no_argument),
    distthresh1(std::string("--distthresh1"), 0,
                std::string("Discards samples (in matrix1) shorter than this threshold (in mm - default=0)"),
                false, Utilities::requires_argument),
    matrix2out(std::string("--omatrix2"), false,
               std::string("Output matrix2 - SeedToLowResMask"),
               false, Utilities::no_argument),
    lrmask(std::string("--target2"), std::string(""),
           std::string("Low resolution binary brain mask for storing connectivity distribution in matrix2 mode"),
           false, Utilities::requires_argument),
    matrix3out(std::string("--omatrix3"), false,
               std::string("Output matrix3 (NxN connectivity matrix)"),
               false, Utilities::no_argument),
    mask3(std::string("--target3"), "",
          std::string("Mask used for NxN connectivity matrix (or Nxn if lrtarget3 is set)"),
          false, Utilities::requires_argument),
    lrmask3(std::string("--lrtarget3"), "",
            std::string("Column-space mask used for Nxn connectivity matrix"),
            false, Utilities::requires_argument),
    distthresh3(std::string("--distthresh3"), 0,
                std::string("Discards samples (in matrix3) shorter than this threshold (in mm - default=0)"),
                false, Utilities::requires_argument),
    matrix4out(std::string("--omatrix4"), false,
               std::string("Output matrix4 - DtiMaskToSeed (special Oxford Sparse Format)"),
               false, Utilities::no_argument),
    mask4(std::string("--colmask4"), std::string(""),
          std::string("Mask for columns of matrix4 (default=seed mask)"),
          false, Utilities::requires_argument),
    dtimask(std::string("--target4"), std::string(""),
            std::string("Brain mask in DTI space\n\n"),
            false, Utilities::requires_argument),

    seeds_to_dti(std::string("--xfm"),"",
                 std::string("\tTransform taking seed space to DTI space (either FLIRT matrix or FNIRT warpfield) - default is identity"),
                 false, Utilities::requires_argument),
    dti_to_seeds(std::string("--invxfm"), std::string(""),
                 std::string("Transform taking DTI space to seed space (compulsory when using a warpfield for seeds_to_dti)"),
                 false, Utilities::requires_argument),
    seedref(std::string("--seedref"),"",
            std::string("Reference vol to define seed space in simple mode - diffusion space assumed if absent"),
            false, Utilities::requires_argument),
    meshspace(std::string("--meshspace"), std::string("caret"),
              std::string("Mesh reference space - either 'caret' (default) or 'freesurfer' or 'first' or 'vox' \n\n"),
              false, Utilities::requires_argument),

    nparticles(std::string("-P,--nsamples"), 5000,
               std::string("Number of samples - default=5000"),
               false, Utilities::requires_argument),
    nsteps(std::string("-S,--nsteps"), 2000,
           std::string("Number of steps per sample - default=2000"),
           false, Utilities::requires_argument),
    steplength(std::string("--steplength"), 0.5,
               std::string("Steplength in mm - default=0.5\n\n"),
               false, Utilities::requires_argument),

    distthresh(std::string("--distthresh"), 0,
               std::string("Discards samples shorter than this threshold (in mm - default=0)"),
               false, Utilities::requires_argument),
    c_thr(std::string("-c,--cthr"), 0.2,
          std::string("Curvature threshold - default=0.2"),
          false, Utilities::requires_argument),
    fibthresh(std::string("--fibthresh"), 0.01,
              std::string("Volume fraction before subsidary fibre orientations are considered - default=0.01"),
              false, Utilities::requires_argument),
    loopcheck(std::string("-l,--loopcheck"), false,
              std::string("Perform loopchecks on paths - slower, but allows lower curvature threshold"),
              false, Utilities::no_argument),
    usef(std::string("-f,--usef"), false,
         std::string("Use anisotropy to constrain tracking"),
         false, Utilities::no_argument),
    modeuler(std::string("--modeuler"), false,
             std::string("Use modified euler streamlining\n\n"),
             false, Utilities::no_argument),


    sampvox(std::string("--sampvox"), 0,
            std::string("Sample random points within a sphere with radius x mm from the center of the seed voxels (e.g. --sampvox=0.5, 0.5 mm radius sphere). Default=0"),
            false, Utilities::requires_argument),
    randfib(std::string("--randfib"), 0,
            std::string("Default 0. Set to 1 to randomly sample initial fibres (with f > fibthresh). \n                        Set to 2 to sample in proportion fibres (with f>fibthresh) to f. \n                        Set to 3 to sample ALL populations at random (even if f<fibthresh)"),
            false, Utilities::requires_argument),
    fibst(std::string("--fibst"),1,
          std::string("\tForce a starting fibre for tracking - default=1, i.e. first fibre orientation. Only works if randfib==0"),
          false, Utilities::requires_argument),
    rseed(std::string("--rseed"), 12345,
          std::string("\tRandom seed"),
          false, Utilities::requires_argument),


    prefdirfile(std::string("--prefdir"), std::string(""),
                std::string("Prefered orientation preset in a 4D mask"),
                false, Utilities::requires_argument),
    skipmask(std::string("--no_integrity"), std::string(""),
             std::string("No explanation needed"),
             false, Utilities::requires_argument),
    forcefirststep(std::string("--forcefirststep"),false,
                   std::string("In case seed and (stop or exclusion) masks are the same"),
                   false, Utilities::no_argument),
    osampfib(std::string("--osampfib"),false,
             std::string("Output sampled fibres"),
             false, Utilities::no_argument),
    onewayonly(std::string("--onewayonly"),false,
               std::string("Track in one direction only (towards the brain - only valid for surface seeds)"),
               false, Utilities::no_argument),
    save_paths(std::string("--savepaths"),false,
               std::string("Save path coordinates to ascii file"),
               false, Utilities::no_argument),
    locfibchoice(std::string("--locfibchoice"),std::string(""),
                 std::string("Local rules for fibre choice - 0=closest direction(default), 1=equal prob (f>thr), 2=equal prob with angle threshold (=40 deg), 3=sample in proportion to f"),
                 false, Utilities::requires_argument),
    loccurvthresh(std::string("--loccurvthresh"),std::string(""),
                  std::string("Local curvature threshold"),
                  false, Utilities::requires_argument),
    targetpaths(std::string("--otargetpaths"),false,
                std::string("Output separate fdt_paths for targets (assumes --os2t is on)"),
                false, Utilities::no_argument),
    noprobinterpol(std::string("--noprobinterpol"),false,
                   std::string("Turn off probabilistic interpolation"),
                   false, Utilities::no_argument),

    options("probtrackx","probtrackx2 -s <basename> -m <maskname> -x <seedfile> -o <output> --targetmasks=<textfile>\n probtrackx2 --help\n")
  {


    try {
      options.add(verbose);
      options.add(help);

      options.add(basename);
      options.add(outfile);
      options.add(logdir);
      options.add(forcedir);

      options.add(maskfile);
      options.add(seedfile);

      options.add(simple);
      options.add(network);
      options.add(simpleout);
      options.add(pathdist);
      options.add(omeanpathlength);
      options.add(pathfile);
      options.add(s2tout);
      options.add(s2tastext);
      options.add(opathdir);

      options.add(closestvertex);
      options.add(targetfile);
      options.add(waypoints);
      options.add(waycond);
      options.add(wayorder);
      options.add(onewaycondition);
      options.add(rubbishfile);
      options.add(stopfile);
      options.add(wtstopfiles);

      options.add(matrix1out);
      options.add(distthresh1);
      options.add(matrix2out);
      options.add(lrmask);
      options.add(matrix3out);
      options.add(mask3);
      options.add(lrmask3);
      options.add(distthresh3);
      options.add(matrix4out);
      options.add(mask4);
      options.add(dtimask);

      options.add(seeds_to_dti);
      options.add(dti_to_seeds);
      options.add(seedref);
      options.add(meshspace);


      options.add(nparticles);
      options.add(nsteps);
      options.add(steplength);

      options.add(distthresh);
      options.add(c_thr);
      options.add(fibthresh);
      options.add(loopcheck);
      options.add(usef);
      options.add(modeuler);

      options.add(sampvox);
      options.add(randfib);
      options.add(fibst);
      options.add(rseed);

      options.add(skipmask);
      options.add(prefdirfile);
      options.add(forcefirststep);
      options.add(osampfib);
      options.add(onewayonly);
      options.add(save_paths);
      options.add(locfibchoice);
      options.add(loccurvthresh);
      options.add(targetpaths);
      options.add(noprobinterpol);

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
