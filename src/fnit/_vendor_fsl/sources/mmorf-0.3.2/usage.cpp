// This is how I imagine MMORF would be used for a single volume:

MMORF::Volume* v_ref = MMORF::read_volume('reference_volume');
MMORF::Volume* v_in = MMORF::read_volume('input_volume');
MMORF::WarpField* w_initial = MMORF::read_warp('initial_guess');
int subsampling = subsample_level;
MMORF::CostFunction* ssd_cf = new MMORF::SumOfSquares;
MMORF::Regulariser* bending = new MMORF::BendingEnergy;
MMORF::ConvergenceTester conv(absolute_error,relative_error,iterations);
MMORF::Optimiser* gn_opt = new MMORF::GaussNewtonOptimiser



MMORF::Registrant<volume>* my_reg_obj = new MMORF::Registrant<volume>(
                                                    *v_ref,
                                                    *v_in,
                                                    *w_initial,
                                                    subsample_level,
                                                    *ssd_cf,
                                                    *bending,
                                                    conv,
                                                    *gn_opt);

my_reg_obj->perform_registration();
MMORF::WarpField* w_final = my_reg_obj->get_warp();

// A thought. Registration isn't registration. It's optimisation. Using a cost function based
// on volumes. This means it is entirely valid to set up a cost function using volumes (or
// anything else for that matter), and then evaluate it at a point in parameter space

// I think it's worth abstracting away a lot of my dependencies, even those that currently
// exist in FSL. The motivation behinf this is that even using NEWIMAGE::volume objects
// makes me heavily reliant on those. So I will wrap them all in interfaces, even if all I do
// is pass calls through to existing objects.

// Classes I need:
MMORF::Volume
    -
MMORF::CostFunction
