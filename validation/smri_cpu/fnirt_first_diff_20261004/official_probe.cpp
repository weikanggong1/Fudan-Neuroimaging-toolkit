// Isolated benchmark oracle. Uses installed, licensed FSL source/libraries;
// FNIT production never loads or invokes this program.
#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>
#include "newimage/newimageall.h"
#include "miscmaths/nonlin.h"
#include "miscmaths/miscmaths.h"
#include "fnirt_costfunctions.h"
#include "fnirtfns.h"

using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace FNIRT;
using namespace MISCMATHS;

class Probe : public SSD_fnirt_CF {
public:
  using SSD_fnirt_CF::SSD_fnirt_CF;
  void dump(const std::string& prefix) {
    save_volume(Ref(), prefix+"_fixed");
    save_volume(Robj(), prefix+"_warped");
    save_volume(Mask(), prefix+"_mask");
    save_volume4D(Deriv(), prefix+"_derivative");
    SaveDefCoefs(prefix+"_coefficients");
    SaveGlobalIntensityMapping(prefix+"_scale.txt");
  }
};

static void vector_file(const std::string& name, const ColumnVector& vector) {
  std::ofstream file(name);
  file << std::setprecision(17);
  for (int i=1; i<=vector.Nrows(); ++i) file << vector(i) << "\n";
}

int main(int argc, char** argv) {
  auto options = parse_fnirt_command_line(argc, argv);
  volume<float> fixed, moving;
  read_volume(fixed, options->Ref());
  read_volume(moving, options->Obj());
  auto fixed_mask = make_mask(options->RefMask(), options->UseRefMask(1) ? InclusiveMask : IgnoreMask,
                             fixed, options->UseImplicitRefMask(), options->ImplicitRefValue());
  const double fixed_mean = spmlike_mean(fixed), moving_mean = spmlike_mean(moving);
  fixed *= 100.0 / fixed_mean;
  moving *= 100.0 / moving_mean;
  auto moving_mask = make_mask(options->ObjMask(), options->UseObjMask(1) ? InclusiveMask : IgnoreMask,
                              moving, options->UseImplicitObjMask(), options->ImplicitObjValue());
  save_volume(fixed, "normalized_fixed");
  save_volume(moving, "normalized_moving");
  // In this fixed GM recipe both implicit input and explicit input masks are absent.
  save_volume(smooth(moving, options->ObjFWHM(1)/std::sqrt(8.0*std::log(2.0))), "smoothed_moving");
  auto fields = init_warpfield(*options);
  auto mapper = init_intensity_mapper(*options);
  Probe cost(fixed, moving, options->Affine(), fields, mapper);
  cost.SetRegularisationModel(options->RegularisationModel());
  cost.SetLambda(options->Lambda(1));
  cost.SetIntensityMappingFixed(!options->EstimateIntensity(1));
  cost.SetHessianPrecision(options->HessianPrecision());
  cost.SetInterpolationModel(options->InterpolationModel());
  cost.WeightLambdaBySSD(options->WeightLambdaBySSD());
  if (options->UseRefDeriv()) cost.UseRefDerivs();
  cost.SmoothRef(options->RefFWHM(1));
  cost.SmoothObj(options->ObjFWHM(1));
  cost.SubsampleRef(options->SubSampling(1));
  if (fixed_mask) cost.SetRefMask(*fixed_mask);
  if (moving_mask) cost.SetObjMask(*moving_mask);
  auto initial = cost.Par();
  const double initial_cost = cost.cf(initial);
  cost.dump("initial");
  auto gradient = cost.grad(initial);
  vector_file("initial_gradient.txt", gradient);
  auto hessian = cost.hess(initial);
  ColumnVector diagonal(initial.Nrows()), probe(initial.Nrows());
  for (int i=1; i<=initial.Nrows(); ++i) {
    diagonal(i)=hessian->Peek(i,i);
    probe(i)=std::sin(0.017*i);
  }
  vector_file("initial_diagonal.txt", diagonal);
  vector_file("initial_hessian_probe.txt", hessian->MulByVec(probe));
  NonlinParam optimizer(cost.NPar(), options->MinimisationMethod(), initial);
  set_nlpars(optimizer);
  optimizer.SetMaxIter(1);
  nonlin(optimizer, cost);
  const double accepted_cost = cost.cf(optimizer.Par());
  cost.dump("accepted");
  vector_file("accepted_parameters.txt", optimizer.Par());
  std::ofstream scalars("scalars.txt");
  scalars << std::setprecision(17) << "fixed_mean " << fixed_mean << "\nmoving_mean " << moving_mean
          << "\ninitial_cost " << initial_cost << "\naccepted_cost " << accepted_cost
          << "\nstatus " << optimizer.TextStatus() << "\n";
  return 0;
}
