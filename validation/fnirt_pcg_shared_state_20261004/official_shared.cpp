// Validation-only bounded real shared-state probe. Production never calls FSL.
// The installed CG header remains unchanged; wrappers observe its vector calls.
#include <fstream>
#include <iomanip>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include "newimage/newimageall.h"
#include "miscmaths/nonlin.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/cg.h"
#include "fnirt_costfunctions.h"
#include "fnirtfns.h"
using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace FNIRT;
using namespace MISCMATHS;

class Probe : public SSD_fnirt_CF {
public:
  using SSD_fnirt_CF::SSD_fnirt_CF;
  double effective_lambda() const { return Lambda(); }
};
static void binary(const std::string& name,const ColumnVector& v) {
  std::ofstream f(name,std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.Store()),v.Nrows()*sizeof(double));
}
static void matrix(const std::string& name,const BFMatrix& h,int n) {
  std::ofstream f(name,std::ios::binary);
  for(int r=1;r<=n;++r) for(int c=1;c<=n;++c) {
    double v=h.Peek(r,c);f.write(reinterpret_cast<const char*>(&v),sizeof(double));
  }
}
struct IterationTrace {
  std::string prefix;
  ColumnVector r,z;
  std::ofstream csv;
  double normb,after;
  int iteration=0;
  IterationTrace(std::string p,double norm):prefix(p),csv(p+"_trace.csv"),normb(norm),after(1) {
    csv<<std::setprecision(17)<<"iteration,relative_before,relative_after,rho,denominator,alpha\n";
  }
};
class TraceDiagonal {
  ColumnVector diagonal;
  std::shared_ptr<IterationTrace> trace;
public:
  TraceDiagonal(ColumnVector d,std::shared_ptr<IterationTrace> t):diagonal(d),trace(t) {}
  ColumnVector solve(const ColumnVector& r) const {
    ColumnVector z(r.Nrows());
    for(int i=1;i<=r.Nrows();++i) z(i)=r(i)/diagonal(i);
    trace->r=r;trace->z=z;++trace->iteration;
    binary(trace->prefix+"_r_"+std::to_string(trace->iteration)+".f64",r);
    binary(trace->prefix+"_z_"+std::to_string(trace->iteration)+".f64",z);
    return z;
  }
};
class TraceMatrix {
  const BFMatrix& h;
  std::shared_ptr<IterationTrace> trace;
public:
  TraceMatrix(const BFMatrix& matrix,std::shared_ptr<IterationTrace> t):h(matrix),trace(t) {}
  ColumnVector operator*(const ColumnVector& p) const {
    ColumnVector q=h.MulByVec(p);
    if(trace->iteration) {
      const double rho=DotProduct(trace->r,trace->z),den=DotProduct(p,q),alpha=rho/den;
      ColumnVector next=trace->r-alpha*q;
      trace->after=next.NormFrobenius()/trace->normb;
      trace->csv<<trace->iteration<<","<<trace->r.NormFrobenius()/trace->normb<<","<<trace->after
                <<","<<rho<<","<<den<<","<<alpha<<"\n";
      binary(trace->prefix+"_p_"+std::to_string(trace->iteration)+".f64",p);
      binary(trace->prefix+"_q_"+std::to_string(trace->iteration)+".f64",q);
      binary(trace->prefix+"_r_after_"+std::to_string(trace->iteration)+".f64",next);
    }
    return q;
  }
};
int main(int argc,char**argv) {
  auto options=parse_fnirt_command_line(argc,argv);
  volume<float> fixed,moving;read_volume(fixed,options->Ref());read_volume(moving,options->Obj());
  auto fm=make_mask(options->RefMask(),options->UseRefMask(1)?InclusiveMask:IgnoreMask,fixed,
                    options->UseImplicitRefMask(),options->ImplicitRefValue());
  fixed*=100.0/spmlike_mean(fixed);moving*=100.0/spmlike_mean(moving);
  auto mm=make_mask(options->ObjMask(),options->UseObjMask(1)?InclusiveMask:IgnoreMask,moving,
                    options->UseImplicitObjMask(),options->ImplicitObjValue());
  auto fields=init_warpfield(*options);auto mapper=init_intensity_mapper(*options);
  Probe cost(fixed,moving,options->Affine(),fields,mapper);
  cost.SetRegularisationModel(options->RegularisationModel());cost.SetLambda(options->Lambda(1));
  cost.SetIntensityMappingFixed(!options->EstimateIntensity(1));cost.SetHessianPrecision(options->HessianPrecision());
  cost.SetInterpolationModel(options->InterpolationModel());cost.WeightLambdaBySSD(options->WeightLambdaBySSD());
  if(options->UseRefDeriv()) cost.UseRefDerivs();
  cost.SmoothRef(options->RefFWHM(1));cost.SmoothObj(options->ObjFWHM(1));cost.SubsampleRef(options->SubSampling(1));
  if(fm) cost.SetRefMask(*fm);if(mm) cost.SetObjMask(*mm);
  const char* path=std::getenv("FNIT_PROBE_PARAMETERS");if(!path) return 2;
  ColumnVector parameters(cost.NPar());std::ifstream f(path);
  for(int i=1;i<=parameters.Nrows();++i) if(!(f>>parameters(i))) return 3;
  // Continue from the saved first accepted vector, with the LM success update
  // 0.1/10. This is a bounded continuation, not the archived stock trajectory.
  double current_cost=cost.cf(parameters),damping=.01,old_damping=0;
  bool success=true;ColumnVector gradient;std::shared_ptr<BFMatrix> hessian;
  std::ofstream report("state.json");report<<std::setprecision(17)<<"{\"dimension\":"<<parameters.Nrows()<<",\"states\":[";
  for(int attempt=2;attempt<=3;++attempt) {
    std::string name="solve"+std::to_string(attempt);
    binary(name+"_parameters.f64",parameters);
    if(success) {gradient=cost.grad(parameters);hessian=cost.hess(parameters,hessian);}
    const double regularization=cost.effective_lambda();
    matrix(name+"_H_before_nudge.f64",*hessian,parameters.Nrows());
    for(int i=1;i<=parameters.Nrows();++i)
      hessian->Set(i,i,((1.0+damping)/(1.0+old_damping))*hessian->Peek(i,i));
    ColumnVector diagonal(parameters.Nrows()),zero(parameters.Nrows());zero=0;
    for(int i=1;i<=parameters.Nrows();++i) diagonal(i)=hessian->Peek(i,i);
    matrix(name+"_A.f64",*hessian,parameters.Nrows());binary(name+"_rhs.f64",gradient);
    binary(name+"_diagonal.f64",diagonal);binary(name+"_x0.f64",zero);
    ColumnVector direct=hessian->SolveForx(gradient,SYM_POSDEF,1e-3,500),observed=zero;
    auto trace=std::make_shared<IterationTrace>(name,gradient.NormFrobenius());
    TraceMatrix a(*hessian,trace);TraceDiagonal preconditioner(diagonal,trace);
    int iterations=500;double tolerance=1e-3;
    int status=CG(a,observed,gradient,preconditioner,iterations,tolerance);
    double maximum=0;for(int i=1;i<=direct.Nrows();++i) maximum=std::max(maximum,std::abs(direct(i)-observed(i)));
    binary(name+"_native_solution.f64",direct);binary(name+"_observed_solution.f64",observed);
    if(maximum!=0 || tolerance!=trace->after) throw std::runtime_error("CG observation differs from unchanged native solve");
    const double trial_cost=cost.cf(parameters-direct);success=trial_cost<current_cost;
    if(attempt>2) report<<",";
    report<<"{\"solve\":"<<attempt<<",\"lm_damping\":"<<damping<<",\"old_damping\":"<<old_damping
          <<",\"regularization_lambda\":"<<regularization<<",\"cost_before\":"<<current_cost
          <<",\"trial_cost\":"<<trial_cost<<",\"accepted\":"<<(success?"true":"false")
          <<",\"native_cg_status\":"<<status<<",\"native_cg_iterations\":"<<iterations
          <<",\"native_relative_residual\":"<<tolerance<<",\"observer_max_abs\":"<<maximum<<"}";
    if(success) {parameters-=direct;current_cost=trial_cost;old_damping=0;damping/=10;}
    else {old_damping=damping;damping*=10;}
  }
  report<<"],\"scope\":\"two bounded coarsest-level continuation solves from saved first accepted FP64 parameters, no full registration\"}\n";
  binary("final_parameters.f64",parameters);
  return 0;
}
