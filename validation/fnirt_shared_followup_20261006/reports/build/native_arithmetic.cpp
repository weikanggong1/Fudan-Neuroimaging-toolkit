
// Own validation bridge; no upstream implementation is copied.
#include <algorithm>
#include <cmath>
#include "armawrap/newmat.h"
extern "C" double fnit_native_dot(const double* a,const double* b,int n) {
  NEWMAT::ColumnVector x(n),y(n);
  std::copy(a,a+n,x.Store());std::copy(b,b+n,y.Store());
  return NEWMAT::DotProduct(x,y);
}
extern "C" double fnit_native_norm(const double* a,int n) {
  NEWMAT::ColumnVector x(n);std::copy(a,a+n,x.Store());
  return x.NormFrobenius();
}
extern "C" int fnit_native_flags() {
  int flags=0;
#ifdef ARMA_USE_BLAS
  flags|=1;
#endif
#ifdef ARMA_USE_ATLAS
  flags|=2;
#endif
#if defined(__FINITE_MATH_ONLY__) && (__FINITE_MATH_ONLY__ > 0)
  flags|=4;
#endif
  return flags;
}
