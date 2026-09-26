#ifndef __SOLVER_MULT_INVERSE_CUH__
#define __SOLVER_MULT_INVERSE_CUH__


namespace Xfibres {

__device__ void solver(
  //INPUT
  float* A,
  float* P,
  int    length,
  //TO USE
  float* C,
  float* el,
  int*   indx,
  //OUTPUT
  float* B);


} // namespace Xfibres

#endif
