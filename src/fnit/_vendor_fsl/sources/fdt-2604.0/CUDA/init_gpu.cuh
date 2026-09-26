/*  init_gpu.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __INIT_GPU_CUH__
#define __INIT_GPU_CUH__


namespace Xfibres {

void init_gpu();

double timeval_diff(struct timeval *a, struct timeval *b);

} // namespace Xfibres


#endif
