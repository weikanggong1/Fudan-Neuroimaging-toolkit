/*  xfibres_gpu.h

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"


void join_subParts(std::string name, int size_part, int nsubparts, int size_sub_part, int last_sub_part, bool mean);

void xfibres_gpu(    //INPUT
            const NEWMAT::Matrix datam,
            const NEWMAT::Matrix bvecs,
            const NEWMAT::Matrix bvals,
            const NEWMAT::Matrix gradm,
            int                  idpart,
            int                  idSubpart,
            float                seed,
            std::string          subjdir);
