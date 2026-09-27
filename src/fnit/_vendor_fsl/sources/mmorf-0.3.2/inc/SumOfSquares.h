/**
 *  \file
 *  \brief Cost function based on a sum of squared differences between two volumes
 *  \details Given two volumes, and a set of parameters, an object of this class will
 *           calculate the sum of the square of the difference in intensity between each
 *           voxel in the volume.
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#ifndef SUM_OF_SQUARES_H
#define SUM_OF_SQUARES_H

/** Multi-Modal Registration Framework */
namespace MMORF
{
    class SumOfSquares : public CostFunction
    {
        public:
    }; // SumOfSquares
}; // MMORF
#endif // SUM_OF_SQUARES_H
