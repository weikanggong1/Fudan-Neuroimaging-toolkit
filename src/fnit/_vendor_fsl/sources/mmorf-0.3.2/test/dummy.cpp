/**
 *  \file
 *  \brief Dummy C++ file intended to help me understand how the model of my framework will
 *         work.
 *  \details This is not runable code, but hopefully describes the way I intend runnable
 *           code to use my classes.
 *  \author Frederik Lange
 *  \date January 2018
 *  \copyright FMRIB Copyright Licence
*/

#include "Volume.h"
#include "NewimageVolume.h"
#include "CostFunction.h"
#include "SumOfSquares.h"

int main()
{
    // Read nifti files in and store them in MMORF::NewImageVolume objects
    auto stationary_volume_name = "stationary_volume.nii.gz";
    auto moving_volume_name = "moving_volume.nii.gz";
    auto stationary_volume = std::make_shared<MMORF::VolumeNewimage>(stationary_volume_name);
    auto moving_volume = std::make_shared<MMORF::VolumeNewimage>(moving_volume_name);
    // Create a warp-field based on the volumes that were read in
    const auto volume_extents_mm = moving_volume.extents();
    auto warp_knot_spacing_mm = 10.0f;
    MMORF::SplineField3D warp_field(volume_extents_mm, warp_knot_spacing_mm);
    // Create cost function for current registration
    MMORF::SumOfSquares cost_function(stationary_volume, moving_volume, warp_field);
    // Use cost_function to get initial values for default parameters
    auto initial_warp_parameters = warp_field.parameters();
    auto cost = cost_function.cost(initial_warp_parameters);
    auto deriv = cost_function.deriv(initial_warp_parameters);
    auto hess = cost_function.hess(initial_warp_parameters);
    return 0;
}
