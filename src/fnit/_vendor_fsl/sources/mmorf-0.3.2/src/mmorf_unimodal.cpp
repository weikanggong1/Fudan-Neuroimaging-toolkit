#include "RegistrationCoordinator.cuh"

#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <iostream>

int main(int argc, char * argv[])
{
  auto filename_warp_space                  = std::string(argv[1]);
  auto filename_volumes_reference           = std::vector<std::string>({argv[2]});
  auto filename_affine_transforms_reference = std::vector<std::string>({argv[3]});
  auto filename_volumes_moving              = std::vector<std::string>({argv[4]});
  auto filename_affine_transforms_moving    = std::vector<std::string>({argv[5]});
  auto filename_masks_reference             = std::vector<std::string>({argv[6]});
  auto smoothing_factor                     = static_cast<float>(std::atof(argv[7]));
  auto starting_regularisation              = static_cast<float>(std::atof(argv[8]));
  auto regularisation_factor                = static_cast<float>(std::atof(argv[9]));

  auto lambda_volumes = std::vector<std::vector<float> >({
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f}),
      std::vector<float>({1.0f})});
  auto smoothing_reference = std::vector<std::vector<float> >({
      std::vector<float>({40.0f/smoothing_factor}),
      std::vector<float>({40.0f/smoothing_factor}),
      std::vector<float>({20.0f/smoothing_factor}),
      std::vector<float>({10.0f/smoothing_factor}),
      std::vector<float>({5.0f/smoothing_factor}),
      std::vector<float>({2.5f/smoothing_factor}),
      std::vector<float>({1.25f/smoothing_factor})});
  auto smoothing_moving = std::vector<std::vector<float> >({
      std::vector<float>({40.0f/smoothing_factor}),
      std::vector<float>({40.0f/smoothing_factor}),
      std::vector<float>({20.0f/smoothing_factor}),
      std::vector<float>({10.0f/smoothing_factor}),
      std::vector<float>({5.0f/smoothing_factor}),
      std::vector<float>({2.5f/smoothing_factor}),
      std::vector<float>({1.25f/smoothing_factor})});
  auto lambda_regularisation = std::vector<float>({
      1.6e6f,
      starting_regularisation,
      starting_regularisation * std::pow(regularisation_factor,1.0f),
      starting_regularisation * std::pow(regularisation_factor,2.0f),
      starting_regularisation * std::pow(regularisation_factor,3.0f),
      starting_regularisation * std::pow(regularisation_factor,4.0f),
      starting_regularisation * std::pow(regularisation_factor,5.0f)});
//      1.5e-1f,
//      1.0e-1f,
//      0.5e-1f,
//      0.5e-1f,
//      0.5e-1f});
  auto warp_scaling = std::vector<int>({
      1,
      1,
      2,
      2,
      2,
      2,
      2});
  auto knot_spacing_initial = 40.0f;
  auto affines_are_inverted = true;

  std::cout << "Here" << std::endl;
  auto reg_coordinator = MMORF::RegistrationCoordinator(
      filename_warp_space,
      filename_volumes_reference,
      filename_affine_transforms_reference,
      filename_volumes_moving,
      filename_affine_transforms_moving,
      filename_masks_reference,
      lambda_volumes,
      smoothing_reference,
      smoothing_moving,
      lambda_regularisation,
      warp_scaling,
      knot_spacing_initial,
      affines_are_inverted);

  reg_coordinator.register_volumes(true,"./vols");
  reg_coordinator.save_warp_field(argv[10]);
  reg_coordinator.save_warped_volumes(argv[11]);
  reg_coordinator.save_jacobian_determinants(argv[12]);

  return 0;
}
