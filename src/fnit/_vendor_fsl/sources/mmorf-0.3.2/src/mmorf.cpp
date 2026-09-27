//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Main function for using MMORF
/// \details Makes use of the CLI11 library to parse command line arguments. The executable is
///          able to handle an arbitraty number of scalar and tensor images for registration.
///          Note that the order in which multiple copies of the same argument are passed in
///          matters (e.g. the first ref_scalar and mov_scalar form a registration pair, the
///          second form the next pair, etc.).
/// \author Frederik Lange
/// \date November 2019
/// \copyright Copyright (C) 2019 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "RegistrationCoordinatorMultimodal.cuh"

#include "CLI/CLI.hpp"

#include <vector>
#include <string>
#include <sys/stat.h>
#include <cstdlib>

/////////////////////////////////////////
// Custom validators for NIfTI files
/////////////////////////////////////////
std::string NiftifyInput(std::string filename){
  if (filename == "NULL") return filename;
  struct stat buffer;
  bool exist = stat(filename.c_str(), &buffer) == 0;
  if (exist){
    return filename;
  }
  auto filename_nii = filename + std::string(".nii");
  exist = stat(filename_nii.c_str(), &buffer) == 0;
  if (exist){
    return filename_nii;
  }
  auto filename_nii_gz = filename + std::string(".nii.gz");
  exist = stat(filename_nii_gz.c_str(), &buffer) == 0;
  if (exist){
    return filename_nii_gz;
  }
  auto filename_gz = filename + std::string(".gz");
  exist = stat(filename_gz.c_str(), &buffer) == 0;
  if (exist){
    return filename_gz;
  }
  else{
    throw CLI::ValidationError("File does not exist: " + filename);
  }
}

/////////////////////////////////////////
// Print out names of input files
/////////////////////////////////////////
void printFilenames(std::string filetype, std::vector<std::string> filenames)
{
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    << std::endl;
  std::cout << "Files passed to parameter " << filetype << ":" << std::endl;
  for (const auto& filename : filenames){
    std::cout << filename << std::endl;
  }
  std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
}

/////////////////////////////////////////
// Main function of mmorf execuatable
/////////////////////////////////////////
int main(int argc, char * argv[])
{
  // Base app for command line parsing
  CLI::App app{"MMORF - The FSL MultiMOdal Registration Framework"};
  // Prettify help output
  app.get_formatter()->column_width(40);
  app.get_formatter()->label("REQUIRED","(REQ)");

  ////////////////////////////////////////
  // Command line options
  ////////////////////////////////////////
  // Multiresolution options
  float warp_res_init;
  app.add_option(
      "--warp_res_init",
      warp_res_init,
      "Initial warp resolution - in mm (isotropic)")
    ->required()
    ->group("Warp Options")
    ->take_first();

  std::vector<int> warp_scaling;
  app.add_option(
      "--warp_scaling",
      warp_scaling,
      "List of warp resolution scalings at each iteration")
    ->group("Warp Options");

  ////////////////////////////////////////
  // Images to register
  std::string img_warp_space;
  app.add_option(
      "--img_warp_space",
      img_warp_space,
      "3D NIfTI volume - space in which warp will be defined")
    ->required()
    ->group("Warp Options")
    ->transform(NiftifyInput);

  std::vector<std::string> img_ref_scalar;
  app.add_option(
      "--img_ref_scalar",
      img_ref_scalar,
      "3D NIfTI volume - scalar reference image")
    ->group("Scalar Image Options")
    ->transform(NiftifyInput);

  std::vector<std::string> img_mov_scalar;
  app.add_option(
      "--img_mov_scalar",
      img_mov_scalar,
      "3D NIfTI volume - scalar moving image")
    ->group("Scalar Image Options")
    ->transform(NiftifyInput);

  std::vector<std::string> img_ref_tensor;
  app.add_option(
      "--img_ref_tensor",
      img_ref_tensor,
      "4D NIfTI volume - tensor reference image")
    ->group("Tensor Image Options")
    ->transform(NiftifyInput);

  std::vector<std::string> img_mov_tensor;
  app.add_option(
      "--img_mov_tensor",
      img_mov_tensor,
      "4D NIfTI volume - tensor moving image")
    ->group("Tensor Image Options")
    ->transform(NiftifyInput);

  ////////////////////////////////////////
  // Affine transforms
  std::vector<std::string> aff_ref_scalar;
  app.add_option(
      "--aff_ref_scalar",
      aff_ref_scalar,
      "FLIRT .mat file - transformation TO warp space")
    ->group("Scalar Image Options")
    ->check(CLI::ExistingFile);

  std::vector<std::string> aff_mov_scalar;
  app.add_option(
      "--aff_mov_scalar",
      aff_mov_scalar,
      "FLIRT .mat file - transformation TO warp space")
    ->group("Scalar Image Options")
    ->check(CLI::ExistingFile);

  std::vector<std::string> aff_ref_tensor;
  app.add_option(
      "--aff_ref_tensor",
      aff_ref_tensor,
      "FLIRT .mat file - transformation TO warp space")
    ->group("Tensor Image Options")
    ->check(CLI::ExistingFile);

  std::vector<std::string> aff_mov_tensor;
  app.add_option(
      "--aff_mov_tensor",
      aff_mov_tensor,
      "FLIRT .mat file - transformation TO warp space")
    ->group("Tensor Image Options")
    ->check(CLI::ExistingFile);

  ////////////////////////////////////////
  // Mask Options
  std::vector<int> use_implicit_mask;
  app.add_option(
      "--use_implicit_mask",
      use_implicit_mask,
      "Treat 0s as missing data during smoothing")
    ->group("Mask Options")
    ->check(CLI::Range(0,1));

  std::vector<std::string> mask_ref_scalar;
  app.add_option(
      "--mask_ref_scalar",
      mask_ref_scalar,
      "3D NIfTI volume - soft mask for ref_scalar")
    ->group("Mask Options")
    ->transform(NiftifyInput);

  std::vector<std::string> mask_mov_scalar;
  app.add_option(
      "--mask_mov_scalar",
      mask_mov_scalar,
      "3D NIfTI volume - hard mask for mov_scalar")
    ->group("Mask Options")
    ->transform(NiftifyInput);

  std::vector<std::vector<int> > use_mask_ref_scalar;
  app.add_option(
      "--use_mask_ref_scalar",
      use_mask_ref_scalar,
      "Use mask for ref_scalar (1) or not (0)")
    ->group("Mask Options")
    ->check(CLI::Range(0,1));

  std::vector<std::vector<int> > use_mask_mov_scalar;
  app.add_option(
      "--use_mask_mov_scalar",
      use_mask_mov_scalar,
      "Use mask for mov_scalar (1) or not (0)")
    ->group("Mask Options")
    ->check(CLI::Range(0,1));

  std::vector<std::string> mask_ref_tensor;
  app.add_option(
      "--mask_ref_tensor",
      mask_ref_tensor,
      "3D NIfTI volume - soft mask for ref_tensor")
    ->group("Mask Options")
    ->transform(NiftifyInput);

  std::vector<std::string> mask_mov_tensor;
  app.add_option(
      "--mask_mov_tensor",
      mask_mov_tensor,
      "3D NIfTI volume - soft mask for mov_tensor")
    ->group("Mask Options")
    ->transform(NiftifyInput);

  std::vector<std::vector<int> > use_mask_ref_tensor;
  app.add_option(
      "--use_mask_ref_tensor",
      use_mask_ref_tensor,
      "Use mask for ref_tensor (1) or not (0)")
    ->group("Mask Options")
    ->check(CLI::Range(0,1));

  std::vector<std::vector<int> > use_mask_mov_tensor;
  app.add_option(
      "--use_mask_mov_tensor",
      use_mask_mov_tensor,
      "Use mask for mov_tensor (1) or not (0)")
    ->group("Mask Options")
    ->check(CLI::Range(0,1));

  ////////////////////////////////////////
  // Bias Field
  std::vector<int> estimate_bias;
  app.add_option(
      "--estimate_bias",
      estimate_bias,
      "Estimate bias field for scalar image pair (1) or not (0)")
    ->group("Bias Field Options")
    ->check(CLI::Range(0,1));

  std::vector<float> bias_res_init;
  app.add_option(
      "--bias_res_init",
      bias_res_init,
      "Initial bias field resolution - in mm (isotropic)")
    ->group("Bias Field Options");

  std::vector<std::vector<float> > lambda_bias_reg;
  app.add_option(
      "--lambda_bias_reg",
      lambda_bias_reg,
      "List of lambda values for weighting bias field regularisation")
    ->group("Bias Field Options");

  ////////////////////////////////////////
  // Lambdas
  std::vector<std::vector<float> > lambda_scalar;
  app.add_option(
      "--lambda_scalar",
      lambda_scalar,
      "List of lambda values for weighting scalar volume similarities")
    ->group("Scalar Image Options");

  std::vector<std::vector<float> > lambda_tensor;
  app.add_option(
      "--lambda_tensor",
      lambda_tensor,
      "List of lambda values for weighting tensor volume similarities")
    ->group("Tensor Image Options");

  std::vector<float> lambda_reg;
  app.add_option(
      "--lambda_reg",
      lambda_reg,
      "List of lambda values for weighting regularisation")
    ->group("Regularisation Options");

  ////////////////////////////////////////
  // Smoothing
  std::vector<std::vector<float> > fwhm_ref_scalar;
  app.add_option(
      "--fwhm_ref_scalar",
      fwhm_ref_scalar,
      "List of FWHMs for Gaussian smoothing img_ref_scalar")
    ->group("Scalar Image Options");

  std::vector<std::vector<float> > fwhm_mov_scalar;
  app.add_option(
      "--fwhm_mov_scalar",
      fwhm_mov_scalar,
      "List of FWHMs for Gaussian smoothing img_mov_scalar")
    ->group("Scalar Image Options");

  std::vector<std::vector<float> > fwhm_ref_tensor;
  app.add_option(
      "--fwhm_ref_tensor",
      fwhm_ref_tensor,
      "List of FWHMs for Gaussian smoothing img_ref_tensor")
    ->group("Tensor Image Options");

  std::vector<std::vector<float> > fwhm_mov_tensor;
  app.add_option(
      "--fwhm_mov_tensor",
      fwhm_mov_tensor,
      "List of FWHMs for Gaussian smoothing img_mov_tensor")
    ->group("Tensor Image Options");

  ////////////////////////////////////////
  // Output
  std::string warp_out = "./mmorf_warp";
  app.add_option(
      "--warp_out",
      warp_out,
      "Basename of resulting warp (omit .nii.gz etc)")
    ->required()
    ->group("Output Options");

  std::string jac_det_out = "./mmorf_jac_det";
  app.add_option(
      "--jac_det_out",
      jac_det_out,
      "Basename of resulting jacobian determinant map (omit .nii.gz etc)")
    ->group("Output Options");

std::string bias_out = "NULL";
app.add_option(
    "--bias_out",
    bias_out,
    "Basename of resulting bias field (omit .nii.gz etc)")
  ->group("Output Options");

  ////////////////////////////////////////
  // Optimiser
  std::vector<std::pair<std::string, MMORF::OptimiserType>> optimiser_map{
    {"LM", MMORF::OptimiserType::LM},
    {"MM", MMORF::OptimiserType::MM},
    {"SCG", MMORF::OptimiserType::SCG}};

  float hires = 4.9f;
  app.add_option(
      "--hires",
      hires,
      "Resolution (mm isotropic) below which low-memory optimisation is necessary")
    ->group("Optimiser Options");

  MMORF::OptimiserType optimiser_lowres = MMORF::OptimiserType::LM;
  app.add_option(
      "--optimiser_lowres",
      optimiser_lowres,
      "Optimiser to use at lower resolution (LM, MM or SCG)")
    ->group("Optimiser Options")
    ->transform(CLI::CheckedTransformer(optimiser_map, CLI::ignore_case));

  int optimiser_max_it_lowres = 5;
  app.add_option(
      "--optimiser_max_it_lowres",
      optimiser_max_it_lowres,
      "Maximum iterations for optimiser at lower resolution")
    ->group("Optimiser Options");

  float optimiser_rel_tol_lowres = 1e-3f;
  app.add_option(
      "--optimiser_rel_tol_lowres",
      optimiser_rel_tol_lowres,
      "Relative tolerance for optimiser at lower resolution")
    ->group("Optimiser Options");

  MMORF::OptimiserType optimiser_hires = MMORF::OptimiserType::MM;
  app.add_option(
      "--optimiser_hires",
      optimiser_hires,
      "Optimiser to use at higher resolution (LM, MM or SCG)")
    ->group("Optimiser Options")
    ->transform(CLI::CheckedTransformer(optimiser_map, CLI::ignore_case));

  int optimiser_max_it_hires = 5;
  app.add_option(
      "--optimiser_max_it_hires",
      optimiser_max_it_hires,
      "Maximum iterations for optimiser at higher resolution")
    ->group("Optimiser Options");

  float optimiser_rel_tol_hires = 1e-3f;
  app.add_option(
      "--optimiser_rel_tol_hires",
      optimiser_rel_tol_hires,
      "Relative tolerance for optimiser at higher resolution")
    ->group("Optimiser Options");

  ////////////////////////////////////////
  // Solver
  int solver_max_it_lowres = 100;
  app.add_option(
      "--solver_max_it_lowres",
      solver_max_it_lowres,
      "Maximum iterations for linear solver at lower resolution")
    ->group("Solver Options");

  float solver_rel_tol_lowres = 1e-3f;
  app.add_option(
      "--solver_rel_tol_lowres",
      solver_rel_tol_lowres,
      "Relative tolerance for linear solver at lower resolution")
    ->group("Solver Options");

  int solver_max_it_hires = 100;
  app.add_option(
      "--solver_max_it_hires",
      solver_max_it_hires,
      "Maximum iterations for linear solver at higher resolution")
    ->group("Solver Options");

  float solver_rel_tol_hires = 1e-3f;
  app.add_option(
      "--solver_rel_tol_hires",
      solver_rel_tol_hires,
      "Relative tolerance for linear solver at higher resolution")
    ->group("Solver Options");

  ////////////////////////////////////////
  // Debug flag (hidden)
  bool debug = false;
  app.add_flag(
      "--debug",
      debug)
    ->group("");

  ////////////////////////////////////////
  // Version flag
  auto callback = [](int count){
    std::cout << "MMORF version 0.3.2, Copyright (C) 2022 University of Oxford"
              << std::endl;
    return 0;
  };
  app.add_flag_function("--version", callback, "Show version number")
    ->group("Miscellaneous Options");

  ////////////////////////////////////////
  // Config file support
  app.set_config(
      "--config",
      "",
      "Use a .ini file to configure options")
    ->group("Miscellaneous Options");

  CLI11_PARSE(app, argc, argv);

  if (debug){
    auto tmp_sysval = std::system("mkdir -p debug");
  }

  // Convert integer values to bools
  auto estimate_bias_bool = std::vector<bool>();
  for (const auto& estimate_i : estimate_bias){
    if (estimate_i == 0){
      estimate_bias_bool.push_back(false);
    }
    else if (estimate_i == 1){
      estimate_bias_bool.push_back(true);
    }
    else{
      estimate_bias_bool.push_back(true);
      std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
    }
  }

  auto use_implicit_mask_bool = std::vector<bool>();
  for (const auto& estimate_i : use_implicit_mask){
    if (estimate_i == 0){
      use_implicit_mask_bool.push_back(false);
    }
    else if (estimate_i == 1){
      use_implicit_mask_bool.push_back(true);
    }
    else{
      use_implicit_mask_bool.push_back(true);
      std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
    }
  }

  auto use_mask_ref_scalar_bool = std::vector<std::vector<bool> >(use_mask_ref_scalar.size());
  for (auto i = 0; i < use_mask_ref_scalar.size(); ++i){
    for (auto j = 0; j < use_mask_ref_scalar[i].size(); ++j){
      if (use_mask_ref_scalar[i][j] == 0){
        use_mask_ref_scalar_bool[i].push_back(false);
      }
      else if (use_mask_ref_scalar[i][j] == 1){
        use_mask_ref_scalar_bool[i].push_back(true);
      }
      else{
        use_mask_ref_scalar_bool[i].push_back(true);
        std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
      }
    }
  }

  auto use_mask_mov_scalar_bool = std::vector<std::vector<bool> >(use_mask_mov_scalar.size());
  for (auto i = 0; i < use_mask_mov_scalar.size(); ++i){
    for (auto j = 0; j < use_mask_mov_scalar[i].size(); ++j){
      if (use_mask_mov_scalar[i][j] == 0){
        use_mask_mov_scalar_bool[i].push_back(false);
      }
      else if (use_mask_mov_scalar[i][j] == 1){
        use_mask_mov_scalar_bool[i].push_back(true);
      }
      else{
        use_mask_mov_scalar_bool[i].push_back(true);
        std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
      }
    }
  }

  auto use_mask_ref_tensor_bool = std::vector<std::vector<bool> >(use_mask_ref_tensor.size());
  for (auto i = 0; i < use_mask_ref_tensor.size(); ++i){
    for (auto j = 0; j < use_mask_ref_tensor[i].size(); ++j){
      if (use_mask_ref_tensor[i][j] == 0){
        use_mask_ref_tensor_bool[i].push_back(false);
      }
      else if (use_mask_ref_tensor[i][j] == 1){
        use_mask_ref_tensor_bool[i].push_back(true);
      }
      else{
        use_mask_ref_tensor_bool[i].push_back(true);
        std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
      }
    }
  }

  auto use_mask_mov_tensor_bool = std::vector<std::vector<bool> >(use_mask_mov_tensor.size());
  for (auto i = 0; i < use_mask_mov_tensor.size(); ++i){
    for (auto j = 0; j < use_mask_mov_tensor[i].size(); ++j){
      if (use_mask_mov_tensor[i][j] == 0){
        use_mask_mov_tensor_bool[i].push_back(false);
      }
      else if (use_mask_mov_tensor[i][j] == 1){
        use_mask_mov_tensor_bool[i].push_back(true);
      }
      else{
        use_mask_mov_tensor_bool[i].push_back(true);
        std::cout << "Non 0/1 bool found. This should never happen!" << std::endl;
      }
    }
  }

  // Print out filenames
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    << std::endl;
  std::cout << "File passed to parameter img_warp_space:"
    << std::endl
    << img_warp_space
    << std::endl;
  std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    << std::endl;
  printFilenames("img_ref_scalar", img_ref_scalar);
  printFilenames("img_mov_scalar", img_mov_scalar);
  printFilenames("mask_ref_scalar", mask_ref_scalar);
  printFilenames("mask_mov_scalar", mask_mov_scalar);
  printFilenames("img_ref_tensor", img_ref_tensor);
  printFilenames("img_mov_tensor", img_mov_tensor);
  printFilenames("mask_ref_tensor", mask_ref_tensor);
  printFilenames("mask_mov_tensor", mask_mov_tensor);

  auto reg_coordinator = MMORF::RegistrationCoordinatorMultimodal(
      // Warp defining parameters
      warp_res_init,
      warp_scaling,
      img_warp_space,
      // Scalar image parameters
      img_ref_scalar,
      img_mov_scalar,
      aff_ref_scalar,
      aff_mov_scalar,
      use_implicit_mask_bool,
      mask_ref_scalar,
      mask_mov_scalar,
      use_mask_ref_scalar_bool,
      use_mask_mov_scalar_bool,
      fwhm_ref_scalar,
      fwhm_mov_scalar,
      lambda_scalar,
      // Tensor image parameters
      img_ref_tensor,
      img_mov_tensor,
      aff_ref_tensor,
      aff_mov_tensor,
      mask_ref_tensor,
      mask_mov_tensor,
      use_mask_ref_tensor_bool,
      use_mask_mov_tensor_bool,
      fwhm_ref_tensor,
      fwhm_mov_tensor,
      lambda_tensor,
      // Regularisation parameters
      lambda_reg,
      // Optimiser parameters
      hires,
      optimiser_lowres,
      optimiser_max_it_lowres,
      optimiser_rel_tol_lowres,
      optimiser_hires,
      optimiser_max_it_hires,
      optimiser_rel_tol_hires,
      // Solver parameters
      solver_max_it_lowres,
      solver_rel_tol_lowres,
      solver_max_it_hires,
      solver_rel_tol_hires,
      // Bias field parameters
      estimate_bias_bool,
      bias_res_init,
      lambda_bias_reg,
      // Debugging option
      debug
      );

  reg_coordinator.register_volumes();
  reg_coordinator.save_warp_field(warp_out);
  reg_coordinator.save_jacobian_determinants(jac_det_out);
  if (bias_out != "NULL"){
    reg_coordinator.save_bias_field(bias_out);
  }

  return 0;
}
