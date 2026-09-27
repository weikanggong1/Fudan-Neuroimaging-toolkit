//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Calculate the average of a number of tensor images
/// \details This is intended for use when creating DTI templates. Averaging is based on the
///          log-Euclidian representation of the tensor.
/// \author Frederik Lange
/// \date January 2020
/// \copyright Copyright (C) 2020 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#include "tensor_fxns.h"
#include "newimage/newimageall.h"
#include "CLI/CLI.hpp"
#include <vector>
#include <string>
#include <sys/stat.h>

/////////////////////////////////////////
// Custom validators for NIfTI files
/////////////////////////////////////////
std::string NiftifyInput(std::string filename){
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
// Main fxn of tensor_average execuatable
/////////////////////////////////////////
int main(int argc, char * argv[])
{
  // Base app for command line parsing
  CLI::App app{"tensor_average - Calculate the Log-Euclidian tensor average of DTI volumes"};
  // Prettify help output
  app.get_formatter()->column_width(40);
  app.get_formatter()->label("REQUIRED","(REQ)");

  ////////////////////////////////////////
  // Command line options
  ////////////////////////////////////////
  std::vector<std::string> input_filenames;
  app.add_option(
      "-i,--input",
      input_filenames,
      "Input DTI volumes - in FSL FDT format")
    ->required()
    ->group("Inputs")
    ->transform(NiftifyInput);

  std::string output_filename;
  app.add_option(
      "-o, --output",
      output_filename,
      "Output average DTI volume - in FSL FDT format")
    ->required()
    ->group("Outputs");
  ////////////////////////////////////////
  // Config file support
//  app.set_config(
//      "--config",
//      "",
//      "Use a .ini file to configure options")
//    ->group("Miscelaneous Options");

  CLI11_PARSE(app, argc, argv);

  auto average_tensor = MMORF::tensor_average(input_filenames);
  NEWIMAGE::write_volume4D(average_tensor, output_filename);

  return 0;
}
