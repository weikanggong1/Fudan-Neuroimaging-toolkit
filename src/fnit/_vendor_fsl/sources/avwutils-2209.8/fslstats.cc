/*  fslstats.cc

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2003-2009 University of Oxford  */

/*  CCOPYRIGHT  */

#include <limits>
#include <string>
#include <typeinfo>

#include "utils/fsl_isfinite.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "newimage/costfns.h"
#include "nlohmann/json.hpp"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;
using json = nlohmann::json;

int print_usage(const string &progname)
{
  cout << "Usage: fslstats [preoptions] <input> [options]" << endl
       << endl;
  // add --json output format option
  // keys will correspond to the options above
  cout << "preoption -json: output in JSON format to standard out" << endl;
  cout << "preoption -t will give a separate output line for each 3D volume of a 4D timeseries" << endl;
  cout << "preoption -K < indexMask > will generate seperate n submasks from indexMask, for indexvalues 1..n where n is the maximum index value in indexMask, and generate statistics for each submask" << endl;
  cout << "Note - options are applied in order, e.g. -M -l 10 -M will report the non-zero mean, apply a threshold and then report the new nonzero mean" << endl
       << endl;
  cout << "-l <lthresh> : set lower threshold" << endl;
  cout << "-u <uthresh> : set upper threshold" << endl;
  cout << "-r           : output <robust min intensity> <robust max intensity>" << endl;
  cout << "-R           : output <min intensity> <max intensity>" << endl;
  cout << "-e           : output mean entropy ; mean(-i*ln(i))" << endl;
  cout << "-E           : output mean entropy (of nonzero voxels)" << endl;
  cout << "-v           : output <voxels> <volume>" << endl;
  cout << "-V           : output <voxels> <volume> (for nonzero voxels)" << endl;
  cout << "-m           : output mean" << endl;
  cout << "-M           : output mean (for nonzero voxels)" << endl;
  cout << "-s           : output standard deviation" << endl;
  cout << "-S           : output standard deviation (for nonzero voxels)" << endl;
  cout << "-w           : output smallest ROI <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize> containing nonzero voxels" << endl;
  cout << "-x           : output co-ordinates of maximum voxel" << endl;
  cout << "-X           : output co-ordinates of minimum voxel" << endl;
  cout << "-c           : output centre-of-gravity (cog) in mm coordinates" << endl;
  cout << "-C           : output centre-of-gravity (cog) in voxel coordinates" << endl;
  cout << "-p <n>       : output nth percentile (n between 0 and 100)" << endl;
  cout << "-P <n>       : output nth percentile (for nonzero voxels)" << endl;
  cout << "-a           : use absolute values of all image intensities" << endl;
  cout << "-n           : treat NaN or Inf as zero for subsequent stats" << endl;
  cout << "-k <mask>    : use the specified image (filename) for masking - overrides lower and upper thresholds" << endl;
  cout << "-d <image>   : take the difference between the base image and the image specified here" << endl;
  cout << "-h <nbins>   : output a histogram (for the thresholded/masked voxels only) with nbins" << endl;
  cout << "-H <nbins> <min> <max>   : output a histogram (for the thresholded/masked voxels only) with nbins and histogram limits of min and max" << endl; 
  cout << "Note - thresholds are not inclusive ie lthresh<allowed<uthresh" << endl;
  return 1;
}

// Some specialised nonzero functions just for speedup
//  (it avoids generating masks when not absolutely necessary)

long nonzerocount(const volume4D<float> &vol)
{
  long totn = 0;
  for (volume4D<float>::fast_const_iterator it = vol.fbegin(), end = vol.fend(); it < end; it++)
    if ((*it) != 0)
      totn++;
  return totn;
}

double nonzeromean(const volume4D<float> &vol)
{
  double total = 0.0;
  long int totn = 0;
  for (volume4D<float>::fast_const_iterator it = vol.fbegin(), end = vol.fend(); it < end; it++)
    if ((*it) != 0)
    {
      total += (double)(*it);
      totn++;
    }
  if (totn > 0)
    total /= totn;
  return total;
}

double nonzerostddev(const volume4D<float> &vol)
{
  double totv = 0.0, totvv = 0.0;
  long int totn = 0;
  for (volume4D<float>::fast_const_iterator it = vol.fbegin(), end = vol.fend(); it < end; it++)
    if ((*it) != 0)
    {
      totvv += (double)((*it) * (*it));
      totv += (double)(*it);
      totn++;
    }
  double var = 0.0;
  if (totn > 1)
  {
    double meanval = totv / totn;
    var = (totvv - totn * meanval * meanval) / (totn - 1);
  }
  return std::sqrt(var);
}

template <class M>
int generateNonZeroMask(const M &mask, volume4D<float> &masknz, const volume4D<float> &input)
{
  masknz = binarise(input, 0.0f, 0.0f, inclusive, true) * mask;
  return 0;
}

int generate_masks(volume4D<float> &mask, volume4D<float> &masknz, const volume4D<float> &input, const float &lthr, const float &uthr)
{
  mask = binarise(input, lthr, uthr, exclusive);
  return generateNonZeroMask(mask, masknz, input);
}

// print function that can handle outputting in JSON format or not. Use a template to allow for different types
template <typename T>
void update_json(string key, T &value, json &j)
{
  j[key] = value;
}

template <typename T>
void update_json(string key, initializer_list<T> &value, json &j)
{
  j[key] = value;
}

template <typename T>
void update_json(string key, vector<T> &value, json &j)
{
  j[key] = value;
}

void update_json(string key, ColumnVector &value, json &j)
{
  j[key] = value;
}

int fmrib_main_float(int argc, char *argv[], const bool timeseriesMode, const string &indexMaskName, const bool jsonMode = false)
{

  json js;

  cout.setf(ios::dec);
  cout.setf(ios::fixed, ios::floatfield);
  cout.setf(ios::left, ios::adjustfield);
  cout.precision(6);
  volume<float> vol, inputMaster;
  volume<int> indexMask;
  if (timeseriesMode || indexMaskName != "")
    read_volume4D(inputMaster, argv[1]);
  else
    read_volume4D(vol, argv[1]);
  volume<float> &indexMaster = (timeseriesMode) ? vol : inputMaster;
  if (indexMaskName != "")
    read_volume4D(indexMask, indexMaskName);
  int nTimepoints((timeseriesMode) ? inputMaster.tsize() : 1), nIndices((indexMaskName != "") ? indexMask.max() : 1);

  // initialise the volumes array in the json object. This will get appended to later
  js["volumes"] = json::array();

  for (int timepoint = 0; timepoint < nTimepoints; timepoint++)
  {
    json tp = json::object();
    tp["indices"] = json::array();
    for (int index = 1; index <= nIndices; index++)
    {
      json idx = json::object();
      if (timeseriesMode)
        vol = inputMaster[timepoint];
      volume<float> mask, masknz;
      float lthr(-numeric_limits<float>::max());
      float uthr(numeric_limits<float>::max());
      if (indexMask.nvoxels())
      {
        if (indexMask.dimensionality() > 3)
        {
          cerr << "Index mask must be 3D" << endl;
          return 3;
        }
        copyconvert(indexMask, mask);
        mask.binarise(index - 1, index + 1, exclusive);
        vol = indexMaster * mask;
        if (mask.max() < 1)
        {
          cout << "missing label: " << index << " in mask" << endl;
          continue;
        }
        generateNonZeroMask(mask, masknz, vol);
      }
      int narg(2);
      while (narg < argc)
      {

        string sarg(argv[narg]);
        if (sarg == "-n")
        {
          for (int t = 0; t < vol.tsize(); t++)
            for (int z = 0; z < vol.zsize(); z++)
              for (int y = 0; y < vol.ysize(); y++)
                for (int x = 0; x < vol.xsize(); x++)
                  if (!isfinite((double)vol(x, y, z, t)))
                    vol(x, y, z, t) = 0;
        }
        else if (sarg == "-m")
        {
          if (mask.nvoxels() > 0){
            auto val = vol.mean(mask);
            if (jsonMode){
              update_json(sarg, val, idx);
            } else {
              cout << val << " ";
            }
          } else {
            // print_value(sarg, val, json, jsonMode);
            auto val = vol.mean();
            if (jsonMode){
              update_json(sarg, val, idx);
            } else {
              cout << val << " ";
            }
          }
        }
        else if (sarg == "-M")
        {
          if (masknz.nvoxels() > 0){
            auto val = vol.mean(masknz);
            if (jsonMode){
              update_json(sarg, val, idx);
            } else {
              cout << val << " ";
            }
          }
          else
          {
            double nzmean = 0;
            nzmean = nonzeromean(vol);
            if (jsonMode){
              update_json(sarg, nzmean, idx);
            } else {
              cout << nzmean << " ";
            }
          }
        }
        else if (sarg == "-X")
        {
          ColumnVector coord(4);
          vector<int64_t> iCoords;
          vol.min(mask, iCoords);
          coord << (Real)iCoords[7] << (Real)iCoords[8] << (Real)iCoords[9] << 1.0;
          coord = vol.niftivox2newimagevox_mat().i() * coord;
          //cout << MISCMATHS::round(coord(1)) << " " << MISCMATHS::round(coord(2)) << " " << MISCMATHS::round(coord(3)) << " ";
          auto coord1 = MISCMATHS::round(coord(1));
          auto coord2 = MISCMATHS::round(coord(2));
          auto coord3 = MISCMATHS::round(coord(3));
          auto coordList = {coord1, coord2, coord3};
          if (jsonMode){
            update_json(sarg, coordList, idx);
          } else {
            cout << coord1 << " " << coord2 << " " << coord3 << " ";
          }
          
        }
        else if (sarg == "-x")
        {
          ColumnVector coord(4);
          vector<int64_t> iCoords;
          vol.min(mask, iCoords);
          coord << (Real)iCoords[0] << (Real)iCoords[1] << (Real)iCoords[2] << 1.0;
          coord = vol.niftivox2newimagevox_mat().i() * coord;
          //cout << MISCMATHS::round(coord(1)) << " " << MISCMATHS::round(coord(2)) << " " << MISCMATHS::round(coord(3)) << " ";
          auto coord1 = MISCMATHS::round(coord(1));
          auto coord2 = MISCMATHS::round(coord(2));
          auto coord3 = MISCMATHS::round(coord(3));
          auto coordList = {coord1, coord2, coord3};
          if (jsonMode){
            update_json(sarg, coordList, idx);
          } else {
            cout << coord1 << " " << coord2 << " " << coord3 << " ";
          }
        }
        else if (sarg == "-w")
        {
          if (masknz.nvoxels() < 1)
          { // Need to generate non-zeromask
            generate_masks(mask, masknz, vol, lthr, uthr);
            vol *= mask;
          }
          int xmin = masknz.xsize() - 1, xmax(0), ymin = masknz.ysize() - 1, ymax(0), zmin = masknz.zsize() - 1, zmax(0), tmin = masknz.tsize() - 1, tmax(0);

          for (int t = 0; t < masknz.tsize(); t++)
            for (int z = 0; z < masknz.zsize(); z++)
              for (int y = 0; y < masknz.ysize(); y++)
                for (int x = 0; x < masknz.xsize(); x++)
                  if (masknz(x, y, z, t) > 0.5)
                  {
                    if (x < xmin)
                      xmin = x;
                    if (x > xmax)
                      xmax = x;
                    if (y < ymin)
                      ymin = y;
                    if (y > ymax)
                      ymax = y;
                    if (z < zmin)
                      zmin = z;
                    if (z > zmax)
                      zmax = z;
                    if (t < tmin)
                      tmin = t;
                    if (t > tmax)
                      tmax = t;
                  }
          // change voxel coords from newimage to nifti convention for output
          ColumnVector v(4);
          v << xmin << ymin << zmin << 1.0;
          v = masknz.niftivox2newimagevox_mat().i() * v;
          xmin = MISCMATHS::round(v(1));
          ymin = MISCMATHS::round(v(2));
          zmin = MISCMATHS::round(v(3));
          v << xmax << ymax << zmax << 1.0;
          v = masknz.niftivox2newimagevox_mat().i() * v;
          xmax = MISCMATHS::round(v(1));
          ymax = MISCMATHS::round(v(2));
          zmax = MISCMATHS::round(v(3));
          if (xmin > xmax)
          {
            int tmp = xmax;
            xmax = xmin;
            xmin = tmp;
          }
          if (ymin > ymax)
          {
            int tmp = ymax;
            ymax = ymin;
            ymin = tmp;
          }
          if (zmin > zmax)
          {
            int tmp = zmax;
            zmax = zmin;
            zmin = tmp;
          }
          vector<int> coords = {xmin, 1 + xmax - xmin, ymin, 1 + ymax - ymin, zmin, 1 + zmax - zmin, tmin, 1 + tmax - tmin};
          if (jsonMode){
            update_json(sarg, coords, idx);
          } else {
            cout << coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << " " << coords[4] << " " << coords[5] << " " << coords[6] << " " << coords[7] << " ";
          }
        }
        else if (sarg == "-e")
        {
          if (mask.nvoxels() < 1)
          {
            generate_masks(mask, masknz, vol, lthr, uthr);
            vol *= mask;
          }
          ColumnVector hist;
          int nbins = 1000;
          double entropy = 0;
          hist = vol.histogram(nbins, mask);
          double ntot = hist.Sum();
          for (int j = 1; j <= nbins; j++)
          {
            if (hist(j) > 0)
            {
              entropy -= (hist(j) / ntot) * log(hist(j) / ntot);
            }
          }
          entropy /= log((double)nbins);
          if (jsonMode){
            update_json(sarg, entropy, idx);
          } else {
            cout << entropy << " ";
          }
        }
        else if (sarg == "-E")
        {
          ColumnVector hist;
          int nbins = 1000;
          double entropy = 0;
          if (mask.nvoxels() < 1)
          {
            generate_masks(mask, masknz, vol, lthr, uthr);
            vol *= mask;
          }
          hist = vol.histogram(nbins, masknz);
          double ntot = hist.Sum();
          for (int j = 1; j <= nbins; j++)
          {
            if (hist(j) > 0)
            {
              entropy -= (hist(j) / ntot) * log(hist(j) / ntot);
            }
          }
          entropy /= log((double)nbins);
          if (jsonMode){
            update_json(sarg, entropy, idx);
          } else {
            cout << entropy << " ";
          }
        }
        else if (sarg == "-k")
        {
          narg++;
          volume<float> mask2;
          if (narg >= argc)
          {
            cerr << "Must specify an argument to -k" << endl;
            exit(2);
          }
          read_volume4D(mask, argv[narg]);
          if (!samesize(mask, vol, 3) || mask.tsize() > vol.tsize())
          {
            cerr << "Mask and image must be the same size" << endl;
            exit(3);
          }
          if (timeseriesMode && mask.tsize() != 1)
          {
            mask2 = mask[timepoint];
          }
          if (mask.tsize() != vol.tsize() && mask.tsize() != 1)
            while (mask.tsize() < vol.tsize())
            { // copy the last max volume until the correct size is reached
              mask.addvolume(mask[mask.tsize() - 1]);
            }
          mask.binarise(0.5);
          if (mask.tsize() != 1)
          {
            generateNonZeroMask(mask, masknz, vol);
            vol *= mask;
          }
          else
          {
            generateNonZeroMask(mask[0], masknz, vol);
            vol *= mask[0];
          }
        }
        else if (sarg == "-d")
        {
          narg++;
          if (narg >= argc)
          {
            cerr << "Must specify an argument to -d" << endl;
            exit(2);
          }
          volume4D<float> image2, image3;
          read_volume4D(image2, argv[narg]);
          if (!samesize(image2[0], vol[0]))
          {
            cerr << "Image used with -d must be the same size as the base image" << endl;
            exit(3);
          }
          if (timeseriesMode)
          {
            image3 = image2[timepoint];
          }
          if (image2.tsize() > vol.tsize())
          {
            cerr << "Image must be the same size as the base image" << endl;
            exit(3);
          }
          if (image2.tsize() != vol.tsize() && image2.tsize() != 1)
          {
            // copy the last max volume until the correct size is reached
            while (image2.tsize() < vol.tsize())
            {
              image2.addvolume(image2[image2.maxt()]);
            }
          }
          // now substract the new image from the base image
          vol -= image2;
        }
        else if (sarg == "-l")
        {
          narg++;
          if (narg < argc)
          {
            lthr = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify an argument to -l" << endl;
            exit(2);
          }
          generate_masks(mask, masknz, vol, lthr, uthr);
          vol *= mask;
        }
        else if (sarg == "-u")
        {
          narg++;
          if (narg < argc)
          {
            uthr = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify an argument to -u" << endl;
            exit(2);
          }
          generate_masks(mask, masknz, vol, lthr, uthr);
          vol *= mask;
        }
        else if (sarg == "-a")
        {
          vol = abs(vol);
        }
        else if (sarg == "-v")
        {
          if (mask.nvoxels() > 0)
          {
            long int nvox = mask.sum();
            if (mask.tsize() == 1)
              nvox = nvox * vol.tsize();
            
            auto volume_v = nvox * vol.xdim() * vol.ydim() * vol.zdim();
            // promote nvox to float to maintain types in list
            auto nvox_volume = {static_cast<float>(nvox), volume_v};
            if (jsonMode){
              update_json(sarg, nvox_volume, idx);
            } else {
              cout << nvox << " " << volume_v << " ";
            }
          }
          else
          {
            long int nvox = vol.nvoxels() * vol.tsize();
            auto volume_v = nvox * vol.xdim() * vol.ydim() * vol.zdim();
            // promote nvox to float to maintain types in list
            auto nvox_volume = {static_cast<float>(nvox), volume_v};
            if (jsonMode){
              update_json(sarg, nvox_volume, idx);
            } else {
              cout << nvox << " " << volume_v << " ";
            }
          }
        }
        else if (sarg == "-V")
        {
          if (masknz.nvoxels() > 0)
          {
            long int nvox = masknz.sum();
            auto volume_V = nvox * vol.xdim() * vol.ydim() * vol.zdim();
            // promote nvox to float to maintain types in list
            auto nvox_volume = {static_cast<float>(nvox), volume_V};
            if (jsonMode){
              update_json(sarg, nvox_volume, idx);
            } else {
              cout << nvox << " " << volume_V << " ";
            }
          }
          else
          {
            long int nzvox;
            nzvox = nonzerocount(vol);
            auto volume_V = nzvox * vol.xdim() * vol.ydim() * vol.zdim();
            // promate nzvox to float to maintain types in list
            auto nvox_volume = {static_cast<float>(nzvox), volume_V};
            if (jsonMode){
              update_json(sarg, nvox_volume, idx);
            } else {
              cout << nzvox << " " << volume_V << " ";
            }
          }
        }
        else if (sarg == "-D")
        {
          // hidden debug option!
          auto sum = vol.sum();
          if (jsonMode){
            update_json(sarg, sum, idx);
          } else {
            cout << sum << " ";
          }
        }
        else if (sarg == "-s")
        {
          if (mask.nvoxels() > 0){
            auto stddev = vol.stddev(mask);
            if (jsonMode){
              update_json(sarg, stddev, idx);
            } else {
              cout << stddev << " ";
            }
          }
          else {
            auto stddev = vol.stddev();
            if (jsonMode){
              update_json(sarg, stddev, idx);
            } else {
              cout << stddev << " ";
            }
          }
        }
        else if (sarg == "-S")
        {
          if (masknz.nvoxels() > 0)
          {
            auto stddev = vol.stddev(masknz);
            if (jsonMode){
              update_json(sarg, stddev, idx);
            } else {
              cout << stddev << " ";
            }
          }
          else
          {
            auto stddev = nonzerostddev(vol);
            if (jsonMode){
              update_json(sarg, stddev, idx);
            } else {
              cout << stddev << " ";
            }
          }
        }
        else if (sarg == "-r")
        {
          vector<float> limits(vol.robustlimits(mask));
          if (jsonMode){
            update_json(sarg, limits, idx);
          } else {
            cout << limits[0] << " " << limits[1] << " ";
          }
        }
        else if (sarg == "-R")
        {
          auto min_R = vol.min(mask);
          auto max_R = vol.max(mask);
          auto minmax_R = {min_R, max_R};
          if (jsonMode){
            update_json(sarg, minmax_R, idx);
          } else {
            cout << min_R << " " << max_R << " ";
          }
        }
        else if (sarg == "-c")
        {
          ColumnVector cog(4);
          cog.SubMatrix(1, 3, 1, 1) = vol.cog();
          cog(4) = 1.0;
          cog = vol.newimagevox2mm_mat() * cog;
          // remove the last element
          cog = cog.Rows(1, 3);
          // reshape to a row vector
          cog = cog.t();
          if (jsonMode){
            update_json(sarg, cog, idx);
          } else {
            cout << cog(1) << " " << cog(2) << " " << cog(3) << " ";
          }
        }
        else if (sarg == "-C")
        {
          ColumnVector cog(4);
          // convert from fsl mm to fsl voxel coord to nifti voxel coord
          cog.SubMatrix(1, 3, 1, 1) = vol.cog();
          cog(4) = 1.0;
          cog = vol.niftivox2newimagevox_mat().i() * cog;
          // remove the last element
          cog = cog.Rows(1, 3);
          // reshape to a row vector
          cog = cog.t();
          if (jsonMode){
            update_json(sarg, cog, idx);
          } else {
            cout << cog(1) << " " << cog(2) << " " << cog(3) << " ";
          }
        }
        else if (sarg == "-p")
        {
          float n;
          narg++;
          if (narg < argc)
          {
            n = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify an argument to -p" << endl;
            exit(2);
          }
          if ((n < 0) || (n > 100))
          {
            cerr << "Percentile must be between 0 and 100" << endl;
            exit(1);
          }
          if (mask.nvoxels() > 0){
            auto percentile_p = vol.percentile((float)n / 100.0, mask);
            if (jsonMode){
              update_json(sarg, percentile_p, idx);
            } else {
              cout << percentile_p << " ";
            }
          }
          else {
            auto percentile_p = vol.percentile((float)n / 100.0);
            if (jsonMode){
              update_json(sarg, percentile_p, idx);
            } else {
              cout << percentile_p << " ";
            }
          }
        }
        else if (sarg == "-P")
        {
          float n;
          narg++;
          if (narg < argc)
          {
            n = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify an argument to -P" << endl;
            exit(2);
          }
          if ((n < 0) || (n > 100))
          {
            cerr << "Percentile must be between 0 and 100" << endl;
            exit(1);
          }
          if (mask.nvoxels() < 1)
          {
            generate_masks(mask, masknz, vol, lthr, uthr);
            vol *= mask;
          }
          auto percentile_P = vol.percentile((float)n / 100.0, masknz);
          if (jsonMode){
            update_json(sarg, percentile_P, idx);
          } else {
            cout << percentile_P << " ";
          }
        }
        else if (sarg == "-h")
        {
          float n;
          narg++;
          if (narg < argc)
          {
            n = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify the number of bins" << endl;
            exit(2);
          }
          int nbins = (int)n;
          if (nbins < 1)
          {
            cerr << "Must specify at least 1 bin" << endl;
            exit(1);
          }
          if (mask.nvoxels() > 0)
          {
            auto hist = vol.histogram(nbins, vol.min(), vol.max(), mask);
            if (jsonMode){
              update_json(sarg, hist, idx);
            } else {
              cout << hist << " ";
            }
          }
          else
          {
            auto hist = vol.histogram(nbins, vol.min(), vol.max());
            if (jsonMode){
              update_json(sarg, hist, idx);
            } else {
              cout << hist << " ";
            }
          }
        }
        else if (sarg == "-H")
        {
          float n;
          narg++;
          if (narg < argc)
          {
            n = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify the number of bins" << endl;
            exit(2);
          }
          int nbins = (int)n;
          if (nbins < 1)
          {
            cerr << "Must specify at least 1 bin" << endl;
            exit(1);
          }
          float min = 0;
          narg++;
          if (narg < argc)
          {
            min = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify the histogram minimum intensity" << endl;
            exit(2);
          }
          float max = 0;
          narg++;
          if (narg < argc)
          {
            max = atof(argv[narg]);
          }
          else
          {
            cerr << "Must specify the histogram maximum intensity" << endl;
            exit(2);
          }
          if (mask.nvoxels() > 0)
          {
            auto hist = vol.histogram(nbins,min,max,mask);
            if (jsonMode){
              update_json(sarg, hist, idx);
            } else {
              cout << hist << " ";
            }
          }
          else
          {
            auto hist = vol.histogram(nbins, min, max);
            if (jsonMode){
              update_json(sarg, hist, idx);
            } else {
              cout << hist << " ";
            }
          }
        }
        else
        {
          cerr << "Unrecognised option: " << sarg << endl;
          exit(3);
        }
        narg++;
      }
      tp["indices"] += idx;
      if (!jsonMode)
        cout << endl;
    } // end nIndices
    // append the timepoint to the the volumes array
    js["volumes"] += tp;
  } // end timepoints
  if (jsonMode){
    // print the json with 4 spaces of indentation
    cout << js.dump(4) << endl;
  }   
  return 0;
}

int main(int argc, char *argv[])
{

  Tracer tr("main");
  string progname(argv[0]);
  bool timeseriesMode(false);
  bool jsonMode(false);
  string indexMask("");
  while (argc > 2 && (string(argv[1]) == "-t" || string(argv[1]) == "-K" || string(argv[1]) == "-json"))
  {
    if (string(argv[1]) == "-t")
      timeseriesMode = true;
    if (string(argv[1]) == "-json")
      jsonMode = true;
    if (string(argv[1]) == "-K")
    {
      indexMask = string(argv[2]);
      argv++;
      argc--;
    }
    argv++;
    argc--;
  }
  if (argc < 3)
    return print_usage(progname);
  try
  {
    return fmrib_main_float(argc, argv, timeseriesMode, indexMask, jsonMode);
  }
  catch (std::exception &e)
  {
    cerr << e.what() << endl;
  }
  catch (...)
  {
    // do nothing - just exit without garbage message
  }

  return -1;
}
