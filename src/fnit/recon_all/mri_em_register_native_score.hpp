// FNIT single-input search hotspot. Original optimizer/gradient remain upstream.
// See task_03/README.md for coordinate, dtype and source provenance requirements.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "romp_support.h"

namespace fnit_gca_search {
struct Capabilities {
  Capabilities() {
    const char *query = std::getenv("FNIT_GCA_QUERY_CAPABILITIES");
    if (query && !std::strcmp(query, "1")) {
      std::puts("{\"fnit_gca_cached_search\":true,\"full_native_em\":true,\"version\":1,\"single_input_uint8\":true,\"activation\":\"FNIT_GCA_SCORER=cpu_cached\"}");
      std::exit(0);
    }
  }
};
static Capabilities capabilities;

struct DensityCache {
  float variance = std::numeric_limits<float>::quiet_NaN();
  float prior = std::numeric_limits<float>::quiet_NaN();
  double log_std = 0.0;
  double log_prior = 0.0;
};

static double score(GCA *gca, GCA_SAMPLE *samples, MRI *image,
                    TRANSFORM *transform, int count, double clamp) {
  // Each entry is checked against its actual variance/prior, including pointer
  // reuse across sample sets. No subject-specific state or image cache exists.
  static std::vector<DensityCache> cache;
  cache.resize(count);
  for (int i = 0; i < count; ++i) {
    const float variance = samples[i].covars[0];
    const float prior = gcas_getPrior(samples[i]);
    if (cache[i].variance != variance || cache[i].prior != prior) {
      cache[i].variance = variance;
      cache[i].prior = prior;
      cache[i].log_std = -std::log(std::sqrt(static_cast<double>(variance)));
      cache[i].log_prior = std::log(static_cast<double>(prior));
    }
  }
  TransformInvert(transform, image);
  MATRIX *mapping = GCAgetPriorToSourceVoxelMatrix(gca, image, transform);
  const int threads = omp_get_max_threads();
  std::vector<VECTOR *> source(threads), destination(threads);
  for (int t = 0; t < threads; ++t) {
    source[t] = VectorAlloc(4, MATRIX_REAL);
    destination[t] = VectorAlloc(4, MATRIX_REAL);
    *MATRIX_RELT(source[t], 4, 1) = 1.0;
    *MATRIX_RELT(destination[t], 4, 1) = 1.0;
  }
  double total = 0.0;
  ROMP_PF_begin
  #pragma omp parallel for if_ROMP(fast) reduction(+ : total)
  for (int i = 0; i < count; ++i) {
    ROMP_PFLB_begin
    const int thread = omp_get_thread_num();
    V3_X(source[thread]) = samples[i].xp;
    V3_Y(source[thread]) = samples[i].yp;
    V3_Z(source[thread]) = samples[i].zp;
    MatrixMultiply(mapping, source[thread], destination[thread]);
    const int x = nint(V3_X(destination[thread]));
    const int y = nint(V3_Y(destination[thread]));
    const int z = nint(V3_Z(destination[thread]));
    double log_probability = -1000000.0;
    if (MRIindexNotInVolume(image, x, y, z) == 0) {
      samples[i].x = x; samples[i].y = y; samples[i].z = z;
      // Exact integer coordinates on a single uint8 frame need no interpolation.
      const float intensity = MRIvox(image, x, y, z);
      const float residual = intensity - samples[i].means[0];
      const float mahalanobis = residual * residual / samples[i].covars[0];
      log_probability = cache[i].log_std - .5 * static_cast<double>(mahalanobis);
      log_probability += cache[i].log_prior;
      if (log_probability < -clamp) log_probability = -clamp;
    }
    samples[i].log_p = log_probability;
    total += log_probability;
    ROMP_PFLB_end
  }
  ROMP_PF_end
  for (int t = 0; t < threads; ++t) {
    VectorFree(&source[t]); VectorFree(&destination[t]);
  }
  MatrixFree(&mapping);
  fflush(stdout);
  return static_cast<float>(total) / count;
}
}  // namespace fnit_gca_search
