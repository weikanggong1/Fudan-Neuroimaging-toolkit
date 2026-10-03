// FNIT single-input search hotspot. Original optimizer/gradient remain upstream.
// See task_03/README.md for coordinate, dtype and source provenance requirements.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <map>
#include "romp_support.h"

namespace fnit_gca_search {
struct Capabilities {
  Capabilities() {
    const char *query = std::getenv("FNIT_GCA_QUERY_CAPABILITIES");
    if (query && !std::strcmp(query, "1")) {
      std::puts("{\"fnit_gca_cached_search\":true,\"full_native_em\":true,\"version\":2,\"single_input_uint8\":true,\"reduction\":\"upstream_ROMP_partials\",\"activation\":\"FNIT_GCA_SCORER=cpu_cached\"}");
      std::exit(0);
    }
  }
};
static Capabilities capabilities;

struct Statistics {
  bool enabled;
  int fast_allowed = -1, reproducible_allowed = -1;
  unsigned long long density_refreshes = 0;
  std::map<int, unsigned long long> original, cached;
  Statistics() {
    const char *value = std::getenv("FNIT_GCA_DIAGNOSTICS");
    enabled = value && !std::strcmp(value, "1");
  }
  ~Statistics() {
    if (!enabled) return;
    for (const auto &entry : original)
      std::fprintf(stderr, "FNIT_GCA_CALLS backend=original samples=%d calls=%llu total_density_refreshes=%llu fast=%d reproducible=%d\n", entry.first, entry.second, density_refreshes, fast_allowed, reproducible_allowed);
    for (const auto &entry : cached)
      std::fprintf(stderr, "FNIT_GCA_CALLS backend=cpu_cached samples=%d calls=%llu total_density_refreshes=%llu fast=%d reproducible=%d\n", entry.first, entry.second, density_refreshes, fast_allowed, reproducible_allowed);
  }
};
static Statistics &statistics() { static Statistics value; return value; }
static void record_call(bool cached, int count) {
  Statistics &value = statistics();
  if (!value.enabled) return;
  value.fast_allowed = ROMP_if_parallel1(ROMP_level_fast);
  value.reproducible_allowed = ROMP_if_parallel1(ROMP_level_assume_reproducible);
  ++(cached ? value.cached : value.original)[count];
}

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
  record_call(true, count);
  static std::vector<DensityCache> cache;
  cache.resize(count);
  for (int i = 0; i < count; ++i) {
    const float variance = samples[i].covars[0];
    const float prior = gcas_getPrior(samples[i]);
    if (cache[i].variance != variance || cache[i].prior != prior) {
      if (statistics().enabled) ++statistics().density_refreshes;
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
  // Match upstream gca.cpp's enabled reproducible branch, including its
  // partial ranges, serial per-partial order, final reduction and ROMP level.
  #define ROMP_VARIABLE i
  #define ROMP_LO 0
  #define ROMP_HI count
  #define ROMP_SUMREDUCTION0 total
  #define ROMP_FOR_LEVEL ROMP_level_assume_reproducible
  #ifdef ROMP_SUPPORT_ENABLED
    const int romp_for_line = __LINE__;
  #endif
  #include "romp_for_begin.h"
  ROMP_for_begin
  #define total ROMP_PARTIALSUM(0)
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
  #undef total
  #include "romp_for_end.h"
  for (int t = 0; t < threads; ++t) {
    VectorFree(&source[t]); VectorFree(&destination[t]);
  }
  MatrixFree(&mapping);
  fflush(stdout);
  return static_cast<float>(total) / count;
}
}  // namespace fnit_gca_search
