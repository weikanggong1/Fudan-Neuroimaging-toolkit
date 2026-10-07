// FNIT: ordered FP32 Jacobi averaging with one local persistent OpenMP team.
// Independently written kernel; no upstream imaging source or binary bundled.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>
#include <omp.h>
#include <xmmintrin.h>

extern "C" int fnit_average_persistent_abi_version() { return 1; }

extern "C" int fnit_average_persistent(
    const float* gradient, const std::int64_t* neighbors,
    const std::int64_t* degrees, const float* reciprocals,
    std::int64_t count, std::int64_t width, std::int64_t iterations,
    int threads, float* output, int* actual_threads) {
  if (count < 0 || width < 0 || iterations < 0 || threads < 1) return 1;
  if (count > std::numeric_limits<std::int64_t>::max() / 3 ||
      (width && count > std::numeric_limits<std::int64_t>::max() / width)) return 1;
  if (!actual_threads || (count && (!gradient || !degrees || !reciprocals || !output)) ||
      (count && width && !neighbors)) return 1;
  *actual_threads = 0;
  // Preserve an explicit caller floating-point policy by leaving it to NumBa.
  constexpr unsigned int policy_mask = 0x8040u | 0x6000u; // FTZ, DAZ, rounding
  if (_mm_getcsr() & policy_mask) return 6;
  // Validate all active indices before touching any output or starting a team.
  for (std::int64_t vertex = 0; vertex < count; ++vertex) {
    if (degrees[vertex] < 0 || degrees[vertex] > width) return 3;
    for (std::int64_t column = 0; column < degrees[vertex]; ++column) {
      auto neighbor = neighbors[vertex * width + column];
      if (neighbor < 0 || neighbor >= count) return 4;
    }
  }
  if (!count) return 0;
  try {
    std::vector<float> first(gradient, gradient + count * 3);
    std::vector<float> second(count * 3);
    float* a = first.data();
    float* b = second.data();
    int incompatible_policy = 0;
    // num_threads applies only to this region. No global OpenMP setter.
    #pragma omp parallel num_threads(threads) shared(a, b, incompatible_policy)
    {
      if (omp_get_thread_num() == 0) *actual_threads = omp_get_num_threads();
      if (_mm_getcsr() & policy_mask) {
        #pragma omp atomic write
        incompatible_policy = 1;
      }
      #pragma omp barrier
      if (!incompatible_policy) for (std::int64_t round = 0; round < iterations; ++round) {
        const float* current = (round % 2 == 0) ? a : b;
        float* following = (round % 2 == 0) ? b : a;
        // Implicit barrier completes each entire Jacobi round before reuse.
        #pragma omp for schedule(static)
        for (std::int64_t vertex = 0; vertex < count; ++vertex) {
          float x = current[3 * vertex];
          float y = current[3 * vertex + 1];
          float z = current[3 * vertex + 2];
          for (std::int64_t column = 0; column < degrees[vertex]; ++column) {
            const auto offset = 3 * neighbors[vertex * width + column];
            x = x + current[offset];
            y = y + current[offset + 1];
            z = z + current[offset + 2];
          }
          following[3 * vertex] = x * reciprocals[vertex];
          following[3 * vertex + 1] = y * reciprocals[vertex];
          following[3 * vertex + 2] = z * reciprocals[vertex];
        }
      }
    }
    if (incompatible_policy) return 6;
    const float* result = (iterations % 2 == 0) ? a : b;
    std::copy(result, result + count * 3, output);
    return 0;
  } catch (const std::bad_alloc&) {
    return 2;
  } catch (...) {
    return 5;
  }
}
