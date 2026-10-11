/* Diagnostic-only qsort interposer for pinned mris_fix_topology 8.2.
 * EDGE is {int vno1, int vno2, float len, short used, padding} (16 bytes).
 * Historical defaults are 6456 or 26560 edges. TOPO_EDGE_COUNT can select
 * another independently computed complete candidate table; no subject IDs.
 * Capture the table immediately before/after native qsort, then stop before GA.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int (*compare_fn)(const void *, const void *);
typedef void (*qsort_fn)(void *, size_t, size_t, compare_fn);

static void capture(const char *prefix, const char *suffix, const void *base,
                    size_t count, size_t size) {
  char path[4096];
  snprintf(path, sizeof(path), "%s.%s.bin", prefix, suffix);
  FILE *file = fopen(path, "wb");
  if (!file || fwrite(base, size, count, file) != count) {
    fprintf(stderr, "topology edge capture failed: %s\n", path);
    _exit(2);
  }
  fclose(file);
}

void qsort(void *base, size_t count, size_t size, compare_fn compare) {
  static qsort_fn native_qsort;
  if (!native_qsort) native_qsort = (qsort_fn)dlsym(RTLD_NEXT, "qsort");
  const char *prefix = getenv("TOPO_EDGE_CAPTURE");
  const char *requested_count = getenv("TOPO_EDGE_COUNT");
  size_t wanted = requested_count ? strtoull(requested_count, NULL, 10) : 0;
  int matching_count = requested_count ? wanted > 0 && count == wanted
                                      : count == 6456 || count == 26560;
  if (prefix && size == 16 && matching_count) {
    capture(prefix, "before", base, count, size);
    native_qsort(base, count, size, compare);
    capture(prefix, "after", base, count, size);
    fprintf(stderr, "captured %zu native topology edges; stopping before GA\n", count);
    fflush(NULL);
    _exit(0);
  }
  native_qsort(base, count, size, compare);
}
