#include "mrisurf.h"
#include <cstdio>

extern "C" int fnit_replace_white_state(void *surface, const char *path) {
  auto *mris = static_cast<MRIS *>(surface);
  FILE *file = std::fopen(path, "rb");
  if (!file || !mris || mris->nvertices != 106622) return 1;
  for (int vertex = 0; vertex < mris->nvertices; ++vertex) {
    float state[6];
    if (std::fread(state, sizeof(float), 6, file) != 6) {
      std::fclose(file);
      return 2;
    }
    VERTEX &v = mris->vertices[vertex];
    if (v.ripflag) continue;
    v.x = state[0];
    v.y = state[1];
    v.z = state[2];
    v.nx = state[3];
    v.ny = state[4];
    v.nz = state[5];
  }
  const int excess = std::fgetc(file);
  std::fclose(file);
  return excess == EOF ? 0 : 3;
}
