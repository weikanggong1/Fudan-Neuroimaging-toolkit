#pragma once
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace fnit_n4_profile {
inline std::string prefix;
struct Record { double seconds = 0; unsigned int calls = 0; };
inline std::map<std::string, Record> records;
inline std::set<unsigned long> captured;
struct Scope {
  std::string name;
  std::chrono::steady_clock::time_point started;
  explicit Scope(std::string label) : name(std::move(label)), started(std::chrono::steady_clock::now()) {}
  ~Scope() { auto &r = records[name]; r.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(); ++r.calls; }
};
template<class Image> void raw(const Image *image, const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  const auto bytes = image->GetBufferedRegion().GetNumberOfPixels() * sizeof(typename Image::PixelType);
  out.write(reinterpret_cast<const char *>(image->GetBufferPointer()), bytes);
  if (!out) throw std::runtime_error("diagnostic raw export failed");
}
template<class Field, class Lattice> void capture(const Field *field, const Lattice *phi) {
  const auto size = phi->GetLargestPossibleRegion().GetSize();
  if (!captured.insert(size[0]).second) return;
  Scope timer("diagnostic_export");
  const auto stem = prefix + ".cp" + std::to_string(size[0]);
  raw(field, stem + ".residual.raw");
  raw(phi, stem + ".phi.raw");
  std::ofstream out(stem + ".json"); out.precision(17);
  const auto shape = field->GetLargestPossibleRegion().GetSize();
  const auto spacing = field->GetSpacing();
  const auto origin = field->GetOrigin();
  out << "{\"field_shape\":[" << shape[0] << ',' << shape[1] << ',' << shape[2]
      << "],\"control_shape\":[" << size[0] << ',' << size[1] << ',' << size[2]
      << "],\"spacing\":[" << spacing[0] << ',' << spacing[1] << ',' << spacing[2]
      << "],\"origin\":[" << origin[0] << ',' << origin[1] << ',' << origin[2]
      << "],\"dtype\":\"float32\",\"order\":\"Fortran x-fast\",\"mask\":\"all ones, no confidence\"}\n";
}
inline void write_report() {
  std::ofstream out(prefix + ".internal.json"); out.precision(17); out << '{'; bool first = true;
  for (const auto &entry : records) { if (!first) out << ','; first = false;
    out << '"' << entry.first << "\":{\"seconds\":" << entry.second.seconds << ",\"calls\":" << entry.second.calls << '}'; }
  out << "}\n";
}
}  // namespace fnit_n4_profile
