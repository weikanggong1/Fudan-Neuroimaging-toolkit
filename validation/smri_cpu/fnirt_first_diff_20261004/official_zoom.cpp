// Isolated FSL ZoomField checkpoint, using the real first accepted coefficients.
#include <fstream>
#include <iomanip>
#include "basisfield/splinefield.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream input(argv[1]);
  std::ofstream output(argv[2]);
  output << std::setprecision(17);
  for (int component=0; component<3; ++component) {
    BASISFIELD::splinefield original({24,28,24}, {8.,8.,8.}, {5,5,5}, 3);
    NEWMAT::ColumnVector coefficients(original.CoefSz());
    for (int i=1; i<=coefficients.Nrows(); ++i) {
      if (!(input >> coefficients(i))) return 3;
    }
    original.SetCoef(coefficients);
    auto zoomed=original.ZoomField({46,55,46}, {4.,4.,4.});
    auto result=zoomed->GetCoef();
    for (int i=1; i<=result->Nrows(); ++i) output << (*result)(i) << "\n";
  }
  double scale;
  if (!(input >> scale)) return 4;
  output << scale << "\n";
  return 0;
}
