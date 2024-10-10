#include <cmath>

#include <Kokkos_Core.hpp>

#include "brak/wrapper_subview.hpp"

#include "utils.hpp"

using View = Kokkos::View<double ***, Kokkos::HostSpace>;
using ViewWrapped = brak::WrapperSubview<View>;

void solve(ViewWrapped field, ViewWrapped fieldTemp,
           unsigned const iterationMax, double const residualMin,
           double const coeff, unsigned const sizeX, unsigned const sizeY,
           unsigned const sizeZ) {
  double residual = 10;

  // initialize
  for (unsigned j = 0; j < sizeY; j++)
    for (unsigned k = 0; k < sizeZ; k++) {
      field[0][j][k] = 1;
    }

  for (unsigned i = 1; i < sizeX; i++)
    for (unsigned j = 0; j < sizeY; j++)
      for (unsigned k = 0; k < sizeY; k++) {
        field[i][j][k] = 0;
      }

  // iteration loop
  for (unsigned iteration = 1; iteration <= iterationMax; iteration++) {
    // check residual
    if (residual <= residualMin)
      return;

    // compute new field
    for (unsigned i = 1; i < sizeX - 1; i++)
      for (unsigned j = 1; j < sizeY - 1; j++)
        for (unsigned k = 1; k < sizeZ - 1; k++) {
          fieldTemp[i][j][k] =
              field[i][j][k] +
              coeff * (-6 * field[i][j][k] + field[i + 1][j][k] +
                       field[i - 1][j][k] + field[i][j + 1][k] +
                       field[i][j - 1][k] + field[i][j][k + 1] +
                       field[i][j][k - 1]);
        }

    // compute residual
    residual = 0;
    for (unsigned i = 1; i < sizeX - 1; i++)
      for (unsigned j = 1; j < sizeY - 1; j++)
        for (unsigned k = 1; k < sizeZ - 1; k++) {
          residual =
              std::max(residual, std::abs(fieldTemp[i][j][k] - field[i][j][k]));
        }

    // swap fields
    for (unsigned i = 1; i < sizeX - 1; i++)
      for (unsigned j = 1; j < sizeY - 1; j++)
        for (unsigned k = 1; k < sizeZ - 1; k++) {
          field[i][j][k] = fieldTemp[i][j][k];
        }

    display(iteration, residual, 100);
  }
}

int main() {
  unsigned const sizeX = 50;
  unsigned const sizeY = 50;
  unsigned const sizeZ = 50;

  double const coeff = 0.1;
  unsigned const iterationMax = 10000;
  double const residualMin = 1e-4;

  Kokkos::ScopeGuard kokkos;

  View field{"field", sizeX, sizeY, sizeZ};
  View fieldTemp{"field_temp", sizeX, sizeY, sizeZ};
  ViewWrapped fieldWrapped{field};
  ViewWrapped fieldTempWrapped{fieldTemp};

  solve(fieldWrapped, fieldTempWrapped, iterationMax, residualMin, coeff, sizeX,
        sizeY, sizeZ);
}
