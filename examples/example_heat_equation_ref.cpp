#include <cmath>

#include "utils.hpp"

void createFields(double ***&field, double ***&fieldTemp, unsigned const sizeX,
                  unsigned const sizeY, unsigned const sizeZ) {
  field = new double **[sizeX];
  for (unsigned i = 0; i < sizeX; i++) {
    field[i] = new double *[sizeY];
    for (unsigned j = 0; j < sizeY; j++) {
      field[i][j] = new double[sizeZ];
      for (unsigned k = 0; k < sizeZ; k++) {
        field[i][j][k] = 0.0;
      }
    }
  }

  fieldTemp = new double **[sizeX];
  for (unsigned i = 0; i < sizeX; i++) {
    fieldTemp[i] = new double *[sizeY];
    for (unsigned j = 0; j < sizeY; j++) {
      fieldTemp[i][j] = new double[sizeZ];
      for (unsigned k = 0; k < sizeZ; k++) {
        fieldTemp[i][j][k] = 0.0;
      }
    }
  }
}

void deleteFields(double ***&field, double ***&fieldTemp, unsigned const sizeX,
                  unsigned const sizeY, unsigned const sizeZ) {
  for (unsigned i = 0; i < sizeX; i++) {
    for (unsigned j = 0; j < sizeY; j++) {
      delete[] field[i][j];
    }
    delete[] field[i];
  }
  delete[] field;

  for (unsigned i = 0; i < sizeX; i++) {
    for (unsigned j = 0; j < sizeY; j++) {
      delete[] fieldTemp[i][j];
    }
    delete[] fieldTemp[i];
  }
  delete[] fieldTemp;
}

void solve(double ***&field, double ***&fieldTemp, unsigned const iterationMax,
           double const residualMin, double const coeff, unsigned const sizeX,
           unsigned const sizeY, unsigned const sizeZ) {
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

  double ***field;
  double ***fieldTemp;

  createFields(field, fieldTemp, sizeX, sizeY, sizeZ);
  solve(field, fieldTemp, iterationMax, residualMin, coeff, sizeX, sizeY,
        sizeZ);
  deleteFields(field, fieldTemp, sizeX, sizeY, sizeZ);
}
