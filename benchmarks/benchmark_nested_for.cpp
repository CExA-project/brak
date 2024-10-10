#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/wrapper_array.hpp>
#include <brak/wrapper_subview.hpp>

void benchmark_set_wrapper_subview(benchmark::State &state) {
  Kokkos::View<int ***, Kokkos::HostSpace> data{"data", 30, 30, 30};
  Kokkos::View<int ***, Kokkos::HostSpace> dataTemp{"data", 30, 30, 30};
  brak::WrapperSubview dataWrapper{data};
  brak::WrapperSubview dataTempWrapper{dataTemp};

  while (state.KeepRunning()) {
      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  dataTempWrapper[i][j][k] = dataWrapper[i][j][k] + i + j + k;
              }

      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  dataWrapper[i][j][k] = dataTempWrapper[i][j][k];
              }
  }
}

BENCHMARK(benchmark_set_wrapper_subview);

void benchmark_set_wrapper_array(benchmark::State &state) {
  Kokkos::View<int ***, Kokkos::HostSpace> data{"data", 30, 30, 30};
  Kokkos::View<int ***, Kokkos::HostSpace> dataTemp{"data", 30, 30, 30};
  brak::WrapperArray dataWrapper{data};
  brak::WrapperArray dataTempWrapper{dataTemp};

  while (state.KeepRunning()) {
      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  dataTempWrapper[i][j][k] = dataWrapper[i][j][k] + i + j + k;
              }

      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  dataWrapper[i][j][k] = dataTempWrapper[i][j][k];
              }
  }
}

BENCHMARK(benchmark_set_wrapper_array);

void benchmark_set_view(benchmark::State &state) {
  Kokkos::View<int ***, Kokkos::HostSpace> data{"data", 30, 30, 30};
  Kokkos::View<int ***, Kokkos::HostSpace> dataTemp{"data", 30, 30, 30};

  while (state.KeepRunning()) {
      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  dataTemp(i, j, k) = data(i, j, k) + i + j + k;
              }

      for (unsigned i = 0; i < data.extent(0); i++)
          for (unsigned j = 0; j < data.extent(1); j++)
              for (unsigned k = 0; k < data.extent(2); k++) {
                  data(i, j, k) = dataTemp(i, j, k);
              }
  }
}

BENCHMARK(benchmark_set_view);
