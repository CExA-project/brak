#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/wrapper_array.hpp>
#include <brak/wrapper_subview.hpp>

const double coeff = 0.1;

void benchmark_set_wrapper_subview(benchmark::State &state) {
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 30, 30, 30};
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      dataTemp{"data temp", 30, 30, 30};
  brak::WrapperSubview dataWrapper{data};
  brak::WrapperSubview dataTempWrapper{dataTemp};

  dataWrapper[14][14][14] = 1;

  while (state.KeepRunning()) {
    for (unsigned i = 1; i < data.extent(0) - 1; i++)
      for (unsigned j = 1; j < data.extent(1) - 1; j++)
        for (unsigned k = 1; k < data.extent(2) - 1; k++) {
          dataTempWrapper[i][j][k] =
              dataWrapper[i][j][k] +
              coeff * (-6 * dataWrapper[i][j][k] + dataWrapper[i - 1][j][k] +
                       dataWrapper[i + 1][j][k] + dataWrapper[i][j - 1][k] +
                       dataWrapper[i][j + 1][k] + dataWrapper[i][j][k - 1] +
                       dataWrapper[i][j][k + 1]);
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
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 30, 30, 30};
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      dataTemp{"data temp", 30, 30, 30};
  brak::WrapperArray dataWrapper{data};
  brak::WrapperArray dataTempWrapper{dataTemp};

  dataWrapper[14][14][14] = 1;

  while (state.KeepRunning()) {
    for (unsigned i = 1; i < data.extent(0) - 1; i++)
      for (unsigned j = 1; j < data.extent(1) - 1; j++)
        for (unsigned k = 1; k < data.extent(2) - 1; k++) {
          dataTempWrapper[i][j][k] =
              dataWrapper[i][j][k] +
              coeff * (-6 * dataWrapper[i][j][k] + dataWrapper[i - 1][j][k] +
                       dataWrapper[i + 1][j][k] + dataWrapper[i][j - 1][k] +
                       dataWrapper[i][j + 1][k] + dataWrapper[i][j][k - 1] +
                       dataWrapper[i][j][k + 1]);
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
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 30, 30, 30};
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      dataTemp{"data temp", 30, 30, 30};

  data(14, 14, 14) = 1;

  while (state.KeepRunning()) {
    for (unsigned i = 1; i < data.extent(0) - 1; i++)
      for (unsigned j = 1; j < data.extent(1) - 1; j++)
        for (unsigned k = 1; k < data.extent(2) - 1; k++) {
          dataTemp(i, j, k) =
              data(i, j, k) + coeff * (-6 * data(i, j, k) + data(i - 1, j, k) +
                                       data(i + 1, j, k) + data(i, j - 1, k) +
                                       data(i, j + 1, k) + data(i, j, k - 1) +
                                       data(i, j, k + 1));
        }

    for (unsigned i = 0; i < data.extent(0); i++)
      for (unsigned j = 0; j < data.extent(1); j++)
        for (unsigned k = 0; k < data.extent(2); k++) {
          data(i, j, k) = dataTemp(i, j, k);
        }
  }
}

BENCHMARK(benchmark_set_view);

void benchmark_set_view_unmanaged(benchmark::State &state) {
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 30, 30, 30};
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space>
      dataTemp{"data temp", 30, 30, 30};

  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      dataUnmanaged(data);
  Kokkos::View<double ***, Kokkos::DefaultHostExecutionSpace::memory_space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      dataTempUnmanaged(dataTemp);

  dataUnmanaged(14, 14, 14) = 1;

  while (state.KeepRunning()) {
    for (unsigned i = 1; i < dataUnmanaged.extent(0) - 1; i++)
      for (unsigned j = 1; j < dataUnmanaged.extent(1) - 1; j++)
        for (unsigned k = 1; k < dataUnmanaged.extent(2) - 1; k++) {
          dataTempUnmanaged(i, j, k) =
              dataUnmanaged(i, j, k) +
              coeff * (-6 * dataUnmanaged(i, j, k) +
                       dataUnmanaged(i - 1, j, k) + dataUnmanaged(i + 1, j, k) +
                       dataUnmanaged(i, j - 1, k) + dataUnmanaged(i, j + 1, k) +
                       dataUnmanaged(i, j, k - 1) + dataUnmanaged(i, j, k + 1));
        }

    for (unsigned i = 0; i < dataUnmanaged.extent(0); i++)
      for (unsigned j = 0; j < dataUnmanaged.extent(1); j++)
        for (unsigned k = 0; k < dataUnmanaged.extent(2); k++) {
          dataUnmanaged(i, j, k) = dataTempUnmanaged(i, j, k);
        }
  }
}

BENCHMARK(benchmark_set_view_unmanaged);
