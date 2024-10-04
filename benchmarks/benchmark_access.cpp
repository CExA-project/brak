#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/wrapper_array.hpp>
#include <brak/wrapper_subview.hpp>

void benchmark_set_wrapper_subview(benchmark::State &state) {
  Kokkos::View<int ********, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperSubview dataWrapper{data};

  while (state.KeepRunning()) {
    dataWrapper[1][1][1][1][1][1][1][1] = 10;
  }
}

BENCHMARK(benchmark_set_wrapper_subview);

void benchmark_set_wrapper_array(benchmark::State &state) {
  Kokkos::View<int ********, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperArray dataWrapper{data};

  while (state.KeepRunning()) {
    dataWrapper[1][1][1][1][1][1][1][1] = 10;
  }
}

BENCHMARK(benchmark_set_wrapper_array);

void benchmark_set_view(benchmark::State &state) {
  Kokkos::View<int ********, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  while (state.KeepRunning()) {
    data(1, 1, 1, 1, 1, 1, 1, 1) = 10;
  }
}

BENCHMARK(benchmark_set_view);
