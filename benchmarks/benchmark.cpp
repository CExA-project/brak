#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/subview.hpp>
#include <brak/compute.hpp>

void benchmark_set_wrapper_subview(benchmark::State &state) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::BracketsWrapperSubview dataWrapper{data};

  while (state.KeepRunning()) {
    dataWrapper[1][1][1][1][1][1][1][1] = 10;
  }
}

BENCHMARK(benchmark_set_wrapper_subview);

void benchmark_set_wrapper_compute(benchmark::State &state) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::BracketsWrapperCompute dataWrapper{data};

  while (state.KeepRunning()) {
    dataWrapper[1][1][1][1][1][1][1][1] = 10;
  }
}

BENCHMARK(benchmark_set_wrapper_compute);

void benchmark_set_view(benchmark::State &state) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  while (state.KeepRunning()) {
    data(1, 1, 1, 1, 1, 1, 1, 1) = 10;
  }
}

BENCHMARK(benchmark_set_view);
