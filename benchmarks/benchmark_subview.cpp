#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/subview.hpp>

void benchmarkSetWrapper(benchmark::State &state) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::BracketsWrapperSubview dataWrapper{data};

  while (state.KeepRunning()) {
    dataWrapper[1][1][1][1][1][1][1][1] = 10;
  }
}

BENCHMARK(benchmarkSetWrapper);

void benchmarkSetView(benchmark::State &state) {
  [[maybe_unused]] Kokkos::View<int ********> data{"data", 2, 2, 2, 2,
                                                   2,      2, 2, 2};

  while (state.KeepRunning()) {
    data(1, 1, 1, 1, 1, 1, 1, 1) = 10;
  }
}

BENCHMARK(benchmarkSetView);
