#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

#include <brak/wrapper_array.hpp>
#include <brak/wrapper_subview.hpp>

void benchmark_set_wrapper_subview(benchmark::State &state) {
  Kokkos::View<int ******> data{"data", 30, 30, 30, 30, 30, 30};
  brak::WrapperSubview dataWrapper{data};

  while (state.KeepRunning()) {
    Kokkos::parallel_for(
        "benchmark_set_wrapper_subview",
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0, 0}, {30, 30, 30, 30, 30, 30}),
        KOKKOS_LAMBDA(int const i, int const j, int const k, int const l,
                      int const m, int const n) {
          dataWrapper[i][j][k][l][m][n] = i + j + k + l + m + n;
        });
    Kokkos::fence();
  }
}

BENCHMARK(benchmark_set_wrapper_subview);

void benchmark_set_wrapper_array(benchmark::State &state) {
  Kokkos::View<int ******> data{"data", 30, 30, 30, 30, 30, 30};
  brak::WrapperArray dataWrapper{data};

  while (state.KeepRunning()) {
    Kokkos::parallel_for(
        "benchmark_set_wrapper_array",
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0, 0}, {30, 30, 30, 30, 30, 30}),
        KOKKOS_LAMBDA(int const i, int const j, int const k, int const l,
                      int const m, int const n) {
          dataWrapper[i][j][k][l][m][n] = i + j + k + l + m + n;
        });
    Kokkos::fence();
  }
}

BENCHMARK(benchmark_set_wrapper_array);

void benchmark_set_view(benchmark::State &state) {
  Kokkos::View<int ******> data{"data", 30, 30, 30, 30, 30, 30};

  while (state.KeepRunning()) {
    Kokkos::parallel_for(
        "benchmark_set_view",
        Kokkos::MDRangePolicy({0, 0, 0, 0, 0, 0}, {30, 30, 30, 30, 30, 30}),
        KOKKOS_LAMBDA(int const i, int const j, int const k, int const l,
                      int const m, int const n) {
          data(i, j, k, l, m, n) = i + j + k + l + m + n;
        });
    Kokkos::fence();
  }
}

BENCHMARK(benchmark_set_view);
