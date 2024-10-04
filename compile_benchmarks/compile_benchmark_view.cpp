#include <Kokkos_Core.hpp>

void benchmarkSetView() {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  data(1, 1, 1, 1, 1, 1, 1, 1) = 10;
}
