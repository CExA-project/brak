#include <Kokkos_Core.hpp>

#include <brak/wrapper_array.hpp>

void benchmarkSetWrapper() {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperArray dataWrapper{data};

  dataWrapper[1][1][1][1][1][1][1][1] = 10;
}
