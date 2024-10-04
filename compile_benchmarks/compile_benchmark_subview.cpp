#include <Kokkos_Core.hpp>

#include <brak/wrapper_subview.hpp>

void benchmarkSetWrapper() {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperSubview dataWrapper{data};

  dataWrapper[1][1][1][1][1][1][1][1] = 10;
}
