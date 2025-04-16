#include <Kokkos_Core.hpp>

void benchmarkSetView() {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  Kokkos::View<int ********, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dataUnmanaged(data);

  dataUnmanaged(1, 1, 1, 1, 1, 1, 1, 1) = 10;
}
