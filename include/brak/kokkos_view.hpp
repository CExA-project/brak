#ifndef __BRAK_KOKKOS_VIEW_HPP__
#define __BRAK_KOKKOS_VIEW_HPP__

#include <Kokkos_Core.hpp>

namespace kokkos_addendum {

/**
 * Recreate a view with the unmanaged memory trait.
 * This should be updated to follow any update in Kokkos view structures.
 * @tparam View Source view.
 */
template <typename View>
using make_unmanaged = Kokkos::View<
    typename View::traits::data_type, typename View::traits::array_layout,
    typename View::traits::device_type, typename View::traits::hooks_policy,
    Kokkos::MemoryTraits<
        Kokkos::Unmanaged |
        (View::traits::memory_traits::is_random_access ? Kokkos::RandomAccess
                                                       : 0) |
        (View::traits::memory_traits::is_atomic ? Kokkos::Atomic : 0) |
        (View::traits::memory_traits::is_restrict ? Kokkos::Restrict : 0) |
        (View::traits::memory_traits::is_aligned ? Kokkos::Aligned : 0)>>;

} // namespace kokkos_addendum

#endif // ifndef __BRAK_KOKKOS_VIEW_HPP__
