#ifndef __BRAK_VIEW_HPP__
#define __BRAK_VIEW_HPP__

#include <Kokkos_Core.hpp>

namespace brak {
namespace impl {

/**
 * Recreate a view with the unmanaged memory trait.
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

} // namespace impl
} // namespace brak

#endif // ifndef __BRAK_VIEW_HPP__
