#ifndef __BRAK_WRAPPER_SUBVIEW_HPP__
#define __BRAK_WRAPPER_SUBVIEW_HPP__

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "kokkos_view.hpp"

namespace brak {

/**
 * Wrapper based on subviews.
 * @tparam View Type of the input view.
 */
template <typename View> class WrapperSubview {
  /**
   * Marker to identify the class.
   */
  using WrapperSubviewType = WrapperSubview<View>;

  /**
   * Wrapped view.
   * We use the fact that a subview is actually a view.
   */
  View mData;
  static_assert(Kokkos::is_view<View>::value);

public:
  /**
   * Construct a wrapper from a view.
   * @param data Input view.
   */
  KOKKOS_FUNCTION
  explicit WrapperSubview(View const data) : mData(data) {}

  /**
   * Get the rank of the wrapped view.
   * @return Rank of the wrapped view.
   */
  KOKKOS_FUNCTION
  static std::size_t constexpr getRank() { return View::rank(); }

  /**
   * Create a wrapped subview with a rank lowered by 1.
   * @param index Left-most index to extract from the wrapped view.
   * @return A wrapped subview or a reference to a scalar if the wrapped view
   * has a dimension of 1.
   */
  KOKKOS_FUNCTION
  decltype(auto) operator[](std::size_t const index) const {
    // NOTE The `decltype(auto)` allows to return either a value (a new instance
    // of the class) or a reference to a value (a scalar).

    if constexpr (getRank() > 1) {
      // return wrapper of the subview
      auto subview = getSubview(index);
      using ViewCurrent = decltype(subview);

      // make the view unmanaged at its first access
      using ViewNext =
          std::conditional_t<ViewCurrent::traits::memory_traits::is_unmanaged,
                             ViewCurrent,
                             kokkos_addendum::make_unmanaged<ViewCurrent>>;
      // NOTE This disables reference counting on CPU for each view created in
      // each successive wrapper retrieved, which greatly improves performance.
      // On GPU, reference counting of views is already disabled by default.

      return WrapperSubview<ViewNext>(subview);
    } else {
      // return a reference to a scalar
      return mData(index);
    }
  }

  /**
   * Directly access to a scalar value.
   * @tparam IndicesType Type of the indices. They will be casted to
   * `std::size_t`.
   * @param indices Pack of indices. The number of indices must match the rank
   * of the current wrapper.
   * @return Reference to a scalar of the view at the given indices.
   */
  template <typename... IndicesType>
  KOKKOS_FUNCTION constexpr auto &operator()(IndicesType const... indices) const {
    static_assert(sizeof...(indices) == getRank(), "Rank mismatch");

    // return reference to scalar
    return mData(static_cast<std::size_t>(indices)...);
  }

  /**
   * Defer the wrapper to the pointer data in the wrapped view.
   * @return Raw pointer to the wrapped data.
   * @note This method may give access to data that are not contiguous in
   * memory and lead to unpredictable behaviors.
   */
  KOKKOS_FUNCTION
  typename View::value_type *operator*() { return mData.data(); }

  /**
   * Retrieve the wrapped view.
   * @return Copy of the wrapped view.
   */
  KOKKOS_FUNCTION
  View getView() { return mData; }

private:
  /**
   * Extract a subview of accurate rank.
   * @param index Left-most index to extract from the wrapped view.
   * @return Subview.
   * @note Can only be used when the rank of the wrapped view is between 2
   * and 8.
   */
  KOKKOS_FUNCTION
  auto getSubview(std::size_t const index) const {
    static_assert(getRank() <= 8, "Rank of view too large");
    static_assert(getRank() > 1, "Rank of view too small");

    // NOTE It's probably possible to write templated code to not handle all
    // the dimensions manually, but it's a lot of troubles for just 8
    // dimensions.

    using Kokkos::ALL;

    if constexpr (getRank() == 8) {
      // return a subview of rank 7
      return Kokkos::subview(mData, index, ALL, ALL, ALL, ALL, ALL, ALL, ALL);
    } else if constexpr (getRank() == 7) {
      // return a subview of rank 6
      return Kokkos::subview(mData, index, ALL, ALL, ALL, ALL, ALL, ALL);
    } else if constexpr (getRank() == 6) {
      // return a subview of rank 5
      return Kokkos::subview(mData, index, ALL, ALL, ALL, ALL, ALL);
    } else if constexpr (getRank() == 5) {
      // return a subview of rank 4
      return Kokkos::subview(mData, index, ALL, ALL, ALL, ALL);
    } else if constexpr (getRank() == 4) {
      // return a subview of rank 3
      return Kokkos::subview(mData, index, ALL, ALL, ALL);
    } else if constexpr (getRank() == 3) {
      // return a subview of rank 2
      return Kokkos::subview(mData, index, ALL, ALL);
    } else {
      // return a subview of rank 1
      return Kokkos::subview(mData, index, ALL);
    }
  }
};

} // namespace brak

#endif // ifndef __BRAK_WRAPPER_SUBVIEW_HPP__
