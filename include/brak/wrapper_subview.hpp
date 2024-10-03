#ifndef __BRAK_WRAPPER_SUBVIEW_HPP__
#define __BRAK_WRAPPER_SUBVIEW_HPP__

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace brak {

/**
 * Wrapper based on subviews.
 */
template <typename View,
          typename Enabled = std::enable_if<Kokkos::is_view<View>::value>>
class WrapperSubview {
  /**
   * Marker to identify the class.
   */
  using WrapperSubviewType = WrapperSubview<View>;

  /**
   * Wrapped view.
   * We use the fact that a subview is actually a view.
   */
  View mData;

public:
  /**
   * Construct a wrapper from a view.
   * @tparam View Type of the input view.
   * @param data Input view.
   */
  explicit WrapperSubview(View const data) : mData(data) {}

  /**
   * Get the rank of the wrapped view.
   * @return Rank of the wrapped view.
   */
  static std::size_t constexpr getRank() { return View::rank(); }

  /**
   * Create a wrapped subview with a rank lowered by 1.
   * @param index Left-most index to extract from the wrapped view.
   * @return A wrapped subview or a reference to a scalar if the wrapped view
   * has a dimension of 1.
   */
  decltype(auto) operator[](std::size_t const index) const {
    // NOTE The `decltype(auto)` allows to return either a value (a new instance
    // of the class) or a reference to a value (a scalar).

    if constexpr (getRank() > 1) {
      // return wrapper of the subview
      auto subview = getSubview(index);
      return WrapperSubview<decltype(subview)>(subview);
      // NOTE The manual passing of the subview type is necessary for NVCC that
      // doesn't do CTAD. (Instead, one would just have used `brak::` to call
      // the template class and not the current one.)
    } else {
      // return a reference to a scalar
      return mData(index);
    }
  }

  /**
   * Defer the wrapper to the pointer data in the wrapped view.
   * @return Raw pointer to the wrapped data.
   * @note This method may give access to data that are not contiguous in
   * memory and lead to unpredictible behaviors.
   */
  typename View::value_type *operator*() { return mData.data(); }

  /**
   * Retrieve the wrapped view.
   * @return Copy of the wrapped view.
   */
  View getView() { return mData; }

private:
  /**
   * Extract a subview of accurate rank.
   * @param index Left-most index to extract from the wrapped view.
   * @return Subview.
   * @note Can only be used when the rank of the wrapped view is between 2
   * and 8.
   */
  auto getSubview(std::size_t const index) const {
    static_assert(getRank() <= 8, "Rank of view too large");
    static_assert(getRank() > 1, "Rank of view too small");

    // NOTE It's probably possible to write templated code to not handle all
    // the dimensions manually, but it's a lot of troubles for just 8
    // dimensions.

    auto &ALL = Kokkos::ALL;

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
