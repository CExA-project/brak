#ifndef __BRAK_WRAPPER_ARRAY_HPP__
#define __BRAK_WRAPPER_ARRAY_HPP__

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace brak {

/**
 * Wrapper based on an array of indices.
 */
template <typename View, std::size_t depth = 0,
          typename Enabled = std::enable_if<Kokkos::is_view<View>::value>>
class WrapperArray {
  /**
   * Marker to identify the class.
   */
  using WrapperArrayType = WrapperArray<View, depth>;

  /**
   * Wrapped view.
   */
  View mData;

  /**
   * Array of the indices.
   */
  Kokkos::Array<std::size_t, depth> mIndices;

public:
  /**
   * Construct a wrapper from a view.
   * @tparam View Type of the input view.
   * @tparam depth Current depth of the wrapper.
   * @param data Input view.
   */
  KOKKOS_FUNCTION
  explicit WrapperArray(View const data) : mData(data) {}

  /**
   * Construct a sub-wrapper from a view and a list of indices.
   * @tparam View Type of the input view.
   * @tparam depth Current depth of the wrapper.
   * @param data Input view.
   * @param indices List of indices above the sub-wrapper.
   */
  KOKKOS_FUNCTION
  WrapperArray(View const data,
               Kokkos::Array<std::size_t, depth> const &indices)
      : mData(data), mIndices(indices) {}

  /**
   * Get the current rank of the wrapper.
   * @return Rank of the wrapper.
   */
  KOKKOS_FUNCTION
  static std::size_t constexpr getRank() { return View::rank() - depth; }

  /**
   * Get the rank of the wrapped view.
   * @return Rank of the wrapped view.
   */
  KOKKOS_FUNCTION
  static std::size_t constexpr getRankSource() { return View::rank(); }

  /**
   * Create a sub-wrapper with a rank lowered by 1.
   * @param index Left-most index to extract from the wrapped view.
   * @return A sub-wrapper or a reference to a scalar if the current wrapper
   * has a dimension of 1.
   */
  KOKKOS_FUNCTION
  decltype(auto) operator[](std::size_t const index) const {
    // recreate array of indices
    Kokkos::Array<std::size_t, depth + 1> indices;
    if constexpr (mIndices.size() > 0) {
      for (std::size_t i = 0; i < mIndices.size(); i++) {
        indices[i] = mIndices[i];
      }
    }
    indices[depth] = index;

    if constexpr (getRank() > 1) {
      // make the view unmanaged at its first access
      using ViewNext = std::conditional_t<
          View::traits::memory_traits::is_unmanaged, View,
          Kokkos::View<typename View::traits::data_type,
                       typename View::traits::array_layout,
                       typename View::traits::device_type,
                       typename View::traits::hooks_policy,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
      // NOTE This disables reference counting on CPU for each view created in
      // each successive wrapper retrieved, which greatly improves performance.
      // On GPU, reference counting of views is already disabled by default.

      return WrapperArray<ViewNext, depth + 1>(mData, indices);
    } else {
      return getValue(indices);
    }
  }

  /**
   * Defer the wrapper to the pointer data in the wrapped view.
   * @return Raw pointer to the wrapped data.
   * @note This method may give access to data that are not contiguous in
   * memory and lead to unpredictible behaviors.
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
   * Get the scalar value of the wrapped view.
   * @param indices List of indices above the sub-wrapper.
   * @return Scalar value of the view.
   * @note Can only be used when the rank of the wrapped view is between 1
   * and 8.
   */
  KOKKOS_FUNCTION
  auto &getValue(Kokkos::Array<std::size_t, depth + 1> const &indices) const {
    static_assert(getRankSource() <= 8, "Rank of view too large");
    static_assert(getRankSource() > 0, "Rank of view too small");

    // NOTE It's probably possible to write templated code to not handle all
    // the dimensions manually, but it's a lot of troubles for just 8
    // dimensions.

    auto &i = indices;

    if constexpr (getRankSource() == 8) {
      return mData(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
    } else if constexpr (getRankSource() == 7) {
      return mData(i[0], i[1], i[2], i[3], i[4], i[5], i[6]);
    } else if constexpr (getRankSource() == 6) {
      return mData(i[0], i[1], i[2], i[3], i[4], i[5]);
    } else if constexpr (getRankSource() == 5) {
      return mData(i[0], i[1], i[2], i[3], i[4]);
    } else if constexpr (getRankSource() == 4) {
      return mData(i[0], i[1], i[2], i[3]);
    } else if constexpr (getRankSource() == 3) {
      return mData(i[0], i[1], i[2]);
    } else if constexpr (getRankSource() == 2) {
      return mData(i[0], i[1]);
    } else {
      return mData(i[0]);
    }
  }
};

} // namespace brak

#endif // ifndef __BRAK_WRAPPER_ARRAY_HPP__
