#ifndef __BRAK_WRAPPER_ARRAY_HPP__
#define __BRAK_WRAPPER_ARRAY_HPP__

#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "kokkos_view.hpp"

namespace brak {

/**
 * Wrapper based on an array of indices.
 * @tparam View Type of the input view.
 * @tparam depth Current depth of the wrapper.
 */
template <typename View, std::size_t depth = 0> class WrapperArray {
  /**
   * Marker to identify the class.
   */
  using WrapperArrayType = WrapperArray<View, depth>;

  /**
   * Wrapped view.
   */
  View mData;
  static_assert(Kokkos::is_view<View>::value);

  /**
   * Array of the indices.
   */
  Kokkos::Array<std::size_t, depth> mIndices;

public:
  /**
   * Construct a wrapper from a view.
   * @param data Input view.
   */
  KOKKOS_FUNCTION
  explicit WrapperArray(View const data) : mData(data) {}

  /**
   * Construct a sub-wrapper from a view and a array of indices.
   * @param data Input view.
   * @param indices Array of indices above the sub-wrapper.
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
  constexpr decltype(auto) operator[](std::size_t const index) const {
    // recreate array of indices
    Kokkos::Array<std::size_t, depth + 1> indices = extendIndices(index);

    if constexpr (getRank() > 1) {
      // return wrapper of the view with a new array of indices
      // make the view unmanaged at its first access
      using ViewNext =
          std::conditional_t<View::traits::memory_traits::is_unmanaged, View,
                             kokkos_addendum::make_unmanaged<View>>;
      // NOTE This disables reference counting on CPU for each view created in
      // each successive wrapper retrieved, which greatly improves performance.
      // On GPU, reference counting of views is already disabled by default.

      return WrapperArray<ViewNext, depth + 1>(mData, indices);
    } else {
      // return a reference to a scalar
      return getValue(indices);
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
  KOKKOS_FUNCTION constexpr auto &
  operator()(IndicesType const... indices) const {
    static_assert(sizeof...(indices) == getRank(), "Rank mismatch");

    // merge the two arrays of indices
    Kokkos::Array<std::size_t, getRankSource()> indicesJoined =
        extendIndices(static_cast<std::size_t>(indices)...);
    // NOTE The indices are converted to `std::size_t` as it seems not possible
    // to easily obtain a parameter pack of the same type.

    // return reference to scalar
    return getValue(indicesJoined);
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
   * Recreate an array of indices with a new index.
   * @param index New index to add to the array of indices.
   * @return Extended array of indices.
   */
  KOKKOS_FUNCTION
  constexpr Kokkos::Array<std::size_t, depth + 1>
  extendIndices(std::size_t const index) const {
    return extendIndices(std::make_index_sequence<depth>(), index);
  }

  /**
   * Recreate an array of indices with new indices.
   * @tparam IndicesType Type of the indices.
   * @param indices Pack of indices.
   * @return Extended array of indices.
   */
  template <typename... IndicesType>
  KOKKOS_FUNCTION constexpr Kokkos::Array<std::size_t, getRankSource()>
  extendIndices(IndicesType const... indices) const {
    return extendIndices(std::make_index_sequence<depth>(), indices...);
  }

  /**
   * Recreate an array of indices with a new index and an index sequence.
   * @tparam indexSequence Index sequence (automatically deduced).
   * @param index New index to add to the array of indices.
   * @param indexSequenceArg Index sequence of the indices from 0 to `depth` to
   * access `mIndices`.
   * @return Extended array of indices.
   */
  template <std::size_t... indexSequence>
  KOKKOS_FUNCTION constexpr Kokkos::Array<std::size_t, depth + 1> extendIndices(
      [[maybe_unused]] std::index_sequence<indexSequence...> indexSequenceArg,
      std::size_t const index) const {
    return {{mIndices[indexSequence]..., index}};
  }

  /**
   * Recreate an array of indices with new indices and an index sequence.
   * @tparam indexSequence Index sequence (automatically deduced).
   * @tparam IndicesType Type of the indices.
   * @param indexSequenceArg Index sequence of the indices from 0 to `depth` to
   * access `mIndices`.
   * @param indices Pack of indices.
   * @return Extended array of indices.
   */
  template <std::size_t... indexSequence, typename... IndicesType>
  KOKKOS_FUNCTION constexpr Kokkos::Array<std::size_t, getRankSource()>
  extendIndices(
      [[maybe_unused]] std::index_sequence<indexSequence...> indexSequenceArg,
      IndicesType const... indices) const {
    return {{mIndices[indexSequence]..., indices...}};
  }

  /**
   * Get the scalar value of the wrapped view from a array of indices.
   * @param indices Array of indices above the sub-wrapper.
   * @return Scalar value of the view.
   */
  template <std::size_t depthNext>
  KOKKOS_FUNCTION constexpr auto &
  getValue(Kokkos::Array<std::size_t, depthNext> const &indices) const {
    return getValue(indices, std::make_index_sequence<depthNext>());
  }

  /**
   * Get the scalar value of the wrapped view from a array of indices and an
   * index sequence.
   * @tparam indexSequence Index sequence (automatically deduced).
   * @param indices Array of indices above the sub-wrapper.
   * @param indexSequenceArg Index sequence of the indices from 0 to `depth` to
   * access `indices`.
   * @return Scalar value of the view.
   */
  template <std::size_t depthNext, std::size_t... indexSequence>
  KOKKOS_FUNCTION constexpr auto &
  getValue(Kokkos::Array<std::size_t, depthNext> const &indices,
           [[maybe_unused]] std::index_sequence<indexSequence...>
               indexSequenceArg) const {
    return mData(indices[indexSequence]...);
  }
};

} // namespace brak

#endif // ifndef __BRAK_WRAPPER_ARRAY_HPP__
