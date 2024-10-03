#ifndef __BRAK_COMPUTE_HPP__
#define __BRAK_COMPUTE_HPP__

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace brak {

template <typename View, std::size_t depth = 0> class BracketsWrapperCompute {

  View mData;
  Kokkos::Array<std::size_t, depth> mIndices;

public:
  BracketsWrapperCompute(View const data) : mData(data) {}

  BracketsWrapperCompute(View const data,
                         Kokkos::Array<std::size_t, depth> const &indices)
      : mData(data), mIndices(indices) {}

  /**
   * Get the rank of the wrapped view.
   * @return Rank of the wrapped view.
   */
  static std::size_t constexpr getRank() { return View::rank() - depth; }
  static std::size_t constexpr getRankSource() { return View::rank(); }

  decltype(auto) operator[](std::size_t const index) const {
    // recreate array of indices
    Kokkos::Array<std::size_t, depth + 1> indices;
    for (std::size_t i = 0; i < mIndices.size(); i++) {
      indices[i] = mIndices[i];
    }
    indices[depth] = index;

    if constexpr (getRank() > 1) {
      return BracketsWrapperCompute<View, depth + 1>(mData, indices);
    } else {
      return getValue(indices);
    }
  }

  typename View::value_type *operator*() { return mData.data(); }
  View getView() { return mData; }

private:
  auto &getValue(Kokkos::Array<std::size_t, depth + 1> const &i) const {
    static_assert(getRankSource() <= 8, "Rank too large");
    static_assert(getRankSource() > 0, "Rank too small");

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

#endif // ifndef __BRAK_COMPUTE_HPP__
