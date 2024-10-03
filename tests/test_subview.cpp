#
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "brak/subview.hpp"

TEST(test_brackets_wrapper_subview, test_create) {
  Kokkos::View<int **> data{"data", 10, 10};
  brak::BracketsWrapperSubview dataWrapper{data};

  static_assert(decltype(dataWrapper)::getRank() == 2);
}

TEST(test_brackets_wrapper_subview, test_access) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  brak::BracketsWrapperSubview dataWrapper8D{data};
  static_assert(decltype(dataWrapper8D)::getRank() == 8);

  auto dataWrapper7D = dataWrapper8D[1];
  static_assert(decltype(dataWrapper7D)::getRank() == 7);

  auto dataWrapper6D = dataWrapper7D[1];
  static_assert(decltype(dataWrapper6D)::getRank() == 6);

  auto dataWrapper5D = dataWrapper6D[1];
  static_assert(decltype(dataWrapper5D)::getRank() == 5);

  auto dataWrapper4D = dataWrapper5D[1];
  static_assert(decltype(dataWrapper4D)::getRank() == 4);

  auto dataWrapper3D = dataWrapper4D[1];
  static_assert(decltype(dataWrapper3D)::getRank() == 3);

  auto dataWrapper2D = dataWrapper3D[1];
  static_assert(decltype(dataWrapper2D)::getRank() == 2);

  auto dataWrapper1D = dataWrapper2D[1];
  static_assert(decltype(dataWrapper1D)::getRank() == 1);
}

TEST(test_brackets_wrapper_subview, test_access_direct) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  brak::BracketsWrapperSubview dataWrapper8D{data};
  static_assert(decltype(dataWrapper8D)::getRank() == 8);
  static_assert(decltype(dataWrapper8D[1])::getRank() == 7);
  static_assert(decltype(dataWrapper8D[1][1])::getRank() == 6);
  static_assert(decltype(dataWrapper8D[1][1][1])::getRank() == 5);
  static_assert(decltype(dataWrapper8D[1][1][1][1])::getRank() == 4);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1])::getRank() == 3);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1][1])::getRank() == 2);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1][1][1])::getRank() == 1);
}

TEST(test_brackets_wrapper_subview, test_write_1d) {
  Kokkos::View<int *, Kokkos::HostSpace> data{"data", 10};
  brak::BracketsWrapperSubview dataWrapper{data};

  ASSERT_EQ(data(1), 0);

  dataWrapper[1] = 1;

  ASSERT_EQ(data(1), 1);
}

TEST(test_brackets_wrapper_subview, test_defer) {
  Kokkos::View<int *, Kokkos::HostSpace> data{"data", 10};
  brak::BracketsWrapperSubview dataWrapper{data};

  ASSERT_EQ(data.data(), *dataWrapper);
}

TEST(test_brackets_wrapper_subview, test_get_view) {
  Kokkos::View<int **> data{"data", 10, 10};
  brak::BracketsWrapperSubview dataWrapper{data};
  auto dataView = dataWrapper.getView();

  static_assert(std::is_same_v<decltype(data), decltype(dataView)>);
  ASSERT_EQ(data.data(), dataView.data());
}

TEST(test_brackets_wrapper_subview_integration, test_access) {
  Kokkos::View<int ******, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2};
  brak::BracketsWrapperSubview dataWrapper{data};

  for (std::size_t i = 0; i < 2; i++)
    for (std::size_t j = 0; j < 2; j++)
      for (std::size_t k = 0; k < 2; k++)
        for (std::size_t l = 0; l < 2; l++)
          for (std::size_t m = 0; m < 2; m++)
            for (std::size_t n = 0; n < 2; n++) {
              dataWrapper[i][j][k][l][m][n] = i + j + k + l + m + n;
            }

  ASSERT_EQ(data(1, 1, 1, 1, 1, 1), 6);
}
