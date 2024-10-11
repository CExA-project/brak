#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#ifndef WRAPPER_CLASS
#error "Undefined wrapper class"
#endif

#ifndef WRAPPER_NAME
#error "Undefined wrapper name"
#endif

#define GET_TEST_NAME_NX(A, B) A##B
#define GET_TEST_NAME_INTEGRATION_NX(A, B, C) A##B##C
#define GET_TEST_NAME(NAME) GET_TEST_NAME_NX(test_, NAME)
#define GET_TEST_NAME_INTEGRATION(NAME)                                        \
  GET_TEST_NAME_INTEGRATION_NX(test_, NAME, _integration)

TEST(GET_TEST_NAME(WRAPPER_NAME), test_create) {
  Kokkos::View<int **> data{"data", 10, 10};
  WRAPPER_CLASS dataWrapper{data};

  static_assert(decltype(dataWrapper)::getRank() == 2);
}

TEST(GET_TEST_NAME(WRAPPER_NAME), test_access) {
  Kokkos::View<int ********, Kokkos::HostSpace> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  WRAPPER_CLASS dataWrapper8D{data};
  static_assert(decltype(dataWrapper8D)::getRank() == 8);
  static_assert(!std::is_same_v<decltype(dataWrapper8D), int ********>);

  auto dataWrapper7D = dataWrapper8D[1];
  static_assert(decltype(dataWrapper7D)::getRank() == 7);
  static_assert(!std::is_same_v<decltype(dataWrapper7D), int *******>);

  auto dataWrapper6D = dataWrapper7D[1];
  static_assert(decltype(dataWrapper6D)::getRank() == 6);
  static_assert(!std::is_same_v<decltype(dataWrapper6D), int ******>);

  auto dataWrapper5D = dataWrapper6D[1];
  static_assert(decltype(dataWrapper5D)::getRank() == 5);
  static_assert(!std::is_same_v<decltype(dataWrapper5D), int *****>);

  auto dataWrapper4D = dataWrapper5D[1];
  static_assert(decltype(dataWrapper4D)::getRank() == 4);
  static_assert(!std::is_same_v<decltype(dataWrapper4D), int ****>);

  auto dataWrapper3D = dataWrapper4D[1];
  static_assert(decltype(dataWrapper3D)::getRank() == 3);
  static_assert(!std::is_same_v<decltype(dataWrapper3D), int ***>);

  auto dataWrapper2D = dataWrapper3D[1];
  static_assert(decltype(dataWrapper2D)::getRank() == 2);
  static_assert(!std::is_same_v<decltype(dataWrapper2D), int **>);

  auto dataWrapper1D = dataWrapper2D[1];
  static_assert(decltype(dataWrapper1D)::getRank() == 1);
  static_assert(!std::is_same_v<decltype(dataWrapper1D), int *>);

  auto dataValue = dataWrapper1D[1];
  static_assert(std::is_same_v<decltype(dataValue), int>);
}

TEST(GET_TEST_NAME(WRAPPER_NAME), test_access_direct) {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};

  WRAPPER_CLASS dataWrapper8D{data};
  static_assert(decltype(dataWrapper8D)::getRank() == 8);
  static_assert(decltype(dataWrapper8D[1])::getRank() == 7);
  static_assert(decltype(dataWrapper8D[1][1])::getRank() == 6);
  static_assert(decltype(dataWrapper8D[1][1][1])::getRank() == 5);
  static_assert(decltype(dataWrapper8D[1][1][1][1])::getRank() == 4);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1])::getRank() == 3);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1][1])::getRank() == 2);
  static_assert(decltype(dataWrapper8D[1][1][1][1][1][1][1])::getRank() == 1);
}

TEST(GET_TEST_NAME(WRAPPER_NAME), test_write_1d) {
  Kokkos::View<int *, Kokkos::HostSpace> data{"data", 10};
  WRAPPER_CLASS dataWrapper{data};

  ASSERT_EQ(data(1), 0);

  dataWrapper[1] = 1;

  ASSERT_EQ(data(1), 1);
}

TEST(GET_TEST_NAME(WRAPPER_NAME), test_defer) {
  Kokkos::View<int **, Kokkos::HostSpace> data{"data", 10, 10};
  WRAPPER_CLASS dataWrapper{data};

  auto dataPointer = *dataWrapper;

  static_assert(std::is_pointer_v<decltype(dataPointer)>);

  ASSERT_EQ(data.data(), dataPointer);
  ASSERT_EQ(data.data(), *(dataWrapper[0]));
}

TEST(GET_TEST_NAME(WRAPPER_NAME), test_get_view) {
  Kokkos::View<int **> data{"data", 10, 10};
  WRAPPER_CLASS dataWrapper{data};
  auto dataView = dataWrapper.getView();

  static_assert(Kokkos::is_view<decltype(dataView)>::value);
  static_assert(std::is_same_v<decltype(data), decltype(dataView)>);
  ASSERT_EQ(data.data(), dataView.data());
}

TEST(GET_TEST_NAME_INTEGRATION(WRAPPER_NAME), test_access_nested_for) {
  Kokkos::View<int ******, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2};
  WRAPPER_CLASS dataWrapper{data};

  for (std::size_t i = 0; i < 2; i++)
    for (std::size_t j = 0; j < 2; j++)
      for (std::size_t k = 0; k < 2; k++)
        for (std::size_t l = 0; l < 2; l++)
          for (std::size_t m = 0; m < 2; m++)
            for (std::size_t n = 0; n < 2; n++) {
              dataWrapper[i][j][k][l][m][n] =
                  i + j * 2 + k * 3 + l * 4 + m * 5 + n * 6;
            }

  ASSERT_EQ(data(0, 1, 0, 1, 0, 1), 12);
}

template <typename Wrapper> struct TestFunctor {
  Wrapper mWrapper;

  TestFunctor(Wrapper const wrapper) : mWrapper(wrapper) {}

  KOKKOS_FUNCTION
  void operator()(std::size_t const i, std::size_t const j, std::size_t const k,
                  std::size_t const l, std::size_t const m,
                  std::size_t const n) const {
    mWrapper[i][j][k][l][m][n] = i + j * 2 + k * 3 + l * 4 + m * 5 + n * 6;
  }
};

TEST(GET_TEST_NAME_INTEGRATION(WRAPPER_NAME), test_access_parallel_for) {
  Kokkos::View<int ******, Kokkos::DefaultHostExecutionSpace::memory_space>
      data{"data", 2, 2, 2, 2, 2, 2};
  WRAPPER_CLASS dataWrapper{data};

  Kokkos::parallel_for(
      "test_access_parallel_for",
      Kokkos::MDRangePolicy(Kokkos::DefaultHostExecutionSpace(),
                            {0, 0, 0, 0, 0, 0}, {2, 2, 2, 2, 2, 2}),
      TestFunctor(dataWrapper));

  ASSERT_EQ(data(0, 1, 0, 1, 0, 1), 12);
}

#ifndef DISABLE_TEST_DEVICE

TEST(GET_TEST_NAME_INTEGRATION(WRAPPER_NAME), test_access_parallel_for_device) {
  Kokkos::View<int ******> data{"data", 2, 2, 2, 2, 2, 2};
  auto dataMirror = Kokkos::create_mirror_view(data);
  WRAPPER_CLASS dataWrapper{data};

  Kokkos::parallel_for(
      "test_access_parallel_for",
      Kokkos::MDRangePolicy({0, 0, 0, 0, 0, 0}, {2, 2, 2, 2, 2, 2}),
      TestFunctor(dataWrapper));

  Kokkos::deep_copy(dataMirror, data);

  ASSERT_EQ(dataMirror(0, 1, 0, 1, 0, 1), 12);
}

#endif // ifndef DISABLE_TEST_DEVICE
