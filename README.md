# Brackets wrapper for Kokkos

Brackets wrapper for Kokkos (or "brak" for short) is a header-only library that proposes a wrapper class to access a Kokkos view with a plain old C array syntax, using brackets.

## Install

### With CMake

The best way is to use CMake.

#### As a subdirectory

Get the library in your project:

```sh
git clone https://github.com/cexa-project/brak.git path/to/brak
```

In your main CMake file:

```cmake
add_subdirectory(path/to/brak)

target_link_libraries(
    my-lib
    PRIVATE
        Brak::brak
)
```

### With FetchContent

In your main CMake file:

<!-- URL https://github.com/CExA-project/brak/archive/refs/tags/0.1.0.tar.gz -->

```cmake
include(FetchContent)
FetchContent_Declare(
    brak
    GIT_REPOSITORY https://github.com/CExA-project/brak.git
    GIT_TAG master
)
FetchContent_MakeAvailable(brak)

target_link_libraries(
    my-lib
    PRIVATE
        Brak::brak
)
```

#### As a locally available dependency

Get, then install the project:

```sh
git clone https://github.com/cexa-project/brak.git
cd brak
cmake -B build -DCMAKE_INSTALL_PREFIX=path/to/install -DCMAKE_BUILD_TYPE=Release # other Kokkos options here if needed
cmake --install build
```

In your main CMake file:

```cmake
find_package(Brak REQUIRED)

target_link_libraries(
    my-lib
    PRIVATE
        Brak::brak
)
```

### Copy files

Alternatively, you can also copy `include/brak` in your project and start using it.

## Tests

You can build tests with the CMake option `BRAK_ENABLE_TESTS`, and run them with `ctest`.

If you don't have a GPU available when compiling, you have to disable the CMake option `BRAK_ENABLE_GTEST_DISCOVER_TESTS`.

<!-- ## Examples -->
<!--  -->
<!-- You can build examples with the CMake option `BRAK_ENABLE_EXAMPLES`. -->
<!-- They should be run individually. -->

## Benchmarks

Benchmarks are built with the CMake option `BRAK_ENABLE_BENCHMARKS`.
They should be run individually.

## Use

The library allows to wrap a Kokkos view to use it like a plain old C array.
If the number of pair of brackets is the same as the rank of the view, then the resulting object is a scalar:

```cpp
#include <Kokkos_Core.hpp>
#include "brak/wrapper_subview.hpp"
// or
#include "brak/wrapper_array.hpp"

void doSomething() {
  Kokkos::View<int ********> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperSubview dataWrapper{data};
  // or
  brak::WrapperArray dataWrapper{data};

  dataWrapper[0][0][0][0][0][0][0] = 10;
}
```

To achieve this, two implementations are proposed (they share the same API).

### Subview wrapper approach

With this approach, the class `brak::WrapperSubview` wraps a view, and each call to the brackets operator gives a new instance of the class wrapping a subview.

This approach is very inefficient in terms of performance at compile time and at runtime, due to the extra overhead.
For a view of rank 8 on CPU, the build is 22 % slower than using the parenthesis operator directly, and the execution is 180 times slower.

### Array wrapper approach

With this different approach, the class `brak::WrapperArray` wraps a view, and each call to the brackets operator gives a sub-wrapper that also stores an array of the requested indices.

This approach is a bit less inefficient than the subview wrapper approach.
For a view of rank 8 on CPU, the build is 3 % slower than using the parenthesis operator directly, and the execution is 120 times slower.
