# Brackets wrapper for Kokkos

Brackets wrapper for Kokkos (or "brak" for short) is a header-only library that proposes a wrapper class to access Kokkos views with a plain old data C array syntax, using brackets.

This library is especially useful if you want to start porting a code to Kokkos by updating the data structures first, while keeping the loops untouched.

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

If you don't have a GPU available when compiling with a GPU backend activated, you have to disable the CMake option `BRAK_ENABLE_GTEST_DISCOVER_TESTS`.

## Examples

You can build examples with the CMake option `BRAK_ENABLE_EXAMPLES`.
They should be run individually.

## Benchmarks

Benchmarks are built with the CMake option `BRAK_ENABLE_BENCHMARKS`.
They should be run individually.

## Documentation

The API documentation is handled by Doxygen (1.9.1 or newer) and is built with the CMake option `BRAK_ENABLE_DOCUMENTATION`.
The private API is not included by default and is added with the option `BRAK_ENABLE_DOCUMENTATION_DEVMODE`.
The documentation is built with the target `docs`.

## Use

The library allows to wrap a Kokkos view to use it like a plain old data C array.
If the number of pair of brackets is the same as the rank of the view, then the resulting object is a scalar:

```cpp
#include <cassert>
#include <Kokkos_Core.hpp>
#include "brak/wrapper_subview.hpp"
// or
#include "brak/wrapper_array.hpp"

void doSomething() {
  Kokkos::View<int ********, Kokkos::HostSpace> data{"data", 2, 2, 2, 2, 2, 2, 2, 2};
  brak::WrapperSubview dataWrapper{data};
  // or
  brak::WrapperArray dataWrapper{data};

  dataWrapper[0][0][0][0][0][0][0][0] = 10;
  assert(data(0, 0, 0, 0, 0, 0, 0, 0) == 10);
}
```

To achieve this, two implementations are proposed (they share the same API) in the next section.

Keep in mind however that not using the wrapped view up to it's scalar value results in a Brak object:

```cpp
  auto subDataWrapper = dataWrapper[0][0][0][0];
  static_assert(!std::is_same_v<decltype(subDataWrapper), int ****>);
  subDataWrapper[0][0][0][0] = 10;
```

It is possible to retrieve the current wrapped view with the `getView` method:

```cpp
  auto subData = dataWrapper[0][0][0][0].getView();
  static_assert(Kokkos::is_view<decltype(subData)>::value);
```

It is also possible to get the raw pointer of the current wrapped view with the defer operator, even if this may lead to unpredictable behaviors:

```cpp
  auto subPointer = *(dataWrapper[0][0][0][0]);
  static_assert(std::is_pointer_v<decltype(*subPointer)>);
```

### Subview wrapper approach

With this approach, the class `brak::WrapperSubview` wraps a view, and each call to the brackets operator gives a new instance of the class wrapping a subview of a rank lowered by one.
The subview is unmanaged, in order to disable reference counting and increase performance.

This approach is not efficient in terms of performance at compile time and at runtime, due to the remaining reference counting that could not be disabled.

This implementation can be still interesting as if you don't go up to the scalar value, the intermediate object returned by the brackets operator is still useable somehow (it's a subview, after all).

### Array wrapper approach

With this different approach, the class `brak::WrapperArray` wraps a view, and each call to the brackets operator gives a sub-wrapper that also stores an array of the requested indices.
The subsequent wrapper contains an unmanaged version of the initial view, in order to disable reference counting and increase performance.

This approach has performance that are on par with Kokkos views.

## Performance

Benchmarks done using an Intel Core i7-13800H and a NVIDIA A500 GPU, for a release build (unless specified in the details), all times in seconds.
Performance ratios are expressed with standard deviation, between parentheses.

| Implementation  | Build Serial          | Access Serial           | Nested-for Serial       | Parallel-for Serial   | Parallel-for OpenMP   | Parallel-for Cuda      |
|-----------------|-----------------------|-------------------------|-------------------------|-----------------------|-----------------------|------------------------|
| Wrapper subview | 984 × 10<sup>-3</sup> | 9.89 × 10<sup>-9</sup>  | 2337e × 10<sup>-3</sup> | 726 × 10<sup>-3</sup> | 353 × 10<sup>-3</sup> | 97.3 × 10<sup>-3</sup> |
| Wrapper array   | 805 × 10<sup>-3</sup> | 0.392 × 10<sup>-9</sup> | 36.1e × 10<sup>-3</sup> | 451 × 10<sup>-3</sup> | 309 × 10<sup>-3</sup> | 89.1 × 10<sup>-3</sup> |
| Reference view  | 768 × 10<sup>-3</sup> | 1.14 × 10<sup>-9</sup>  | 58.5e × 10<sup>-3</sup> | 443 × 10<sup>-3</sup> | 332 × 10<sup>-3</sup> | 87.7 × 10<sup>-3</sup> |

Benchmarks are detailed in the next sections.

In terms of compilation time, building a code using a subview wrapper is 1.28 (3 %) times slower than a code using a reference view, and a code using an array wrapper is 1.05 (3 %) times slower.

When accessing a single element, a subview wrapper is 8.7 (1 %) times slower than a view, and an array wrapper is 2.9 (1 %) times faster.
The later is due to reference counting being disabled for wrappers.
Though using it, the subview wrapper does not benefit of it much, but the same order of magnitude of execution time can be obtained if the initial view is already unmanaged.

For a more realistic use of the arrays, a subview wrapper is 40 (1 %) times slower than a view, and an array wrapper is 1.6 (2 %) times faster.
Frequent accesses to data is less well handled by the subview wrapper.
Using an already unmanaged view brings the performance of the subview wrapper similar to the use of Kokkos views.

For a heavy access of elements, a subview wrapper is 1.64 (2 %) times slower than a view for CPU serial execution.
It is 1.06 (7 %) times slower, respectively 1.11 (0.4 %) times slower, for CPU parallel execution, respectively GPU execution, meaning that parallel execution tends to lower the difference.
An array wrapper is 1.02 (2 %) times slower, respectively 1.8 (7 %) faster and 1.02 (0.3 %) times slower, for CPU serial execution, respectively CPU parallel execution and GPU execution, which shows that this implementation has a limited impact on performance.

### Build benchmark details

This [compile benchmark](./compile_benchmarks) consists in compiling in debug mode a function that creates a view of rank 8 of dimension 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 (256 elements) containing 4 bits integers (1.024 kB) and that accesses and sets its element 1, 1, 1, 1, 1, 1, 1, 1 to 10.

### Access benchmark details

This [benchmark](./benchmarks/benchmark_access.cpp) uses a view of rank 8 of dimension 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 (256 elements) containing 4 bits integers (1.024 kB). 
It consists in measuring the time to access and set the element 1, 1, 1, 1, 1, 1, 1, 1 to 10.

### Nested-for benchmark details

This [benchmark](./benchmarks/benchmark_nested_for.cpp) uses two views of rank 3 of dimension 30 × 30 × 30 (27 × 10<sup>3</sup> elements) containing 4 bits integers each (216 kB).
It consists in measuring the time to update one view from the other with a stencil, then to swap the two views.
This benchmark loosely relates to the heat equation.
Loops are performed using traditional nested `for` loops.

### Parallel-for benchmark details

This [benchmark](./benchmarks/benchmark_parallel_for.cpp) uses a view of rank 6 of dimension 30 × 30 × 30 × 30 × 30 × 30 (729 × 10<sup>6</sup> elements) containing 4 bits integers (2.916 GB).
It consists in measuring the time for a Kokkos `parallel_for` loop to fill all the elements of the view with the sum of their coordinates.
The time spent in launching the kernel is counterbalanced by the large size of the view.
