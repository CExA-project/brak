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

## Documentation

The API documentation is handled by Doxygen (1.9.1 or newer) and is built with the CMake option `BRAK_ENABLE_DOCUMENTATION`.
The private API is not included by default and is added with the option `BRAK_ENABLE_DOCUMENTATION_DEVMODE`.
The documentation is built with the target `docs`.

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
The subview is unmanaged, in order to disable reference counting and increase performance.

This approach is less inefficient in terms of performance at compile time and at runtime, due to the extra overhead.

### Array wrapper approach

With this different approach, the class `brak::WrapperArray` wraps a view, and each call to the brackets operator gives a sub-wrapper that also stores an array of the requested indices.
The subsequent wrapper contains an unmanaged version of the initial view, in order to disable reference counting and increase performance.

This approach is more inefficient than the subview wrapper approach.

## Performance

Benchmarks done using an Intel Core i7-13800H and a NVIDIA A500 GPU, all times in seconds.

| Implementation  | Build Serial          | Access Serial           | Parallel-for Serial   | Parallel-for OpenMP   | Parallel-for Cuda      |
|-----------------|-----------------------|-------------------------|-----------------------|-----------------------|------------------------|
| Wrapper subview | 968 × 10<sup>-3</sup> | 10.4 × 10<sup>-9</sup>  | 711 × 10<sup>-3</sup> | 252 × 10<sup>-3</sup> | 97.7 × 10<sup>-3</sup> |
| Wrapper array   | 800 × 10<sup>-3</sup> | 0.406 × 10<sup>-9</sup> | 446 × 10<sup>-3</sup> | 230 × 10<sup>-3</sup> | 92.7 × 10<sup>-3</sup> |
| Reference view  | 771 × 10<sup>-3</sup> | 1.20 × 10<sup>-9</sup>  | 438 × 10<sup>-3</sup> | 245 × 10<sup>-3</sup> | 87.5 × 10<sup>-3</sup> |

Benchmarks are detailed in the next sections.

In terms of compilation time, the subview wrapper approach takes 25 % more time than a reference view, and the array approach 4 %.

When accessing a single element, a subview wrapper is 8.7 times slower than a view, and an array wrapper is 3 times faster.
The later is due to reference counting being disabled for wrappers.
Though using it, the subview wrapper does not benefit of it much, but the same order of magnitude of execution time can be obtained if the initial view is already unmanaged.

For a more realistic access of elements, a subview wrapper is 62 % times slower than a view for CPU serial execution.
It is 3 % slower, respectively 12 % slower, for CPU parallel execution, respectively GPU execution, meaning that parallel execution tends to lower the difference.
An array wrapper is 2 % slower, respectively 6 % faster and 6 % slower, for CPU serial execution, respectively CPU parallel execution and GPU execution, which shows that this implementation has a limited impact on performance.

### Build benchmark details

This [compile benchmark](./compile_benchmarks) consists in compiling in debug mode a function that creates a view of rank 8 of dimension 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 (256 elements) containing 4 bits integers (1.024 kB) and that accesses and sets its element 1, 1, 1, 1, 1, 1, 1, 1 to 10.

### Access benchmark details

This [benchmark](./benchmarks/benchmark_access.cpp) uses a view of rank 8 of dimension 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 (256 elements) containing 4 bits integers (1.024 kB). 
It consists in measuring the time to access and set the element 1, 1, 1, 1, 1, 1, 1, 1 to 10.

### Parallel-for benchmark details

This [benchmark](./benchmarks/benchmark_parallel_for.cpp) uses a view of rank 6 of dimension 30 × 30 × 30 × 30 × 30 × 30 (729 × 10<sup>6</sup> elements) containing 4 bits integers (2.916 GB).
It consists in measuring the time for a Kokkos `parallel_for` loop to fill all the elements of the view with the sum of their coordinates.
The time spent in launching the kernel is counterbalanced by the large size of the view.
