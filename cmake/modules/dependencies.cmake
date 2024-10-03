include(FetchContent)

find_package(Kokkos 4.3.1 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Treating Kokkos as an internal dependency")
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz
    )
    FetchContent_MakeAvailable(kokkos)
endif()

if(BRAK_ENABLE_TESTS)
    find_package(googletest 1.15.2 QUIET)
    if(NOT googletest_FOUND)
        message(STATUS "Treating Gtest as an internal dependency")
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
        )
        FetchContent_MakeAvailable(googletest)
        include(GoogleTest)
    endif()
endif()

if(BRAK_ENABLE_BENCHMARKS)
    find_package(benchmark 1.9.0 QUIET)
    if(NOT benchmark_FOUND)
        message(STATUS "Treating Google benchmark as an internal dependency")
        FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.9.0.tar.gz
        )
        FetchContent_MakeAvailable(googlebenchmark)

        # override configuration
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    endif()
endif()
