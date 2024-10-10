cmake_minimum_required(VERSION 3.22)

project(
    Brak
    LANGUAGES CXX
    VERSION 0.0.0
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 17)

include(options)
include(dependencies)

add_subdirectory(include)

include(install)

if(BRAK_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BRAK_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BRAK_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BRAK_ENABLE_COMPILE_BENCHMARKS)
    add_subdirectory(compile_benchmarks)
endif()

if(BRAK_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif()
