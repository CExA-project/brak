cmake_minimum_required(VERSION 3.22)

project(
    BracketsAdapterKokkos
    LANGUAGES CXX
    VERSION 0.0.0
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 17)

include(options)
include(dependencies)

add_subdirectory(include)

if(BRAK_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# if(BRAK_ENABLE_EXAMPLES)
#     add_subdirectory(examples)
# endif()
