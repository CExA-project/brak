add_library(
    compile-benchmark-view
    OBJECT
    compile_benchmark_view.cpp
)

target_link_libraries(
    compile-benchmark-view
    Brak::brak
)

add_library(
    compile-benchmark-subview
    OBJECT
    compile_benchmark_subview.cpp
)

target_link_libraries(
    compile-benchmark-subview
    Brak::brak
)

add_library(
    compile-benchmark-array
    OBJECT
    compile_benchmark_array.cpp
)

target_link_libraries(
    compile-benchmark-array
    Brak::brak
)
