add_executable(
    benchmark-access
    benchmark_access.cpp
    main.cpp
)

target_link_libraries(
    benchmark-access
    benchmark::benchmark
    Brak::brak
)

add_executable(
    benchmark-nested-for
    benchmark_nested_for.cpp
    main.cpp
)

target_link_libraries(
    benchmark-nested-for
    benchmark::benchmark
    Brak::brak
)

add_executable(
    benchmark-parallel-for
    benchmark_parallel_for.cpp
    main.cpp
)

target_link_libraries(
    benchmark-parallel-for
    benchmark::benchmark
    Brak::brak
)
