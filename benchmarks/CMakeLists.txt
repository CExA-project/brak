add_executable(
    benchmark-subview
    benchmark_subview.cpp
    main.cpp
)

target_link_libraries(
    benchmark-subview
    benchmark::benchmark
    Brak::brak
)
