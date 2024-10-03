add_executable(
    benchmark-brak
    benchmark.cpp
    main.cpp
)

target_link_libraries(
    benchmark-brak
    benchmark::benchmark
    Brak::brak
)
