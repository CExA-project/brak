add_executable(
    example-heat-equation-ref
    example_heat_equation_ref.cpp
)

add_executable(
    example-heat-equation-wrapper-array
    example_heat_equation_wrapper_array.cpp
)

target_link_libraries(
    example-heat-equation-wrapper-array
    Brak::brak
    Kokkos::kokkos
)

add_executable(
    example-heat-equation-wrapper-subview
    example_heat_equation_wrapper_subview.cpp
)

target_link_libraries(
    example-heat-equation-wrapper-subview
    Brak::brak
    Kokkos::kokkos
)
