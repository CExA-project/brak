add_library(
    brak
    INTERFACE
)

target_include_directories(
    brak
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    brak
    INTERFACE
        Kokkos::kokkos
)

install(
    DIRECTORY
        "${CMAKE_CURRENT_LIST_DIR}/brak"
    DESTINATION
        include
)

add_library(
    Brak::brak
    ALIAS
        brak
)
