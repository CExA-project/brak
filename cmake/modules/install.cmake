include(GNUInstallDirs)

install(
    TARGETS
        brak
    EXPORT
        BrakTargets
    ARCHIVE DESTINATION
        "${CMAKE_INSTALL_LIBDIR}"
)

install(
    EXPORT
        BrakTargets
    NAMESPACE Brak::
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/Brak"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/BrakConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/BrakConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/Brak"
)
