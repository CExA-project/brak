include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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
        "${CMAKE_INSTALL_LIBDIR}/cmake/Brak"
)

configure_package_config_file( 
    "${PROJECT_SOURCE_DIR}/cmake/modules/Config.cmake.in" 
    "${CMAKE_CURRENT_BINARY_DIR}/BrakConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Brak"
    PATH_VARS
        CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/BrakConfigVersion.cmake"
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/BrakConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/BrakConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/Brak"
)
