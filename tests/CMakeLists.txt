add_executable(
    test-wrapper-subview
    main.cpp
    test_wrapper_subview.cpp
)

target_link_libraries(
    test-wrapper-subview
    Brak::brak
    GTest::gtest
)

if(BRAK_ENABLE_GTEST_DISCOVER_TESTS)
    gtest_discover_tests(test-wrapper-subview)
endif()

add_executable(
    test-wrapper-array
    main.cpp
    test_wrapper_array.cpp
)

target_link_libraries(
    test-wrapper-array
    Brak::brak
    GTest::gtest
)

if(BRAK_ENABLE_GTEST_DISCOVER_TESTS)
    gtest_discover_tests(test-wrapper-array)
endif()
