add_executable(
    test-subview
    main.cpp
    test_subview.cpp
)

target_link_libraries(
    test-subview
    Brak::brak
    GTest::gtest
)

if(BRAK_ENABLE_GTEST_DISCOVER_TESTS)
    gtest_discover_tests(test-subview)
endif()
