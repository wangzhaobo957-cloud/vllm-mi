# 本文件定义学习型 C++ 推理引擎的构建目标、工具程序和测试入口。
cmake_minimum_required(VERSION 3.20)

project(
    mini_infer
    VERSION 0.1.0
    DESCRIPTION "A minimal C++ inference engine for learning"
    LANGUAGES CXX
)

option(MINI_INFER_BUILD_TOOLS "Build command-line tools" ON)
option(MINI_INFER_BUILD_TESTS "Build tests" ON)

# 库改为纯头文件（header-only）实现，因此使用 INTERFACE 目标，无需编译单独的源文件。
add_library(mini_infer INTERFACE)

add_library(mini_infer::mini_infer ALIAS mini_infer)

target_include_directories(
    mini_infer
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_compile_features(mini_infer INTERFACE cxx_std_20)

# 由于库已是 header-only，编译警告转由真正编译代码的可执行目标承担。
function(mini_infer_enable_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    endif()
endfunction()

if(MINI_INFER_BUILD_TOOLS)
    add_executable(mini-infer-generate tools/generate.cpp)
    target_link_libraries(mini-infer-generate PRIVATE mini_infer::mini_infer)
    mini_infer_enable_warnings(mini-infer-generate)
endif()

if(MINI_INFER_BUILD_TESTS)
    include(CTest)
    add_executable(mini-infer-tests tests/test_main.cpp)
    target_link_libraries(mini-infer-tests PRIVATE mini_infer::mini_infer)
    mini_infer_enable_warnings(mini-infer-tests)
    add_test(NAME mini-infer-tests COMMAND mini-infer-tests)

    if(MINI_INFER_BUILD_TOOLS)
        add_test(
            NAME mini-infer-generate-smoke
            COMMAND mini-infer-generate a 5
        )
        set_tests_properties(
            mini-infer-generate-smoke
            PROPERTIES PASS_REGULAR_EXPRESSION "^bcdef"
        )
    endif()
endif()
