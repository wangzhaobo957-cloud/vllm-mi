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

add_library(
    mini_infer
    src/kv_cache.cpp
    src/model_config.cpp
    src/reference_cpu_backend.cpp
    src/sampler.cpp
    src/tensor.cpp
)

add_library(mini_infer::mini_infer ALIAS mini_infer)

target_include_directories(
    mini_infer
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_compile_features(mini_infer PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mini_infer PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(mini_infer PRIVATE /W4)
endif()

if(MINI_INFER_BUILD_TOOLS)
    add_executable(mini-infer-generate tools/generate.cpp)
    target_link_libraries(mini-infer-generate PRIVATE mini_infer::mini_infer)
endif()

if(MINI_INFER_BUILD_TESTS)
    include(CTest)
    add_executable(mini-infer-tests tests/test_main.cpp)
    target_link_libraries(mini-infer-tests PRIVATE mini_infer::mini_infer)
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
