cmake_minimum_required(VERSION 3.16)
project(curriculum_tokenizer VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

add_library(tokenizer_lib STATIC src/tokenizer.cpp)
target_include_directories(tokenizer_lib PUBLIC include)
target_link_libraries(tokenizer_lib PUBLIC Threads::Threads)

pybind11_add_module(_cpp_tokenizer src/bindings.cpp src/tokenizer.cpp)
target_include_directories(_cpp_tokenizer PUBLIC include)
target_link_libraries(_cpp_tokenizer PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(tokenizer_lib PRIVATE /O2 /W4)
    target_compile_options(_cpp_tokenizer PRIVATE /O2 /W4)
else()
    target_compile_options(tokenizer_lib PRIVATE -O3 -Wall -Wextra -Wpedantic)
    target_compile_options(_cpp_tokenizer PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(tokenizer_test tests/test_tokenizer.cpp src/tokenizer.cpp)
    target_include_directories(tokenizer_test PUBLIC include)
    target_link_libraries(tokenizer_test PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(tokenizer_test PRIVATE /W4)
    else()
        target_compile_options(tokenizer_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # The native test requires trained artifacts. Supply, for example:
    # -DTOKENIZER_TEST_ARTIFACT_DIR=C:/path/to/artifacts
    set(TOKENIZER_TEST_ARTIFACT_DIR "" CACHE PATH "Directory containing vocab.json and merges.txt")
    if(TOKENIZER_TEST_ARTIFACT_DIR)
        add_test(
            NAME tokenizer_roundtrip
            COMMAND tokenizer_test
                "${TOKENIZER_TEST_ARTIFACT_DIR}/vocab.json"
                "${TOKENIZER_TEST_ARTIFACT_DIR}/merges.txt"
        )
    endif()
endif()
