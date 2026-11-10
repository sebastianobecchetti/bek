cmake_minimum_required(VERSION 3.14)
project(gs_encodingc VERSION 3.67 LANGUAGES CXX)

configure_file(
    CMake/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    @ONLY
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimization flags
option(BEK_FAST_MATH "Build with -ffast-math (default OFF; OFF = strict IEEE)" OFF)
add_compile_options(-O3 -march=native)
if(BEK_FAST_MATH)
    add_compile_options(-ffast-math)
else()
    add_compile_options(-fno-fast-math -ffp-contract=off)
endif()

find_package(Threads REQUIRED)

# zstd v1.5.5 still ships `cmake_minimum_required(VERSION 2.8)` in its own
# build/cmake/CMakeLists.txt, and CMake 4 refuses to configure that at all --
# so a plain `cmake -S . -B build` fails on any recent toolchain unless the
# sub-project is given a policy floor. Set it for the fetched dependencies.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# Zstd (fetch via CMake)
include(FetchContent)
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.5
    SOURCE_SUBDIR build/cmake
)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(zstd)

# Nlohmann JSON (fetch via CMake)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# WebP (install with OS package managers)
find_path(WEBP_INCLUDE_DIR
    NAMES webp/encode.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include
)
find_library(WEBP_LIBRARY
    NAMES webp
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib
)

if (WEBP_INCLUDE_DIR AND WEBP_LIBRARY)
    message(STATUS "Found WebP: ${WEBP_LIBRARY}")
else()
    message(FATAL_ERROR "WebP not found. Please install libwebp.")
endif()

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${zstd_SOURCE_DIR}/lib
    ${WEBP_INCLUDE_DIR}
)

set(SRCS
    src/ply_reader.cpp
    src/quantization.cpp
    src/sh_vq.cpp
    src/tinyply.cpp
    src/ply_writer.cpp
    src/range_coder.cpp
)

add_executable(ply_to_bek apps/ply_to_bek.cpp ${SRCS})
target_link_libraries(ply_to_bek
    PRIVATE
    Threads::Threads
    libzstd_static
    nlohmann_json::nlohmann_json
    ${WEBP_LIBRARY}
)

add_executable(bek_to_ply apps/bek_to_ply.cpp ${SRCS})
target_link_libraries(bek_to_ply
    PRIVATE
    Threads::Threads
    libzstd_static
    nlohmann_json::nlohmann_json
    ${WEBP_LIBRARY}
)

# Tests
option(BUILD_TESTING "Build and register codec tests" OFF)
include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    function(bek_roundtrip_test name fixture)
        add_test(
            NAME ${name}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.py
                --encoder $<TARGET_FILE:ply_to_bek>
                --decoder $<TARGET_FILE:bek_to_ply>
                --fixture ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/${fixture}.ply
                ${ARGN}
        )
        set_tests_properties(${name} PROPERTIES TIMEOUT 150)
    endfunction()

    bek_roundtrip_test(roundtrip_single single)
    bek_roundtrip_test(roundtrip_coincident coincident)
    bek_roundtrip_test(roundtrip_cube cube)
    bek_roundtrip_test(roundtrip_cloud cloud)
    bek_roundtrip_test(roundtrip_binary cloud --binary)
    bek_roundtrip_test(roundtrip_cloud_binary_normals cloud_binary)
    bek_roundtrip_test(roundtrip_precision cloud --pos-bits 12 --scale-bits 4)
    bek_roundtrip_test(roundtrip_joint cloud --variant joint)
    bek_roundtrip_test(roundtrip_rotation_split cloud --variant split)
    bek_roundtrip_test(roundtrip_legacy cloud --variant legacy)
endif()

# Installation
include(GNUInstallDirs)

install(
    TARGETS ply_to_bek bek_to_ply
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)

install(
    FILES README.md LICENSE CITATION.cff
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT Documentation
)

install(
    FILES docs/compression.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/docs
    COMPONENT Documentation
)

install(
    DIRECTORY assets/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/assets
    COMPONENT Documentation
    FILES_MATCHING
    PATTERN "*.png"
    PATTERN "*.jpg"
)
