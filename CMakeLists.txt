cmake_minimum_required(VERSION 3.15)
project(guanaco VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GUANACO_BUILD_EXAMPLES "Build example programs" ON)
option(GUANACO_BUILD_TESTS "Build tests" OFF)
option(GUANACO_USE_IO_URING "Use io_uring for async I/O (Linux only)" ON)
option(GUANACO_PATCH_LLAMA_CPP "Patch llama.cpp for Guanaco integration" ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
endif()

find_package(Threads REQUIRED)

set(LLAMA_CPP_BUILD_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp/build")

# Find llama.cpp libraries
find_library(GGML_LIBRARY NAMES ggml ggml-base
    PATHS ${LLAMA_CPP_BUILD_DIR}/bin ${LLAMA_CPP_BUILD_DIR}
    NO_DEFAULT_PATH
)

find_library(LLAMA_LIBRARY NAMES llama
    PATHS ${LLAMA_CPP_BUILD_DIR}/bin ${LLAMA_CPP_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(GUANACO_USE_IO_URING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(IO_URING_LIBRARY NAMES uring)
    if(IO_URING_LIBRARY)
        add_compile_definitions(GUANACO_HAVE_IO_URING=1)
        message(STATUS "Found io_uring: ${IO_URING_LIBRARY}")
    else()
        message(WARNING "io_uring not found, falling back to thread pool")
    endif()
endif()

find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
    add_compile_definitions(GUANACO_HAVE_MADVISE=1)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/include
)

# Patch llama.cpp if enabled
if(GUANACO_PATCH_LLAMA_CPP)
    message(STATUS "Applying Guanaco patches to llama.cpp...")
    
    # Apply patches
    execute_process(
        COMMAND git apply --check ${CMAKE_SOURCE_DIR}/patches/llama-model.h.patch
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE PATCH_CHECK_H
    )
    if(PATCH_CHECK_H EQUAL 0)
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/llama-model.h.patch
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_APPLY_H
        )
        if(PATCH_APPLY_H EQUAL 0)
            message(STATUS "Applied llama-model.h patch")
        else()
            message(WARNING "Failed to apply llama-model.h patch")
        endif()
    else()
        message(STATUS "llama-model.h patch already applied or not applicable")
    endif()
    
    execute_process(
        COMMAND git apply --check ${CMAKE_SOURCE_DIR}/patches/llama-model.cpp.patch
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE PATCH_CHECK_CPP
    )
    if(PATCH_CHECK_CPP EQUAL 0)
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/llama-model.cpp.patch
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_APPLY_CPP
        )
        if(PATCH_APPLY_CPP EQUAL 0)
            message(STATUS "Applied llama-model.cpp patch")
        else()
            message(WARNING "Failed to apply llama-model.cpp patch")
        endif()
    else()
        message(STATUS "llama-model.cpp patch already applied or not applicable")
    endif()
    
    execute_process(
        COMMAND git apply --check ${CMAKE_SOURCE_DIR}/patches/llama-graph.cpp.patch
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE PATCH_CHECK_GRAPH
    )
    if(PATCH_CHECK_GRAPH EQUAL 0)
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/llama-graph.cpp.patch
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_APPLY_GRAPH
        )
        if(PATCH_APPLY_GRAPH EQUAL 0)
            message(STATUS "Applied llama-graph.cpp patch")
        else()
            message(WARNING "Failed to apply llama-graph.cpp patch")
        endif()
    else()
        message(STATUS "llama-graph.cpp patch already applied or not applicable")
    endif()
    
    execute_process(
        COMMAND git apply --check ${CMAKE_SOURCE_DIR}/patches/llama.h.patch
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE PATCH_CHECK_LLAMA_H
    )
    if(PATCH_CHECK_LLAMA_H EQUAL 0)
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/llama.h.patch
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_APPLY_LLAMA_H
        )
        if(PATCH_APPLY_LLAMA_H EQUAL 0)
            message(STATUS "Applied llama.h patch")
        else()
            message(WARNING "Failed to apply llama.h patch")
        endif()
    else()
        message(STATUS "llama.h patch already applied or not applicable")
    endif()
    
    execute_process(
        COMMAND git apply --check ${CMAKE_SOURCE_DIR}/patches/llama.cpp.patch
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE PATCH_CHECK_LLAMA_CPP
    )
    if(PATCH_CHECK_LLAMA_CPP EQUAL 0)
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/llama.cpp.patch
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_APPLY_LLAMA_CPP
        )
        if(PATCH_APPLY_LLAMA_CPP EQUAL 0)
            message(STATUS "Applied llama.cpp patch")
        else()
            message(WARNING "Failed to apply llama.cpp patch")
        endif()
    else()
        message(STATUS "llama.cpp patch already applied or not applicable")
    endif()
    
    # Rebuild llama.cpp with patches
    execute_process(
        COMMAND cmake --build ${LLAMA_CPP_BUILD_DIR} --config Release -j$(nproc)
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
        RESULT_VARIABLE LLAMA_BUILD_RESULT
    )
    if(LLAMA_BUILD_RESULT EQUAL 0)
        message(STATUS "Rebuilt llama.cpp with Guanaco patches")
    else()
        message(WARNING "Failed to rebuild llama.cpp with patches")
    endif()
    
    add_compile_definitions(GUANACO_ENABLED=1)
endif()

# Find patched llama.cpp libraries
find_library(GGML_LIBRARY NAMES ggml ggml-base
    PATHS ${LLAMA_CPP_BUILD_DIR}/bin ${LLAMA_CPP_BUILD_DIR}
    NO_DEFAULT_PATH
)

find_library(LLAMA_LIBRARY NAMES llama
    PATHS ${LLAMA_CPP_BUILD_DIR}/bin ${LLAMA_CPP_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(GUANACO_USE_IO_URING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(IO_URING_LIBRARY NAMES uring)
    if(IO_URING_LIBRARY)
        add_compile_definitions(GUANACO_HAVE_IO_URING=1)
        message(STATUS "Found io_uring: ${IO_URING_LIBRARY}")
    else()
        message(WARNING "io_uring not found, falling back to thread pool")
    endif()
endif()

find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
    add_compile_definitions(GUANACO_HAVE_MADVISE=1)
endif()

# Guanaco sources
add_library(guanaco
    src/guanaco_manifest.cpp
    src/guanaco_steppeloader.cpp
    src/guanaco_herdcache.cpp
    src/guanaco_router.cpp
    src/guanaco_executor.cpp
    src/guanaco_model_hook.cpp
)

target_include_directories(guanaco PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/include
)

target_link_libraries(guanaco PUBLIC
    Threads::Threads
    ${GGML_LIBRARY}
    ${LLAMA_LIBRARY}
    ${RT_LIBRARY}
)

if(IO_URING_LIBRARY)
    target_link_libraries(guanaco PUBLIC ${IO_URING_LIBRARY})
endif()

target_compile_definitions(guanaco PUBLIC
    GUANACO_API_EXPORT
    GUANACO_ENABLED
)

if(BUILD_SHARED_LIBS)
    set_target_properties(guanaco PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
endif()

if(GUANACO_BUILD_EXAMPLES)
    add_executable(guanaco_example examples/guanaco_example.cpp)
    target_link_libraries(guanaco_example guanaco)
endif()

if(GUANACO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS guanaco
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/guanaco/ DESTINATION include/guanaco)

set(CPACK_PACKAGE_NAME "guanaco")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
include(CPack)