cmake_minimum_required(VERSION 3.15)
project(guanaco VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GUANACO_BUILD_EXAMPLES "Build example programs" ON)
option(GUANACO_BUILD_TESTS "Build tests" OFF)
option(GUANACO_USE_IO_URING "Use io_uring for async I/O (Linux only)" ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
endif()

find_package(Threads REQUIRED)

set(LLAMA_CPP_BUILD_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp/build")

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

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp/include
)

add_library(guanaco
    src/guanaco_manifest.cpp
    src/guanaco_steppeloader.cpp
    src/guanaco_herdcache.cpp
    src/guanaco_router.cpp
    src/guanaco_executor.cpp
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