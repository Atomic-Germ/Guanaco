cmake_minimum_required(VERSION 3.15)
project(guanaco VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === Options ===
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GUANACO_BUILD_EXAMPLES "Build example programs" ON)
option(GUANACO_BUILD_TESTS "Build tests" OFF)
option(GUANACO_PATCH_LLAMA_CPP "Apply Guanaco patches to llama.cpp" ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
endif()

find_package(Threads REQUIRED)

# === io_uring ===
find_library(IO_URING_LIBRARY NAMES uring)
if(IO_URING_LIBRARY)
    add_compile_definitions(GUANACO_HAVE_IO_URING=1)
    set(GUANACO_IO_URING_FOUND TRUE)
    message(STATUS "Found io_uring: ${IO_URING_LIBRARY}")
else()
    message(STATUS "io_uring not found, falling back to thread pool")
endif()

find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
    add_compile_definitions(GUANACO_HAVE_MADVISE=1)
endif()

# === Patch llama.cpp ===
if(GUANACO_PATCH_LLAMA_CPP)
    message(STATUS "Applying Guanaco patches to llama.cpp...")
    file(GLOB PATCH_FILES ${CMAKE_SOURCE_DIR}/patches/*.patch)
    foreach(PATCH ${PATCH_FILES})
        get_filename_component(PATCH_NAME ${PATCH} NAME)
        execute_process(
            COMMAND patch -p1 --forward --dry-run -i "${PATCH}"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
            RESULT_VARIABLE PATCH_CHECK
            ERROR_QUIET
            OUTPUT_QUIET
        )
        if(PATCH_CHECK EQUAL 0)
            execute_process(
                COMMAND patch -p1 --forward -i "${PATCH}"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../llama.cpp
                RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_RESULT EQUAL 0)
                message(STATUS "  Applied: ${PATCH_NAME}")
            else()
                message(WARNING "  Failed to apply: ${PATCH_NAME}")
            endif()
        else()
            message(STATUS "  Skipped (already applied): ${PATCH_NAME}")
        endif()
    endforeach()
    add_compile_definitions(GUANACO_ENABLED=1)
endif()

# === Build llama.cpp as a subdirectory ===
# Enable tools and server so they link against the patched libllama
set(LLAMA_BUILD_COMMON ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS  ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP    OFF CACHE BOOL "" FORCE)

# Output goes to ${CMAKE_BINARY_DIR}/bin (guanaco/build/bin)
# Set RPATH so tools pick up our patched libllama.so AND libguanaco.so
# (libguanaco.so is emitted to build/bin via LIBRARY_OUTPUT_DIRECTORY below)
set(CMAKE_BUILD_RPATH  "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
add_subdirectory(${CMAKE_SOURCE_DIR}/../llama.cpp llama.cpp)

# Add guanaco include path so llama.cpp can find guanaco/guanaco.h when patched
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR}/include)

# libllama references guanaco streaming symbols (via GUANACO_ENABLED patches),
# so it must record a DT_NEEDED on libguanaco. Circular with guanaco->llama is
# fine for shared libraries (both resolve symbols at load time).
target_link_libraries(llama PRIVATE guanaco)

# === Guanaco library ===
add_library(guanaco
    src/guanaco_manifest.cpp
    src/guanaco_steppeloader.cpp
    src/guanaco_herdcache.cpp
    src/guanaco_router.cpp
    src/guanaco_executor.cpp
    src/guanaco_model_hook.cpp
    src/guanaco_io_uring.cpp
    src/guanaco_backend_wrap.cpp
)

target_include_directories(guanaco PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama.cpp
    ${CMAKE_SOURCE_DIR}/../llama.cpp/ggml/src
)

# imatrix prior: compile a copy of llama.cpp's imatrix-loader.cpp directly
# into guanaco (instead of linking libllama-common, which would create a
# circular dependency llama -> guanaco -> llama-common -> llama). A minimal
# shim provides the common.h/log.h it includes. We copy the .cpp into the
# shim dir so its quote-includes ("common.h") resolve to the shim rather
# than the real common/ next to the original source file.
target_sources(guanaco PRIVATE ${CMAKE_SOURCE_DIR}/src/imatrix_shim/imatrix-loader.cpp)
target_include_directories(guanaco PRIVATE
    ${CMAKE_SOURCE_DIR}/src/imatrix_shim
)

target_link_libraries(guanaco PUBLIC
    ggml
    Threads::Threads
)

if(GUANACO_IO_URING_FOUND)
    target_link_libraries(guanaco PUBLIC ${IO_URING_LIBRARY})
endif()

target_compile_definitions(guanaco PUBLIC GUANACO_API_EXPORT)

if(BUILD_SHARED_LIBS)
    set_target_properties(guanaco PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
endif()

# Emit libguanaco.so next to libllama.so so the build-tree RPATH finds it,
# and install it alongside llama.cpp's libs.
set_target_properties(guanaco PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS guanaco LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(GUANACO_BUILD_EXAMPLES)
    add_executable(guanaco_example examples/guanaco_example.cpp)
    target_link_libraries(guanaco_example guanaco)
endif()

if(GUANACO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
