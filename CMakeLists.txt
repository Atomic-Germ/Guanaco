cmake_minimum_required(VERSION 3.15)
project(guanaco VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === Options ===
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GUANACO_BUILD_EXAMPLES "Build example programs" ON)
option(GUANACO_BUILD_TESTS "Build tests" OFF)
option(GUANACO_PATCH_LLAMA_CPP "Apply Guanaco patches to llama.cpp" ON)

# GPU-pill, UMA buffer types and the Vulkan patches are the core of this project.
# Upstream defaults GGML_VULKAN OFF, so a plain `cmake -S . -B build` silently
# produces a CPU-only binary that exercises none of them. Default it ON: someone
# without Vulkan should have to disable it deliberately.
set(GGML_VULKAN ON CACHE BOOL "Vulkan backend (Guanaco gpu-pill and UMA paths require it)")
option(GUANACO_REQUIRE_VULKAN "Fail configuration when the Vulkan backend is unavailable" ON)

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

# === Upstream llama.cpp checkout ===
# llama.cpp/ is a disposable clone of upstream ggml-org/llama.cpp that is not
# tracked here. Clone it at configure time when it is missing so a fresh copy of
# this repository builds without manual setup.
#
# The clone is pinned to the upstream commit that patches/*.patch was generated
# against, so a fresh configure reproduces the exact tree the patchset was
# validated on. llama.cpp moves fast: 150 commits landed after this pin, 19 of
# them touching patched files, and the patchset no longer applies cleanly there.
# Bump GUANACO_LLAMA_CPP_REF only as part of an upstream refresh, together with
# regenerating patches/*.patch -- never on its own.
set(GUANACO_LLAMA_CPP_REPO "https://github.com/ggml-org/llama.cpp" CACHE STRING
    "Upstream repository cloned into ./llama.cpp")
set(GUANACO_LLAMA_CPP_REF "aa94f2086147cd4bfbcff818ce998dffce5fcb39" CACHE STRING
    "Upstream llama.cpp commit the patchset was built against; empty means upstream HEAD")

set(LLAMA_CPP_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)

# Report the commit an existing checkout is on, so a drifted tree is visible
# instead of silently building an untested upstream/patchset combination.
function(guanaco_llama_cpp_head out_var dir)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${dir}" rev-parse HEAD
        RESULT_VARIABLE head_result
        OUTPUT_VARIABLE head_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(head_result EQUAL 0)
        set(${out_var} "${head_out}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

if(EXISTS ${LLAMA_CPP_DIR}/CMakeLists.txt)
    find_package(Git QUIET)
    message(STATUS "Using existing llama.cpp checkout: ${LLAMA_CPP_DIR}")
    if(GIT_FOUND)
        guanaco_llama_cpp_head(existing_head ${LLAMA_CPP_DIR})
        if(existing_head STREQUAL "")
            message(WARNING
                "  Cannot read a git commit for this checkout (missing or broken .git). "
                "Its provenance cannot be verified against the patchset.")
        elseif(GUANACO_LLAMA_CPP_REF AND NOT existing_head STREQUAL GUANACO_LLAMA_CPP_REF)
            message(WARNING
                "  Checkout is at ${existing_head}, but patches/*.patch target "
                "${GUANACO_LLAMA_CPP_REF}. The patchset may not apply, or may apply "
                "to a combination that was never tested. Remove llama.cpp/ to "
                "re-clone at the pinned commit.")
        endif()
    endif()
elseif(EXISTS ${LLAMA_CPP_DIR})
    message(FATAL_ERROR
        "${LLAMA_CPP_DIR} exists but is not a llama.cpp checkout (no CMakeLists.txt). "
        "Remove or move it, then reconfigure.")
else()
    find_package(Git REQUIRED)
    message(STATUS "llama.cpp checkout not found, cloning ${GUANACO_LLAMA_CPP_REPO}")
    # Upstream is large and the clone is long, so transient transport failures
    # (HTTP 408, dropped connections) are common enough to be worth retrying.
    # Start from a clean directory each attempt: a half-written clone makes the
    # next attempt fail with a confusing error.
    set(CLONE_ATTEMPTS 3)
    foreach(attempt RANGE 1 ${CLONE_ATTEMPTS})
        if(attempt GREATER 1)
            file(REMOVE_RECURSE "${LLAMA_CPP_DIR}")
            message(STATUS "  clone attempt ${attempt}/${CLONE_ATTEMPTS}")
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --recursive "${GUANACO_LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
            RESULT_VARIABLE CLONE_RESULT
        )
        if(CLONE_RESULT EQUAL 0)
            break()
        endif()
    endforeach()
    if(NOT CLONE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to clone ${GUANACO_LLAMA_CPP_REPO} after ${CLONE_ATTEMPTS} attempts. "
            "If this is a network or proxy problem, fetch it manually:\n"
            "  git clone --recursive ${GUANACO_LLAMA_CPP_REPO} ${LLAMA_CPP_DIR}\n"
            "  git -C ${LLAMA_CPP_DIR} checkout --detach ${GUANACO_LLAMA_CPP_REF}\n"
            "and reconfigure.")
    endif()
    if(GUANACO_LLAMA_CPP_REF)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${LLAMA_CPP_DIR}" checkout --detach "${GUANACO_LLAMA_CPP_REF}"
            RESULT_VARIABLE CHECKOUT_RESULT
        )
        if(NOT CHECKOUT_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Cloned ${GUANACO_LLAMA_CPP_REPO} but could not check out pinned ref "
                "${GUANACO_LLAMA_CPP_REF}. If it is not reachable from the default "
                "branch, fetch it manually and reconfigure.")
        endif()
        # The clone synced submodules for the default branch, not the pinned
        # commit, so realign them with the tree we are about to patch.
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C "${LLAMA_CPP_DIR}" submodule update --recursive --init
            RESULT_VARIABLE SUBMODULE_RESULT
            ERROR_QUIET
        )
        if(NOT SUBMODULE_RESULT EQUAL 0)
            message(WARNING "  Failed to sync submodules to ${GUANACO_LLAMA_CPP_REF}")
        endif()
        message(STATUS "  Pinned to ${GUANACO_LLAMA_CPP_REF}")
    else()
        message(STATUS "  Using upstream default branch (GUANACO_LLAMA_CPP_REF is empty)")
    endif()
endif()

# === Patch llama.cpp ===
if(GUANACO_PATCH_LLAMA_CPP)
    message(STATUS "Applying Guanaco patches to llama.cpp...")
    file(GLOB PATCH_FILES ${CMAKE_SOURCE_DIR}/patches/*.patch)
    foreach(PATCH ${PATCH_FILES})
        get_filename_component(PATCH_NAME ${PATCH} NAME)
        execute_process(
            COMMAND patch -p1 --forward --dry-run --no-backup-if-mismatch -i "${PATCH}"
            WORKING_DIRECTORY ${LLAMA_CPP_DIR}
            RESULT_VARIABLE PATCH_CHECK
            ERROR_QUIET
            OUTPUT_QUIET
        )
        if(PATCH_CHECK EQUAL 0)
            execute_process(
                COMMAND patch -p1 --forward --no-backup-if-mismatch -i "${PATCH}"
                WORKING_DIRECTORY ${LLAMA_CPP_DIR}
                RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_RESULT EQUAL 0)
                message(STATUS "  Applied: ${PATCH_NAME}")
            else()
                message(WARNING "  Failed to apply: ${PATCH_NAME}")
            endif()
        else()
            execute_process(
                COMMAND patch -p1 --reverse --dry-run --no-backup-if-mismatch -i "${PATCH}"
                WORKING_DIRECTORY ${LLAMA_CPP_DIR}
                RESULT_VARIABLE PATCH_REVERSE_CHECK
                ERROR_QUIET
                OUTPUT_QUIET
            )
            if(PATCH_REVERSE_CHECK EQUAL 0)
                message(STATUS "  Skipped (already applied): ${PATCH_NAME}")
            endif()
        endif()
    endforeach()
    add_compile_definitions(GUANACO_ENABLED=1)
endif()

# === Build llama.cpp as a subdirectory ===
# Enable tools and server so they link against the patched libllama
if(GUANACO_REQUIRE_VULKAN AND NOT GGML_VULKAN)
    message(FATAL_ERROR
        "Guanaco needs the Vulkan backend for gpu-pill and UMA buffer types.\n"
        "Re-run with -DGGML_VULKAN=ON, or -DGUANACO_REQUIRE_VULKAN=OFF to build CPU-only.")
endif()
set(LLAMA_BUILD_COMMON ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS  ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP    OFF CACHE BOOL "" FORCE)

# The server binary links libllama-server-impl.so; upstream only installs that
# library when LLAMA_TOOLS_INSTALL is on, so without this `cmake --install`
# would produce a llama-server that cannot find its own impl library.
set(LLAMA_TOOLS_INSTALL ON  CACHE BOOL "" FORCE)

# Output goes to ${CMAKE_BINARY_DIR}/bin (build/bin)
# Set RPATH so tools pick up our patched libllama.so AND libguanaco.so
# (libguanaco.so is emitted to build/bin via LIBRARY_OUTPUT_DIRECTORY below)
set(CMAKE_BUILD_RPATH  "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
if(UNIX AND NOT APPLE AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    add_link_options(LINKER:--disable-new-dtags)
endif()
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

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
    ${LLAMA_CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/src
)

# imatrix prior: compile a copy of llama.cpp's imatrix-loader.cpp directly
# into guanaco (instead of linking libllama-common, which would create a
# circular dependency llama -> guanaco -> llama-common -> llama). A minimal
# shim provides the common.h/log.h it includes. We copy the .cpp into the
# shim dir so its quote-includes ("common.h") resolve to the shim rather
# than the real common/ next to the original source file.
target_sources(guanaco PRIVATE ${CMAKE_SOURCE_DIR}/src/imatrix-loader.cpp)
target_include_directories(guanaco PRIVATE
    ${CMAKE_SOURCE_DIR}/src
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

# Phase-0 measurement probe for the dense pseudo-expert experiment. Off by
# default; it only links llama.cpp and does not touch production code.
option(GUANACO_BUILD_PROBES "Build Phase-0 measurement probes" OFF)
if(GUANACO_BUILD_PROBES)
    add_executable(tomoe_probe examples/tomoe_probe.cpp)
    target_link_libraries(tomoe_probe PRIVATE llama-common llama Threads::Threads)
    target_compile_features(tomoe_probe PRIVATE cxx_std_17)
endif()

if(GUANACO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
