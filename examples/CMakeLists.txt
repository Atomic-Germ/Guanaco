cmake_minimum_required(VERSION 3.15)

add_executable(guanaco_example guanaco_example.cpp)
target_link_libraries(guanaco_example PRIVATE guanaco)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(guanaco_example PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS guanaco_example DESTINATION bin)