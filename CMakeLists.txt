cmake_minimum_required (VERSION 3.5)
project (myproject)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Collect all source files in the current directory
aux_source_directory(. DIR_SRCS)

# Add the src subdirectory
add_subdirectory(src)

# Add the executable
add_executable(demo demo.cc)
add_executable(main main.cpp)

# Link the executable to the library
target_link_libraries(demo MiniDNNLib)
target_link_libraries(main MiniDNNLib)

