cmake_minimum_required (VERSION 3.0)
project (myproject)
enable_language(CUDA)
find_package(CUDA REQUIRED)	#find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)
add_subdirectory(src)


add_executable(main main.cpp)

target_link_libraries(main MiniDNNLib)
target_link_libraries(main MiniDNNLib ${CUDA_LIBRARIES})   