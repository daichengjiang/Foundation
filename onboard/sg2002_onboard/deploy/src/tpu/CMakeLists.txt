cmake_minimum_required(VERSION 3.10)
project(tpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH to point to the TPU_SDK installation")
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

set(CVI_LIBS ${CVI_LIBS} cviruntime-static cvikernel-static)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel-static)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread z)

# Add librarie
add_library(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/tpu.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CVI_LIBS}
    ${EXTRA_LIBS}
    pthread
    m
    dl
)