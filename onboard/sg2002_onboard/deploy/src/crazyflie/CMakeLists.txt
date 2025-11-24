cmake_minimum_required(VERSION 3.10)

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Add serial library
add_subdirectory(${CMAKE_SOURCE_DIR}/src/serial ${CMAKE_BINARY_DIR}/src/serial)

# Create crazyflie library
add_library(crazyflie
    src/CrazyflieLink.cpp
    src/Logging.cpp
    src/Commander.cpp
)

# Set include directories for the library
target_include_directories(crazyflie PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/serial/include
)

# We need to make the include directory available to other targets
set(CRAZYFLIE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)

# Link with the serial library
target_link_libraries(crazyflie PRIVATE serial)
