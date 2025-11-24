cmake_minimum_required(VERSION 3.10)
project(tofsensem)

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define nlink_unpack as a library first, before adding subdirectories
# Source files for nlink_unpack library
file(GLOB NLINK_UNPACK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nlink_unpack/*.c
)

# Create the nlink_unpack library
add_library(nlink_unpack STATIC ${NLINK_UNPACK_SOURCES})
target_include_directories(nlink_unpack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nlink_unpack
)

# Define protocol_extracter as a library
# Source files for protocol_extracter library
file(GLOB PROTOCOL_EXTRACTER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/protocol_extracter/*.cpp
)

# Create the protocol_extracter library
add_library(protocol_extracter STATIC ${PROTOCOL_EXTRACTER_SOURCES})
target_include_directories(protocol_extracter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/protocol_extracter
)

# Source files for the main tofsensem library
set(TOFSENSEM_SOURCES
    utils/init_serial.cpp
    utils/nlink_protocol.cpp
    wrapper/tofImpl.cpp
)

# Create the tofsensem library
add_library(${PROJECT_NAME} ${TOFSENSEM_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/wrapper
        ${CMAKE_CURRENT_SOURCE_DIR}/../serial/include
)

# Link against dependencies
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        serial
        nlink_unpack
        protocol_extracter
)

# Build the example if desired
option(BUILD_TOFSENSEM_EXAMPLES "Build the tofsensem examples" ON)
if(BUILD_TOFSENSEM_EXAMPLES)
    add_executable(tofsensem_example examples/main.cpp)
    target_link_libraries(tofsensem_example
        PRIVATE
            ${PROJECT_NAME}
    )
endif()
