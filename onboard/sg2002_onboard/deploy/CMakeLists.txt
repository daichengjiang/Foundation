cmake_minimum_required(VERSION 3.10)
project(crazyflie-deploy)

# Add TEST option (default OFF)
option(TEST "Build test programs" OFF)

# Set build type to Release with maximum optimization level
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-static")

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH to point to the TPU_SDK installation")
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

# Add necessary subdirectories
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/crazyflie)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/tofsensem)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/tpu)

# Create executable
add_executable(${PROJECT_NAME}
    main/main.cpp
    main/observation/observation_buffer.cpp
    main/logger/logger.cpp
    main/logger/recorder.cpp
    main/receiver/receiver.cpp
)

# Include the observation files in your main target
target_include_directories(${PROJECT_NAME} PRIVATE
    main/observation
    main/logger
    main/receiver
    ${CMAKE_CURRENT_LIST_DIR}/src/crazyflie/include
    ${CMAKE_CURRENT_LIST_DIR}/src/tofsensem/utils
    ${CMAKE_CURRENT_LIST_DIR}/src/tofsensem/wrapper
    ${CMAKE_CURRENT_LIST_DIR}/src/tpu
)

# Link with crazyflie library
target_link_libraries(${PROJECT_NAME} PRIVATE crazyflie tofsensem tpu)

# Install the executable
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)

# -----------------------------------------------------------------------------
# TEST-RELATED SECTION
# All test-related code is grouped here for better organization
# -----------------------------------------------------------------------------
if(TEST)
    # Add test executables
    # Add new ToF test executable
    add_executable(tof-test
        main/test/tof_main.cpp
        main/observation/observation_buffer.cpp
        main/logger/logger.cpp
    )

    # Add new serial test executable
    add_executable(serial-test
        main/test/serial_main.cpp
    )

    # Add standard library-only serial test executable
    add_executable(serial-test-std
        main/test/serial_main_std.cpp
    )

    # Add check endian
    add_executable(check_endian
        main/test/check_endian.c
    )

    # Configure test executables
    # Include directories for ToF test target
    target_include_directories(tof-test PRIVATE
        main
        main/observation
        main/logger
        ${CMAKE_CURRENT_LIST_DIR}/src/crazyflie/include
        ${CMAKE_CURRENT_LIST_DIR}/src/tofsensem/utils
        ${CMAKE_CURRENT_LIST_DIR}/src/tofsensem/wrapper
        ${CMAKE_CURRENT_LIST_DIR}/src/tpu
    )

    # Include directories for serial test target
    target_include_directories(serial-test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/serial/include
    )
    
    # Link libraries for test executables
    target_link_libraries(tof-test PRIVATE crazyflie tofsensem tpu)
    target_link_libraries(serial-test PRIVATE serial pthread)
    target_link_libraries(serial-test-std PRIVATE pthread)
    
    # Install test executables
    install(TARGETS tof-test
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test
    )
    install(TARGETS serial-test
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test
    )
    install(TARGETS serial-test-std
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test
    )
    install(TARGETS check_endian
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test
    )
endif()
# -----------------------------------------------------------------------------
# END OF TEST-RELATED SECTION
# -----------------------------------------------------------------------------

install(PROGRAMS ${CMAKE_CURRENT_LIST_DIR}/run.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
install(PROGRAMS ${CMAKE_CURRENT_LIST_DIR}/model/model.cvimodel
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
