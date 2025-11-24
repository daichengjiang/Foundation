/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <mutex>

class DataRecorder {
public:
    // Constructor takes parameters for enabling recording and log directory path
    DataRecorder(bool enabled = false, const std::string& logDir = "log");
    
    // Destructor to cleanly close file handles
    ~DataRecorder();
    
    // Method to write complete control and state data to the log file
    void writeData(
        // Control commands
        float target_roll, float target_pitch, float target_yaw_rate, uint16_t thrust,
        // Position
        float pos_x, float pos_y, float pos_z,
        // Velocity
        float vel_x, float vel_y, float vel_z,
        // Attitude
        float roll, float pitch, float yaw,
        // Angular velocity
        float gyro_x, float gyro_y, float gyro_z,
        // Acceleration
        float acc_x, float acc_y, float acc_z
    );
    
    // Check if recorder is enabled
    bool isEnabled() const { return enabled_; }

private:
    bool enabled_;
    std::string logDir_;
    std::ofstream logFile_;
    std::mutex fileMutex_;
    
    // Create log directory and file with timestamp
    bool createLogFile();
};
