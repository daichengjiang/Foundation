// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "recorder.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

DataRecorder::DataRecorder(bool enabled, const std::string& logDir)
    : enabled_(enabled), logDir_(logDir), logFile_() {
    if (enabled_) {
        if (!createLogFile()) {
            std::cerr << "Failed to create log file, recording disabled." << std::endl;
            enabled_ = false;
        }
    }
}

DataRecorder::~DataRecorder() {
    std::lock_guard<std::mutex> lock(fileMutex_);
    if (logFile_.is_open()) {
        logFile_.close();
    }
}

bool DataRecorder::createLogFile() {
    // Create directory if it doesn't exist
    struct stat info;
    if (stat(logDir_.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        // Directory doesn't exist, try to create it
        if (mkdir(logDir_.c_str(), 0755) != 0) {
            std::cerr << "Failed to create directory: " << logDir_ << std::endl;
            return false;
        }
    }
    
    // Create timestamp for filename
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* timeinfo = std::localtime(&now_time_t);
    
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", timeinfo);
    
    std::stringstream ss;
    ss << logDir_ << "/" << time_str << ".log";
    std::string filename = ss.str();
    
    // Open the file
    logFile_.open(filename);
    if (!logFile_.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        return false;
    }
    
    // Write expanded CSV header
    logFile_ << "timestamp,"
             // Control commands
             << "target_roll,target_pitch,target_yaw_rate,thrust,"
             // Position
             << "pos_x,pos_y,pos_z,"
             // Velocity
             << "vel_x,vel_y,vel_z,"
             // Attitude
             << "roll,pitch,yaw,"
             // Angular velocity
             << "gyro_x,gyro_y,gyro_z,"
             // Acceleration
             << "acc_x,acc_y,acc_z"
             << std::endl;
    
    std::cout << "Recording complete flight data to: " << filename << std::endl;
    return true;
}

void DataRecorder::writeData(
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
    float acc_x, float acc_y, float acc_z) {
    
    if (!enabled_ || !logFile_.is_open()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(fileMutex_);
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    // Write complete data as CSV row
    logFile_ << timestamp << ","
             // Control commands
             << target_roll << "," << target_pitch << "," << target_yaw_rate << "," << thrust << ","
             // Position
             << pos_x << "," << pos_y << "," << pos_z << ","
             // Velocity
             << vel_x << "," << vel_y << "," << vel_z << ","
             // Attitude
             << roll << "," << pitch << "," << yaw << ","
             // Angular velocity
             << gyro_x << "," << gyro_y << "," << gyro_z << ","
             // Acceleration
             << acc_x << "," << acc_y << "," << acc_z
             << std::endl;
}
