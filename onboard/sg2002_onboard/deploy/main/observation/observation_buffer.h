/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#ifndef OBSERVATION_BUFFER_H
#define OBSERVATION_BUFFER_H

#include <vector>
#include <array>
#include <mutex>
#include <memory>
#include <functional>
#include <chrono>

#include "tpu.h" // Changed from rknn_impl.h to tpu.h

// Define constants based on simulation environment
#define TOF_WIDTH 8
#define TOF_HEIGHT 8
#define TOF_SIZE (TOF_WIDTH * TOF_HEIGHT)
// Updated frame size to match the RL training code:
// [vx, vy, vz] + [9D rotation matrix] + [delta_x, delta_y, delta_z] + 
// [ref_vx, ref_vy] + [target_z] + [target_vel] + [z] + 
// [last_roll_rate, last_pitch_rate, last_yaw_rate, last_thrust] + 
// [8x8 tof] + [left,right,back]
#define FRAME_SIZE (3 + 9 + 3 + 2 + 1 + 1 + 1 + 4 + TOF_SIZE + 3)

class ObservationBuffer {
public:
    ObservationBuffer(int historyLength, const std::string& modelPath);
    ~ObservationBuffer();

    // Sensor data update methods
    void updateOrientation(float roll, float pitch, float yaw);
    void updatePosition(float x, float y, float z);
    void updateVelocity(float vx, float vy, float vz);
    void updateDirectionalTof(float left, float right, float back);
    void updateToFData(const std::array<float, TOF_SIZE>& tofData);
    
    // Reset time tracking
    void resetTimestamps();
    
    // Set reference/target values
    void setTargetVelocity(float vx, float vy, float target_vel, float target_z);
    
    // Check if we have a complete frame
    bool isFrameComplete();
    
    // Process current frame and run inference if possible
    bool processFrame();
    
    // Get latest command from inference
    void getCommand(float& rollRate, float& pitchRate, float& yawRate, float& thrust);
    
    // Debug data inspection
    const std::vector<float>& getCurrentFrame() const;
    const std::vector<std::vector<float>>& getHistoryBuffer() const;

private:
    // Buffer for observation history
    std::vector<std::vector<float>> historyBuffer;
    
    // Current frame being assembled
    std::vector<float> currentFrame;
    
    // Latest inference output command
    std::array<float, 4> latestCommand;
    
    // Configuration
    int historyLength;
    
    // Sensor data tracking flags
    bool haveOrientation;
    bool havePosition;
    bool haveVelocity;
    bool haveToFData;
    bool haveDirectionalTof;
    
    // Latest sensor values
    float roll, pitch, yaw;
    float x, y, z;
    float vx, vy, vz;
    float ref_vx, ref_vy, target_vel, target_z;
    float tof_left, tof_right, tof_back;
    std::array<float, TOF_SIZE> tofData;
    
    // Time tracking for delta position calculation
    std::chrono::steady_clock::time_point lastUpdateTime;
    float last_vx_body, last_vy_body, last_vz_body;
    float last_x, last_y, last_z;
    
    // Neural network model (changed from RKNN to TPU)
    std::shared_ptr<TPUInference> tpu;
    
    // Thread safety
    std::mutex dataMutex;
    
    // Private helper methods
    void updateHistoryBuffer();
    void resetFrameFlags();
    std::vector<float> createNetworkInput();
    
    // Helper to convert Euler angles to rotation matrix (3x3=9 elements)
    std::array<float, 9> eulerToRotationMatrix(float roll, float pitch, float yaw);
};

// Convert velocity from world to body frame
void transform_vel_from_world_to_body(
    float vx_world, float vy_world, float vz_world,
    float roll, float pitch, float yaw,
    float& vx_body, float& vy_body, float& vz_body);

#endif // OBSERVATION_BUFFER_H
