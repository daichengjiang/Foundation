// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "observation_buffer.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>

ObservationBuffer::ObservationBuffer(int historyLength, const std::string& modelPath)
    : historyLength(historyLength), 
      haveOrientation(false), 
      havePosition(false), 
      haveVelocity(false), 
      haveToFData(false),
      haveDirectionalTof(false),
      roll(0.0f), pitch(0.0f), yaw(0.0f),
      x(0.0f), y(0.0f), z(0.0f),
      vx(0.0f), vy(0.0f), vz(0.0f),
      ref_vx(0.0f), ref_vy(0.0f), target_vel(0.0f), target_z(0.5f),
      tof_left(0.0f), tof_right(0.0f), tof_back(0.0f),
      last_vx_body(0.0f), last_vy_body(0.0f), last_vz_body(0.0f),
      last_x(0.0f), last_y(0.0f), last_z(0.0f)
{
    // Initialize arrays and vectors with pre-allocation for efficiency
    currentFrame.resize(FRAME_SIZE, 0.0f);
    
    // Pre-allocate history buffer to avoid resizing
    historyBuffer.reserve(historyLength);
    for (int i = 0; i < historyLength; i++) {
        historyBuffer.emplace_back(FRAME_SIZE, 0.0f);
    }
    
    tofData.fill(0.0f);
    latestCommand.fill(0.0f);
    
    // Initialize time tracking
    lastUpdateTime = std::chrono::steady_clock::now();
    
    // Initialize TPU with debug output disabled
    try {
        tpu = std::make_shared<TPUInference>(modelPath, false);
        
        if (!tpu->isInitialized()) {
            std::cerr << "TPU model initialization failed" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize TPU model: " << e.what() << std::endl;
    }
}

ObservationBuffer::~ObservationBuffer() {
    // TPU will be cleaned up by the shared_ptr
}

void ObservationBuffer::resetTimestamps() {
    // Reset timing info to avoid large dt values after reconnection or initialization
    lastUpdateTime = std::chrono::steady_clock::now();
    
    // Reset velocity and position trackers to avoid jumps
    last_vx_body = vx;
    last_vy_body = vy;
    last_vz_body = vz;
    last_x = x;
    last_y = y;
    last_z = z;
}

void ObservationBuffer::updateOrientation(float roll, float pitch, float yaw) {
    this->roll = roll;
    this->pitch = pitch;
    this->yaw = yaw;
    haveOrientation = true;
}

void ObservationBuffer::updatePosition(float x, float y, float z) {
    // Store the last position
    this->last_x = this->x;
    this->last_y = this->y;
    this->last_z = this->z;
    
    // Update current position
    this->x = x;
    this->y = y;
    this->z = z;
    havePosition = true;
}

void ObservationBuffer::updateVelocity(float vx, float vy, float vz) {
    this->vx = vx;
    this->vy = vy;
    this->vz = vz;
    haveVelocity = true;
}

void ObservationBuffer::updateToFData(const std::array<float, TOF_SIZE>& tofData) {
    this->tofData = tofData;
    haveToFData = true;
}

void ObservationBuffer::setTargetVelocity(float vx, float vy, float target_vel, float target_z) {
    this->ref_vx = vx;
    this->ref_vy = vy;
    this->target_vel = target_vel;
    this->target_z = target_z;
}

void ObservationBuffer::updateDirectionalTof(float left, float right, float back) {
    this->tof_left = left;
    this->tof_right = right;
    this->tof_back = back;
    haveDirectionalTof = true;
}

bool ObservationBuffer::isFrameComplete() {
    bool status = haveOrientation && havePosition && haveVelocity && haveToFData && haveDirectionalTof;
    return status;
}

std::array<float, 9> ObservationBuffer::eulerToRotationMatrix(float roll, float pitch, float yaw) {
    // Create rotation matrix from Euler angles (roll, pitch, yaw)
    // This matches the rotation matrix used in the simulation
    float cr = std::cos(roll);
    float sr = std::sin(roll);
    float cp = std::cos(pitch);
    float sp = std::sin(pitch);
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    
    // Create 3x3 rotation matrix (body to world)
    std::array<float, 9> rotMatrix;
    
    // Row 1
    rotMatrix[0] = cp * cy;
    rotMatrix[1] = (sr * sp * cy) - (cr * sy);
    rotMatrix[2] = (cr * sp * cy) + (sr * sy);
    
    // Row 2
    rotMatrix[3] = cp * sy;
    rotMatrix[4] = (sr * sp * sy) + (cr * cy);
    rotMatrix[5] = (cr * sp * sy) - (sr * cy);
    
    // Row 3
    rotMatrix[6] = -sp;
    rotMatrix[7] = sr * cp;
    rotMatrix[8] = cr * cp;
    
    return rotMatrix;
}

bool ObservationBuffer::processFrame() {
    // Calculate dt since last update for delta position
    auto currentTime = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(currentTime - lastUpdateTime).count();
    lastUpdateTime = currentTime;
    
    // Limit dt to reasonable values
    dt = std::clamp(dt, 0.001f, 0.1f);
    
    // Transform velocity from world to body frame
    float vx_body, vy_body, vz_body;
    transform_vel_from_world_to_body(vx, vy, vz, roll, pitch, yaw, vx_body, vy_body, vz_body);
    
    // Calculate delta position in body frame
    float delta_x = vx_body * dt;
    float delta_y = vy_body * dt;
    float delta_z = vz_body * dt;
    
    // Convert Euler angles to rotation matrix
    std::array<float, 9> rotMatrix = eulerToRotationMatrix(roll, pitch, yaw);
    
    // Build the current frame
    int idx = 0;

    // 1. Add velocity data [vx, vy, vz] - scaled by 26
    currentFrame[idx++] = vx_body * 26.0f;
    currentFrame[idx++] = vy_body * 26.0f;
    currentFrame[idx++] = vz_body * 26.0f;

    // 2. Add rotation matrix (9 elements) - scaled by 128
    for (int i = 0; i < 9; i++) {
        currentFrame[idx++] = rotMatrix[i] * 128.0f;
    }
    
    // 3. Add delta position in body frame - scaled by 250
    currentFrame[idx++] = delta_x * 250.0f;
    currentFrame[idx++] = delta_y * 250.0f;
    currentFrame[idx++] = delta_z * 250.0f;
    
    // 4. Add direction to goal (normalized) [ref_vx, ref_vy] - scaled by 128
    // These values should already be normalized before passing to setTargetVelocity
    currentFrame[idx++] = ref_vx * 128.0f;
    currentFrame[idx++] = ref_vy * 128.0f;
    
    // 5. Add target z - scaled by 60
    currentFrame[idx++] = target_z * 60.0f;
    
    // 6. Add target velocity - scaled by 60
    currentFrame[idx++] = target_vel * 60.0f;
    
    // 7. Add current z position - scaled by 30
    currentFrame[idx++] = z * 30.0f;
    
    // 8. Add last actions [last_roll_rate, last_pitch_rate, last_yaw_rate, last_thrust] - scaled by 128
    currentFrame[idx++] = latestCommand[0] * 128.0f; // roll_rate
    currentFrame[idx++] = latestCommand[1] * 128.0f; // pitch_rate
    currentFrame[idx++] = latestCommand[2] * 128.0f; // yaw_rate
    currentFrame[idx++] = latestCommand[3] * 128.0f; // thrust

    // 9. Add ToF data - scaled by 30
    for (size_t i = 0; i < TOF_SIZE; i++) {
        currentFrame[idx++] = tofData[i] * 30.0f;
    }
    
    // 10. Add directional ToF readings - scaled by 30
    currentFrame[idx++] = tof_left * 30.0f;
    currentFrame[idx++] = tof_right * 30.0f;
    currentFrame[idx++] = tof_back * 30.0f;

    // Store velocity for next frame's calculations
    last_vx_body = vx_body;
    last_vy_body = vy_body;
    last_vz_body = vz_body;

    // Update history buffer
    updateHistoryBuffer();

    // Run inference if TPU is initialized
    bool result = false;
    if (tpu && tpu->isInitialized()) {
        // Get network input (avoid copy by using move semantics)
        std::vector<float> networkInput = createNetworkInput();

        // Run inference through TPU
        double inference_time_ms = 0.0;
        try {
            // Call TPU inference which returns a vector of outputs
            std::vector<float> output = tpu->inference(networkInput, inference_time_ms);

            // Process the output (assuming output is [roll_rate, pitch_rate, yaw_rate, thrust])
            if (!output.empty()) {
                // Take only the first 4 values
                for (int i = 0; i < 4 && i < output.size(); i++) {
                    // Clamp to [-1,1] range directly
                    latestCommand[i] = std::clamp(output[i], -1.0f, 1.0f);
                }
                
                // Map thrust from [-1,1] to [0,1]
                latestCommand[3] = (latestCommand[3] + 1.0f) * 0.5f;

                // Reset frame flags after successfully processing
                resetFrameFlags();
                result = true;
            }
        } catch (const std::exception& e) {
            std::cerr << "TPU inference error: " << e.what() << std::endl;
        }
    }

    // Reset frame flags even if inference failed
    if (!result) {
        resetFrameFlags();
    }

    return result;
}

void ObservationBuffer::getCommand(float& rollRate, float& pitchRate, float& yawRate, float& thrust) {
    rollRate = latestCommand[0];
    pitchRate = latestCommand[1];
    yawRate = latestCommand[2];
    thrust = latestCommand[3];
}

const std::vector<float>& ObservationBuffer::getCurrentFrame() const {
    return currentFrame;
}

const std::vector<std::vector<float>>& ObservationBuffer::getHistoryBuffer() const {
    return historyBuffer;
}

void ObservationBuffer::updateHistoryBuffer() {
    // Shift all entries one step back efficiently
    if (historyLength > 1) {
        for (int i = historyLength - 1; i > 0; i--) {
            // Use swap for efficiency rather than copying
            historyBuffer[i].swap(historyBuffer[i - 1]);
        }
        // Copy current frame to the front
        historyBuffer[0] = currentFrame;
    } else if (historyLength == 1) {
        historyBuffer[0] = currentFrame;
    }
}

void ObservationBuffer::resetFrameFlags() {
    // haveOrientation = false;
    // havePosition = false;
    // haveVelocity = false;
    // haveToFData = false;
    // haveDirectionalTof = false;
}

std::vector<float> ObservationBuffer::createNetworkInput() {
    // Create a flattened vector of all history frames for network input
    const size_t expectedSize = historyLength * FRAME_SIZE;
    std::vector<float> input;
    input.reserve(expectedSize);

    // Flatten the history buffer
    for (int i = 0; i < historyLength; i++) {
        input.insert(input.end(), historyBuffer[i].begin(), historyBuffer[i].end());
    }

    return input;
}

void transform_vel_from_world_to_body(
    float vx_world, float vy_world, float vz_world,
    float roll, float pitch, float yaw,
    float& vx_body, float& vy_body, float& vz_body
) {
    float cr = cos(roll);
    float sr = sin(roll);
    float cp = cos(pitch);
    float sp = sin(pitch);
    float cy = cos(yaw);
    float sy = sin(yaw);

    vx_body = cp * cy * vx_world + (cy * sp * sr - cr * sy) * vy_world + (cr * cy * sp + sr * sy) * vz_world;
    vy_body = cp * sy * vx_world + (cr * cy + sp * sr * sy) * vy_world + (cr * sy * sp - cy * sr) * vz_world;
    vz_body = -sp * vx_world + cp * sr * vy_world + cr * cp * vz_world;
}