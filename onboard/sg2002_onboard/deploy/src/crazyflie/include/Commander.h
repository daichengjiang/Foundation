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

#include <memory>
#include <iostream>
#include <cmath>
#include "CrazyflieLink.h"

namespace crazyflie {

#define SET_SETPOINT_CHANNEL 0
#define META_COMMAND_CHANNEL 1

#define TYPE_STOP 0
#define TYPE_VELOCITY_WORLD_LEGACY 1
#define TYPE_ZDISTANCE_LEGACY 2
#define TYPE_HOVER_LEGACY 5
#define TYPE_FULL_STATE 6
#define TYPE_POSITION 7
#define TYPE_VELOCITY_WORLD 8
#define TYPE_ZDISTANCE 9
#define TYPE_HOVER 10

#define TYPE_META_COMMAND_NOTIFY_SETPOINT_STOP 0

// High level commande definitions
#define COMMAND_SET_GROUP_MASK 0
#define COMMAND_STOP 3
#define COMMAND_GO_TO 4
#define COMMAND_START_TRAJECTORY 5
#define COMMAND_DEFINE_TRAJECTORY 6
#define COMMAND_TAKEOFF_2 7
#define COMMAND_LAND_2 8
#define COMMAND_SPIRAL 11
#define COMMAND_GO_TO_2 12

// Platform port and channel definitions
#define PLATFORM_PORT 13
#define PLATFORM_CHANNEL 0

// Platform command definitions
#define PLATFORM_COMMAND_ARM_DISARM 1
#define PLATFORM_COMMAND_ARM 1
#define PLATFORM_COMMAND_DISARM 0

#define ALL_GROUPS 0

#define TRAJECTORY_LOCATION_MEM 1

#define TRAJECTORY_TYPE_POLY4D 0
#define TRAJECTORY_TYPE_POLY4D_COMPRESSED 1

class Commander {
public:
    Commander(std::shared_ptr<CrazyflieLink> link);
    ~Commander();
    
    // Send control command
    void sendSetpoint(float roll, float pitch, float yawrate, uint16_t thrust);

    // Send velocity command in world frame
    void sendVelocityWorld(float vx, float vy, float vz, float yawrate);

    // Send position command
    void sendPosition(float x, float y, float z, float yaw);

    // Send control command with height
    void sendHover(float roll, float pitch, float yawrate, float height);

    // Send stop command
    void sendStop();
    
    // Notify that the setpoint should stop
    void sendNotifySetpointStop(uint32_t remain_valid_milliseconds = 0);
    
    // Send Z distance command
    void sendZDistance(float roll, float pitch, float yawrate, float zdistance);
    
    // Send full state setpoint
    void sendFullState(
        const std::vector<float>& pos, 
        const std::vector<float>& vel, 
        const std::vector<float>& acc, 
        const std::vector<float>& orientation, 
        float rollrate, 
        float pitchrate, 
        float yawrate);

    // Takeoff command
    void takeoff_wait(float *cur_height, float height);

    // Land command
    void land_wait(float *cur_height);
    
    // High-level commands
    void takeoff(float absolute_height_m, float duration_s, uint8_t group_mask = ALL_GROUPS, 
                 float yaw = 0.0, bool use_current_yaw = false);
    void land(float absolute_height_m, float duration_s, uint8_t group_mask = ALL_GROUPS,
              float yaw = 0.0, bool use_current_yaw = false);
    void stop(uint8_t group_mask = ALL_GROUPS);
    
    // Motor arming commands (for brushless motor version)
    void arm();   // Unlock/enable motors
    void disarm(); // Lock/disable motors

private:
    std::shared_ptr<CrazyflieLink> link_;
    uint32_t compressQuaternion(const std::vector<float>& orientation);
};

} // namespace crazyflie