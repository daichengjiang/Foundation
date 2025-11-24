// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "Commander.h"
#include <cstring>

namespace crazyflie {

Commander::Commander(std::shared_ptr<CrazyflieLink> link)
    : link_(link)
{
}

Commander::~Commander() {
}

void Commander::sendSetpoint(float roll, float pitch, float yawrate, uint16_t thrust) {
    Packet p(COMMANDER, 0);
    // Pack data: roll, pitch, yaw (floats) and thrust (uint16_t)
    uint8_t data[14];
    
    // Copy float values
    memcpy(&data[0], &roll, sizeof(float));
    memcpy(&data[4], &pitch, sizeof(float));
    memcpy(&data[8], &yawrate, sizeof(float));
    
    // Add thrust (little endian)
    data[12] = thrust & 0xFF;
    data[13] = (thrust >> 8) & 0xFF;
    
    // Add data to packet
    p.addBytes(data, sizeof(data));
    
    // Send the packet
    link_->sendPacket(p);
}

void Commander::sendVelocityWorld(float vx, float vy, float vz, float yawrate) {
    Packet p(COMMANDER_GENERIC, SET_SETPOINT_CHANNEL);

    uint8_t data[17];

    // Use TYPE_VELOCITY_WORLD instead of legacy
    data[0] = TYPE_VELOCITY_WORLD;
    memcpy(&data[1], &vx, sizeof(float));
    memcpy(&data[5], &vy, sizeof(float));
    memcpy(&data[9], &vz, sizeof(float));
    memcpy(&data[13], &yawrate, sizeof(float));

    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::sendPosition(float x, float y, float z, float yaw) {
    Packet p(COMMANDER_GENERIC, SET_SETPOINT_CHANNEL);

    uint8_t data[17];
    
    data[0] = TYPE_POSITION; 
    memcpy(&data[1], &x, sizeof(float));
    memcpy(&data[5], &y, sizeof(float));
    memcpy(&data[9], &z, sizeof(float));
    memcpy(&data[13], &yaw, sizeof(float));
    
    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::sendHover(float roll, float pitch, float yawrate, float height)
{
    Packet p(COMMANDER_GENERIC, SET_SETPOINT_CHANNEL);

    uint8_t data[17];

    // Copy float values
    data[0] = TYPE_HOVER;
    *((float *)&data[1]) = roll; // roll
    *((float *)&data[5]) = pitch; // pitch
    *((float *)&data[9]) = yawrate; // yawrate
    *((float *)&data[13]) = height; // zdistance

    // Add data to packet
    p.addBytes(data, sizeof(data));

    // Send the packet
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::sendStop() {
    Packet p(COMMANDER_GENERIC, 0);

    uint8_t data[1];
    data[0] = TYPE_STOP; 
    
    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::sendNotifySetpointStop(uint32_t remain_valid_milliseconds) {
    Packet p(COMMANDER_GENERIC, META_COMMAND_CHANNEL);

    uint8_t data[5];
    data[0] = TYPE_META_COMMAND_NOTIFY_SETPOINT_STOP;
    
    // Copy the uint32 value
    memcpy(&data[1], &remain_valid_milliseconds, sizeof(uint32_t));
    
    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::sendZDistance(float roll, float pitch, float yawrate, float zdistance) {
    Packet p(COMMANDER_GENERIC, SET_SETPOINT_CHANNEL);

    uint8_t data[17];
    
    data[0] = TYPE_ZDISTANCE;
    memcpy(&data[1], &roll, sizeof(float));
    memcpy(&data[5], &pitch, sizeof(float));
    memcpy(&data[9], &yawrate, sizeof(float));
    memcpy(&data[13], &zdistance, sizeof(float));
    
    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

uint32_t Commander::compressQuaternion(const std::vector<float>& orientation) {
    // This function compresses a quaternion according to the algorithm in quatcompress.h
    // It assumes orientation is [qx, qy, qz, qw] and returns a 32-bit integer
    
    if (orientation.size() != 4) {
        return 0;  // Invalid quaternion
    }
    
    // Normalize the quaternion
    float qx = orientation[0];
    float qy = orientation[1];
    float qz = orientation[2];
    float qw = orientation[3];
    
    float norm = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
    qx /= norm;
    qy /= norm;
    qz /= norm;
    qw /= norm;
    
    // Find the component with largest absolute value
    int i_largest = 0;
    float largest_val = fabsf(qx);
    
    if (fabsf(qy) > largest_val) {
        i_largest = 1;
        largest_val = fabsf(qy);
    }
    if (fabsf(qz) > largest_val) {
        i_largest = 2;
        largest_val = fabsf(qz);
    }
    if (fabsf(qw) > largest_val) {
        i_largest = 3;
        largest_val = fabsf(qw);
    }
    
    // Determine if the largest component is negative
    bool negate = false;
    switch (i_largest) {
        case 0: negate = (qx < 0); break;
        case 1: negate = (qy < 0); break;
        case 2: negate = (qz < 0); break;
        case 3: negate = (qw < 0); break;
    }
    
    // Compress the quaternion
    uint32_t comp = i_largest;
    float components[4] = {qx, qy, qz, qw};
    
    for (int i = 0; i < 4; i++) {
        if (i != i_largest) {
            bool negbit = ((components[i] < 0) != negate);
            uint32_t mag = static_cast<uint32_t>(((1 << 9) - 1) * (fabsf(components[i]) / M_SQRT1_2) + 0.5f);
            comp = (comp << 10) | (negbit << 9) | mag;
        }
    }
    
    return comp;
}

// FIXME: We did not test this function
void Commander::sendFullState(
    const std::vector<float>& pos, 
    const std::vector<float>& vel, 
    const std::vector<float>& acc, 
    const std::vector<float>& orientation, 
    float rollrate, 
    float pitchrate, 
    float yawrate) {
    
    // Check that we have valid vectors
    if (pos.size() != 3 || vel.size() != 3 || acc.size() != 3 || orientation.size() != 4) {
        return;  // Invalid input
    }
    
    Packet p(COMMANDER_GENERIC, 0);

    uint8_t data[31];  // TYPE + 3*3 int16 + uint32 + 3 int16
    data[0] = TYPE_FULL_STATE;
    
    // Convert position, velocity, and acceleration to mm (int16_t)
    int16_t x_mm = static_cast<int16_t>(pos[0] * 1000);
    int16_t y_mm = static_cast<int16_t>(pos[1] * 1000);
    int16_t z_mm = static_cast<int16_t>(pos[2] * 1000);
    
    int16_t vx_mm = static_cast<int16_t>(vel[0] * 1000);
    int16_t vy_mm = static_cast<int16_t>(vel[1] * 1000);
    int16_t vz_mm = static_cast<int16_t>(vel[2] * 1000);
    
    int16_t ax_mm = static_cast<int16_t>(acc[0] * 1000);
    int16_t ay_mm = static_cast<int16_t>(acc[1] * 1000);
    int16_t az_mm = static_cast<int16_t>(acc[2] * 1000);
    
    // Convert rotation rates to 16-bit values
    int16_t roll_rate_packed = static_cast<int16_t>(rollrate * 1000);
    int16_t pitch_rate_packed = static_cast<int16_t>(pitchrate * 1000);
    int16_t yaw_rate_packed = static_cast<int16_t>(yawrate * 1000);
    
    // Compress quaternion
    uint32_t quat_compressed = compressQuaternion(orientation);
    
    // Pack the data
    int idx = 1;
    memcpy(&data[idx], &x_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &y_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &z_mm, sizeof(int16_t)); idx += 2;
    
    memcpy(&data[idx], &vx_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &vy_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &vz_mm, sizeof(int16_t)); idx += 2;
    
    memcpy(&data[idx], &ax_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &ay_mm, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &az_mm, sizeof(int16_t)); idx += 2;
    
    memcpy(&data[idx], &quat_compressed, sizeof(uint32_t)); idx += 4;
    
    memcpy(&data[idx], &roll_rate_packed, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &pitch_rate_packed, sizeof(int16_t)); idx += 2;
    memcpy(&data[idx], &yaw_rate_packed, sizeof(int16_t)); idx += 2;
    
    p.addBytes(data, idx);
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::takeoff(float absolute_height_m, float duration_s, uint8_t group_mask, float yaw, bool use_current_yaw) {
    float target_yaw = yaw;
    bool useCurrentYaw = use_current_yaw;
    
    Packet p(SETPOINT_HL, 0); // Using channel 0 for high-level commands

    uint8_t data[15];
    int idx = 0;
    
    data[idx++] = COMMAND_TAKEOFF_2;
    data[idx++] = group_mask;
    
    memcpy(&data[idx], &absolute_height_m, sizeof(float));
    idx += sizeof(float);
    
    memcpy(&data[idx], &target_yaw, sizeof(float));
    idx += sizeof(float);
    
    data[idx++] = useCurrentYaw ? 1 : 0;
    
    memcpy(&data[idx], &duration_s, sizeof(float));
    idx += sizeof(float);
    
    p.addBytes(data, idx);
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::land(float absolute_height_m, float duration_s, uint8_t group_mask, float yaw, bool use_current_yaw) {
    float target_yaw = yaw;
    bool useCurrentYaw = use_current_yaw;
    
    Packet p(SETPOINT_HL, 0); // Using channel 0 for high-level commands

    uint8_t data[15];
    int idx = 0;
    
    data[idx++] = COMMAND_LAND_2;
    data[idx++] = group_mask;
    
    memcpy(&data[idx], &absolute_height_m, sizeof(float));
    idx += sizeof(float);
    
    memcpy(&data[idx], &target_yaw, sizeof(float));
    idx += sizeof(float);
    
    data[idx++] = useCurrentYaw ? 1 : 0;
    
    memcpy(&data[idx], &duration_s, sizeof(float));
    idx += sizeof(float);
    
    p.addBytes(data, idx);
    link_->sendPacket(p);
}

// FIXME: We did not test this function
void Commander::stop(uint8_t group_mask) {
    Packet p(SETPOINT_HL, 0); // Using channel 0 for high-level commands
    
    uint8_t data[2];
    data[0] = COMMAND_STOP;
    data[1] = group_mask;
    
    p.addBytes(data, sizeof(data));
    link_->sendPacket(p);
}

void Commander::arm() {
    // Create a packet with platform port and channel
    Packet p(PLATFORM_PORT, PLATFORM_CHANNEL);
    
    // Prepare data for the arm command
    uint8_t data[2];
    data[0] = PLATFORM_COMMAND_ARM_DISARM; // Command ID for arm/disarm
    data[1] = PLATFORM_COMMAND_ARM;        // 1 = arm the motors
    
    // Add data to packet
    p.addBytes(data, sizeof(data));
    
    // Send the packet
    link_->sendPacket(p);
}

void Commander::disarm() {
    // Create a packet with platform port and channel
    Packet p(PLATFORM_PORT, PLATFORM_CHANNEL);
    
    // Prepare data for the disarm command
    uint8_t data[2];
    data[0] = PLATFORM_COMMAND_ARM_DISARM; // Command ID for arm/disarm
    data[1] = PLATFORM_COMMAND_DISARM;     // 0 = disarm the motors
    
    // Add data to packet
    p.addBytes(data, sizeof(data));
    
    // Send the packet
    link_->sendPacket(p);
}

} // namespace crazyflie