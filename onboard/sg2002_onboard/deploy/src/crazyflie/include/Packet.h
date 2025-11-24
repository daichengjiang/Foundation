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

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace crazyflie {

#define CRTP_MAXSIZE 32

// CRTP PORT definitions
enum Port : uint8_t {
    CONSOLE     = 0x00,
    PARAMETERS  = 0x02,
    COMMANDER   = 0x03,
    LOGGING     = 0x05,
    LOCALIZATION= 0x06,
    COMMANDER_GENERIC = 0x07,
    SETPOINT_HL = 0x08,
};

// Packet channel definitions
enum LoggingChannel : uint8_t {
    CHAN_TOC        = 0x00,
    CHAN_SETTINGS   = 0x01,
    CHAN_LOGDATA    = 0x02,
};

// Commands used when accessing the Log configurations
enum LoggingCommands : uint8_t {
    CMD_CREATE_BLOCK = 0,
    CMD_APPEND_BLOCK = 1,
    CMD_DELETE_BLOCK = 2,
    CMD_START_LOGGING = 3,
    CMD_STOP_LOGGING = 4,
    CMD_RESET_LOGGING = 5,
    CMD_CREATE_BLOCK_V2 = 6,
    CMD_APPEND_BLOCK_V2 = 7,
};

class Packet {
public:
    // Constructor
    Packet() {
        data_.reserve(CRTP_MAXSIZE);
        data_.push_back(0);  // Set header to 0
    }
    
    // Destructor
    ~Packet() {
    }
    
    // Create a packet with predefined port and channel
    Packet(uint8_t port, uint8_t channel) {
        data_.reserve(CRTP_MAXSIZE);
        data_.push_back(((port & 0x0F) << 4) | 3 << 2 | (channel & 0x03));
    }
    
    // Add data to the packet
    void addByte(uint8_t value) {
        if (data_.size() < CRTP_MAXSIZE) {
            data_.push_back(value);
        }
    }
    
    void addBytes(const uint8_t* data, size_t length) {
        if (data_.size() + length > CRTP_MAXSIZE)
        {
            throw std::runtime_error("Packet overflow: not enough space to add all bytes");
        }
        data_.insert(data_.end(), data, data + length);
    }
    
    // Get raw packet data
    uint8_t* raw() {
        return data_.data();
    }
    
    const uint8_t* raw() const {
        return data_.data();
    }
    
    // Get packet size
    size_t size() const {
        return data_.size();
    }
    
    // Get/set packet port
    uint8_t port() const {
        return (data_[0] >> 4) & 0x0F;
    }
    
    void setPort(uint8_t port) {
        data_[0] = (data_[0] & 0x0F) | ((port & 0x0F) << 4);
    }
    
    // Get/set packet channel
    uint8_t channel() const {
        return data_[0] & 0x03;
    }
    
    void setChannel(uint8_t channel) {
        data_[0] = (data_[0] & 0xFC) | (channel & 0x03);
    }
    
    // Used by priority queue
    uint32_t seq_ = 0;
    
    bool operator<(const Packet& other) const {
        return seq_ > other.seq_;
    }
    
private:
    std::vector<uint8_t> data_;
};

} // namespace crazyflie