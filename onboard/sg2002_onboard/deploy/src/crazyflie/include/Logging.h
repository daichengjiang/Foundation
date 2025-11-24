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
#include <vector>
#include <functional>
#include <memory>
#include <map>
#include "CrazyflieLink.h"

namespace crazyflie {

enum LogVariableType : uint8_t {
    LOG_UINT8 = 0x01,
    LOG_UINT16 = 0x02,
    LOG_UINT32 = 0x03,
    LOG_INT8 = 0x04,
    LOG_INT16 = 0x05,
    LOG_INT32 = 0x06,
    LOG_FLOAT = 0x07,
    LOG_FP16 = 0x08
};

class LoggingBlock : public std::enable_shared_from_this<LoggingBlock> {
public:
    using DataCallback = std::function<void(const uint32_t, const std::vector<uint8_t>&)>;
    
    LoggingBlock(std::shared_ptr<CrazyflieLink> link);
    ~LoggingBlock();
    
    // Add a variable to log
    void addVariable(const std::string& name, const std::string& type, uint16_t id);
    
    // Start logging with this block at given period (ms)
    bool start(uint8_t period);
    
    // Stop logging this block
    bool stop();
    
    // Set callback for received data
    void setCallback(DataCallback callback);
    
    // Get block ID
    uint8_t getId() const { return blockId_; }
    
private:
    std::shared_ptr<CrazyflieLink> link_;
    uint8_t blockId_;
    static uint8_t nextBlockId_;
    std::vector<std::tuple<std::string, std::string, uint16_t>> variables_;
    DataCallback callback_;
    bool started_;
    
    // Convert type string to enum
    LogVariableType getTypeFromString(const std::string& type);
};

} // namespace crazyflie