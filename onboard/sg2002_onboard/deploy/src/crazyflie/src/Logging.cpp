// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "Logging.h"
#include <iostream>
#include <sstream>

namespace crazyflie
{

    uint8_t LoggingBlock::nextBlockId_ = 0;

    LoggingBlock::LoggingBlock(std::shared_ptr<CrazyflieLink> link)
        : link_(link), blockId_(nextBlockId_++), started_(false)
    {
        // Register callback for LOG_DATA channel
        link_->registerCallback(blockId_,
                                [this](const std::vector<uint8_t> &data)
                                {
                                    if (data.empty() || data[1] != blockId_)
                                    {
                                        // Not for this block
                                        return;
                                    }

                                    const uint32_t timestamp = data[2] | data[3] << 8 | data[4] << 16;

                                    // If callback is set, call it with the data (skip block ID)
                                    if (callback_)
                                    {
                                        std::vector<uint8_t> payload(data.begin() + 5, data.end());
                                        callback_(timestamp, payload);
                                    }
                                });
    }

    LoggingBlock::~LoggingBlock()
    {
        if (started_)
        {
            stop();
        }
    }

    void LoggingBlock::addVariable(const std::string &name, const std::string &type, uint16_t id)
    {
        variables_.push_back(std::make_tuple(name, type, id));
    }

    bool LoggingBlock::start(uint8_t period)
    {
        if (variables_.empty())
        {
            std::cerr << "No variables added to logging block" << std::endl;
            return false;
        }

        // Create a LOG_CREATE packet
        Packet createPacket(LOGGING, CHAN_SETTINGS);
        createPacket.addByte(CMD_CREATE_BLOCK_V2);
        createPacket.addByte(blockId_);

        // Add all variables
        for (const auto &var : variables_)
        {
            const std::string &name = std::get<0>(var);
            const std::string &type = std::get<1>(var);
            uint16_t id = std::get<2>(var);

            // Add type
            LogVariableType typeEnum = getTypeFromString(type);
            createPacket.addByte(static_cast<uint8_t>(typeEnum | typeEnum << 4)); // todo. fetch_as and store_as

            // Add variable ID (little-endian)
            createPacket.addByte(id & 0xFF);
            createPacket.addByte((id >> 8) & 0xFF);
        }

        // Send the create packet
        link_->sendPacket(createPacket);

        // Create and send a LOG_START packet
        Packet startPacket(LOGGING, CHAN_SETTINGS);
        startPacket.addByte(CMD_START_LOGGING);
        startPacket.addByte(blockId_);
        startPacket.addByte(period);
        link_->sendPacket(startPacket);

        started_ = true;
        return true;
    }

    bool LoggingBlock::stop()
    {
        if (!started_)
        {
            return false;
        }

        // Create and send a LOG_STOP packet
        Packet stopPacket(LOGGING, CHAN_SETTINGS);
        stopPacket.addByte(CMD_STOP_LOGGING);
        stopPacket.addByte(blockId_);
        link_->sendPacket(stopPacket);

        started_ = false;
        return true;
    }

    void LoggingBlock::setCallback(DataCallback callback)
    {
        callback_ = callback;
    }

    LogVariableType LoggingBlock::getTypeFromString(const std::string &type)
    {
        if (type == "uint8" || type == "uint8_t")
            return LOG_UINT8;
        if (type == "uint16" || type == "uint16_t")
            return LOG_UINT16;
        if (type == "uint32" || type == "uint32_t")
            return LOG_UINT32;
        if (type == "int8" || type == "int8_t")
            return LOG_INT8;
        if (type == "int16" || type == "int16_t")
            return LOG_INT16;
        if (type == "int32" || type == "int32_t")
            return LOG_INT32;
        if (type == "float")
            return LOG_FLOAT;
        if (type == "fp16")
            return LOG_FP16;

        // Default to float
        return LOG_FLOAT;
    }

} // namespace crazyflie