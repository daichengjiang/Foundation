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

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "logger.h"

enum class ControlState {
    LOCKED,
    UNLOCKED,
    AUTO
};

enum class CommandType {
    TAKEOFF,
    LAND,
    ENTER_AUTO_MODE,
    EXIT_AUTO_MODE,
    JOYSTICK_CONTROL,
    SET_STATE
};

class Receiver {
public:
    Receiver(int port, Logger* logger);
    ~Receiver();

    void start();
    void stop();

    // Control state
    void setControlState(ControlState state);
    ControlState getControlState();

    // Command getters
    bool getTakeoffCommand();
    bool getLandCommand();
    bool getAutoModeEnabled();
    bool getJoystickCommand(float& roll, float& pitch, float& yawRate, float& heightDelta);

    // Command clearers
    void clearTakeoffCommand();
    void clearLandCommand();
    void clearJoystickCommand();
    
    // Height management
    float getCurrentHeight() const;
    void setCurrentHeight(float height);

private:
    // Socket management
    bool initSocket();
    void closeSocket();
    void acceptClients();
    void handleClient(int clientSocket);
    bool parseMessage(const std::vector<uint8_t>& data);

    // State management
    void setState(ControlState newState);
    void logCommand(CommandType type, const std::vector<float>& params);

    // Utility functions
    std::string commandTypeToString(CommandType type);
    std::string controlStateToString(ControlState state);

    // Socket variables
    int port_;
    int serverSocket_;
    std::atomic<bool> running_;

    // Client management
    std::mutex clientsMutex_;
    std::unordered_map<int, std::thread> clientThreads_;

    // Command variables
    std::mutex commandMutex_;
    ControlState controlState_ = ControlState::LOCKED;
    bool takeoffRequested_ = false;
    bool landRequested_ = false;
    bool autoModeEnabled_ = false;
    bool joystickCommandReceived_ = false;
    float joystickRoll_ = 0.0f;
    float joystickPitch_ = 0.0f;
    float joystickYawRate_ = 0.0f;
    float joystickHeightDelta_ = 0.0f;
    float currentHeight_ = 0.0f;

    // Logger
    Logger* logger_;
};
