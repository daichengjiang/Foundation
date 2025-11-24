// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "receiver.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <cstring>

Receiver::Receiver(int port, Logger* logger)
    : port_(port),
      serverSocket_(-1),
      running_(false),
      logger_(logger),
      currentHeight_(0.0f) {
    // Add essential log variables
    logger_->addVariable("cmd_joystick", "Joystick Command: {}", LogLevel::INFO);
    logger_->addVariable("cmd_takeoff", "Takeoff Command: {}", LogLevel::INFO);
    logger_->addVariable("cmd_land", "Land Command: {}", LogLevel::INFO);
    logger_->addVariable("control_state", "Control State: {}", LogLevel::INFO);
    
    // Initial state is LOCKED
    setState(ControlState::LOCKED);
}

Receiver::~Receiver() {
    stop();
}

void Receiver::start() {
    if (running_.load()) {
        return;
    }

    if (!initSocket()) {
        std::cerr << "Failed to initialize socket" << std::endl;
        return;
    }

    running_ = true;
    // Start the accept thread
    std::thread acceptThread(&Receiver::acceptClients, this);
    acceptThread.detach();

    logger_->addTempMessage("Receiver started on port " + std::to_string(port_), LogLevel::INFO);
}

void Receiver::stop() {
    if (!running_.load()) {
        return;
    }

    running_ = false;

    // Close all client connections
    {
        std::lock_guard<std::mutex> lock(clientsMutex_);
        for (auto& client : clientThreads_) {
            if (client.second.joinable()) {
                client.second.detach();
            }
            close(client.first);
        }
        clientThreads_.clear();
    }

    closeSocket();
    logger_->addTempMessage("Receiver stopped", LogLevel::INFO);
}

void Receiver::setControlState(ControlState state) {
    std::lock_guard<std::mutex> lock(commandMutex_);
    controlState_ = state;
    logger_->setValue("control_state", controlStateToString(state));
}

bool Receiver::getTakeoffCommand() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    // Only allow takeoff in UNLOCKED state
    return takeoffRequested_ && controlState_ == ControlState::UNLOCKED;
}

bool Receiver::getLandCommand() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    // Allow landing in any state for safety
    return landRequested_;
}

bool Receiver::getAutoModeEnabled() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    return controlState_ == ControlState::AUTO;
}

bool Receiver::getJoystickCommand(float& roll, float& pitch, float& yawRate, float& heightDelta) {
    std::lock_guard<std::mutex> lock(commandMutex_);
    // Only provide joystick commands if in UNLOCKED or AUTO state
    if (joystickCommandReceived_ && (controlState_ == ControlState::UNLOCKED || controlState_ == ControlState::AUTO)) {
        roll = joystickRoll_;
        pitch = joystickPitch_;
        yawRate = joystickYawRate_;
        heightDelta = joystickHeightDelta_;
        return true;
    }
    return false;
}

// Add method to get current drone height
float Receiver::getCurrentHeight() const {
    return currentHeight_;
}

// Add method to set current drone height (called from main.cpp)
void Receiver::setCurrentHeight(float height) {
    std::lock_guard<std::mutex> lock(commandMutex_);
    currentHeight_ = height;
}

ControlState Receiver::getControlState() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    return controlState_;
}

void Receiver::clearTakeoffCommand() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    takeoffRequested_ = false;
    logger_->setValue("cmd_takeoff", "Cleared");
}

void Receiver::clearLandCommand() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    landRequested_ = false;
    logger_->setValue("cmd_land", "Cleared");
}

void Receiver::clearJoystickCommand() {
    std::lock_guard<std::mutex> lock(commandMutex_);
    joystickCommandReceived_ = false;
}

std::string Receiver::commandTypeToString(CommandType type) {
    switch (type) {
        case CommandType::TAKEOFF:
            return "TAKEOFF";
        case CommandType::LAND:
            return "LAND";
        case CommandType::ENTER_AUTO_MODE:
            return "ENTER_AUTO_MODE";
        case CommandType::EXIT_AUTO_MODE:
            return "EXIT_AUTO_MODE";
        case CommandType::JOYSTICK_CONTROL:
            return "JOYSTICK_CONTROL";
        case CommandType::SET_STATE:
            return "SET_STATE";
        default:
            return "UNKNOWN";
    }
}

std::string Receiver::controlStateToString(ControlState state) {
    switch (state) {
        case ControlState::LOCKED: return "LOCKED";
        case ControlState::UNLOCKED: return "UNLOCKED";
        case ControlState::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

bool Receiver::initSocket() {
    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "Failed to set socket options" << std::endl;
        close(serverSocket_);
        return false;
    }

    // Bind the socket
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(serverSocket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind socket to port " << port_ << std::endl;
        close(serverSocket_);
        return false;
    }

    // Listen for connections
    if (listen(serverSocket_, 5) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        close(serverSocket_);
        return false;
    }

    return true;
}

void Receiver::closeSocket() {
    if (serverSocket_ >= 0) {
        close(serverSocket_);
        serverSocket_ = -1;
    }
}

void Receiver::acceptClients() {
    while (running_.load()) {
        // Set up select for timeout
        fd_set readfds;
        struct timeval timeout = {1, 0}; // 1 second timeout
        
        FD_ZERO(&readfds);
        FD_SET(serverSocket_, &readfds);
        
        int ret = select(serverSocket_ + 1, &readfds, NULL, NULL, &timeout);
        if (ret <= 0) {
            if (ret < 0 && errno != EINTR) {
                std::cerr << "Select error: " << strerror(errno) << std::endl;
            }
            continue; // Timeout or interrupted, check running_
        }

        // Accept connection
        struct sockaddr_in client_addr;
        socklen_t addrlen = sizeof(client_addr);
        int clientSocket = accept(serverSocket_, (struct sockaddr*)&client_addr, &addrlen);
        
        if (clientSocket < 0) {
            std::cerr << "Failed to accept client connection" << std::endl;
            continue;
        }

        // Log connection
        char clientIP[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, clientIP, sizeof(clientIP));
        std::string clientInfo = std::string(clientIP) + ":" + std::to_string(ntohs(client_addr.sin_port));
        logger_->addTempMessage("Client connected: " + clientInfo, LogLevel::INFO);

        // Set TCP_NODELAY option
        int flag = 1;
        setsockopt(clientSocket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));

        // Start a thread to handle this client
        {
            std::lock_guard<std::mutex> lock(clientsMutex_);
            clientThreads_[clientSocket] = std::thread(&Receiver::handleClient, this, clientSocket);
            clientThreads_[clientSocket].detach();
        }
    }
}

void Receiver::handleClient(int clientSocket) {
    constexpr size_t bufferSize = 1024;
    std::vector<uint8_t> buffer(bufferSize);
    std::vector<uint8_t> messageBuffer;

    while (running_.load()) {
        // Set up select for timeout
        fd_set readfds;
        struct timeval timeout = {1, 0}; // 1 second timeout
        
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);
        
        int ret = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);
        if (ret <= 0) {
            if (ret < 0 && errno != EINTR) {
                std::cerr << "Select error in client handler: " << strerror(errno) << std::endl;
                break;
            }
            continue; // Timeout or interrupted, check running_
        }

        // Read data
        ssize_t bytesRead = recv(clientSocket, buffer.data(), buffer.size(), 0);
        if (bytesRead <= 0) {
            // Connection closed or error
            if (bytesRead < 0) {
                std::cerr << "Recv error: " << strerror(errno) << std::endl;
            }
            break;
        }

        // Process received data
        for (size_t i = 0; i < bytesRead; i++) {
            messageBuffer.push_back(buffer[i]);
            
            // Simple protocol: Each message ends with '\n'
            if (buffer[i] == '\n') {
                if (parseMessage(messageBuffer)) {
                    // Send acknowledgment to client (optional)
                    const char* ack = "ACK\n";
                    send(clientSocket, ack, strlen(ack), 0);
                }
                messageBuffer.clear();
            }
        }
    }

    // Clean up
    {
        std::lock_guard<std::mutex> lock(clientsMutex_);
        close(clientSocket);
        clientThreads_.erase(clientSocket);
    }
    
    logger_->addTempMessage("Client disconnected", LogLevel::INFO);
}

bool Receiver::parseMessage(const std::vector<uint8_t>& data) {
    // Convert data to string for easier parsing
    std::string message(data.begin(), data.end());
    
    std::istringstream iss(message);
    std::string commandStr;
    iss >> commandStr;
    
    if (commandStr == "JOYSTICK") {
        float roll, pitch, yawRate, heightDelta;
        if (iss >> roll >> pitch >> yawRate >> heightDelta) {
            std::lock_guard<std::mutex> lock(commandMutex_);
            joystickRoll_ = roll;
            joystickPitch_ = pitch;
            joystickYawRate_ = yawRate;
            joystickHeightDelta_ = heightDelta;
            joystickCommandReceived_ = true;
            
            // Log only occasionally to reduce spam
            static int logCounter = 0;
            if (++logCounter % 10 == 0) {
                logCommand(CommandType::JOYSTICK_CONTROL, {roll, pitch, yawRate, heightDelta});
            }
            return true;
        }
    } else if (commandStr == "TAKEOFF") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        if (controlState_ == ControlState::UNLOCKED) {
            takeoffRequested_ = true;
            logCommand(CommandType::TAKEOFF, {});
        }
        return true;
    } else if (commandStr == "LAND") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        landRequested_ = true;
        logCommand(CommandType::LAND, {});
        return true;
    } else if (commandStr == "AUTO_ON") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        if (controlState_ == ControlState::UNLOCKED) {
            setState(ControlState::AUTO);
            logCommand(CommandType::ENTER_AUTO_MODE, {});
        }
        return true;
    } else if (commandStr == "AUTO_OFF") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        if (controlState_ == ControlState::AUTO) {
            setState(ControlState::UNLOCKED);
            logCommand(CommandType::EXIT_AUTO_MODE, {});
        }
        return true;
    } else if (commandStr == "UNLOCK") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        if (controlState_ == ControlState::LOCKED) {
            setState(ControlState::UNLOCKED);
            logger_->addTempMessage("System UNLOCKED", LogLevel::INFO);
        }
        return true;
    } else if (commandStr == "LOCK") {
        std::lock_guard<std::mutex> lock(commandMutex_);
        if (controlState_ != ControlState::LOCKED) {
            setState(ControlState::LOCKED);
            logger_->addTempMessage("System LOCKED", LogLevel::INFO);
        }
        return true;
    }

    return false;
}

void Receiver::setState(ControlState newState) {
    controlState_ = newState;
    logger_->setValue("control_state", controlStateToString(newState));
}

void Receiver::logCommand(CommandType type, const std::vector<float>& params) {
    switch (type) {
        case CommandType::TAKEOFF:
            logger_->setValue("cmd_takeoff", "Initiated");
            break;
            
        case CommandType::LAND:
            logger_->setValue("cmd_land", "Initiated");
            break;
            
        case CommandType::JOYSTICK_CONTROL: {
            if (params.size() >= 4) {
                std::stringstream ss;
                ss << "roll=" << params[0] << ", pitch=" << params[1] 
                   << ", yaw_rate=" << params[2] << ", height_delta=" << params[3];
                logger_->setValue("cmd_joystick", ss.str());
            }
            break;
        }
        
        default:
            break;
    }
}
