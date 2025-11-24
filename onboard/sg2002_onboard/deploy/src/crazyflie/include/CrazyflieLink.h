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
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include <serial/serial.h>
#include "Packet.h"
#include "ConnectionImpl.h"
#include <mutex>
#include <chrono>

namespace crazyflie {

// CPX Protocol constants
constexpr uint8_t UART_START = 0xFF;
constexpr uint8_t UART_RESET = 0x00;

// CPX Targets
#define CPX_TARGET_STM32 1
#define CPX_TARGET_ESP32 2
#define CPX_TARGET_HOST 3
#define CPX_TARGET_GAP8 4

// CPX Functions
#define CPX_FUNCTION_SYSTEM 1
#define CPX_FUNCTION_CONSOLE 2
#define CPX_FUNCTION_CRTP 3
#define CPX_FUNCTION_WIFI_CTRL 4
#define CPX_FUNCTION_APP 5
#define CPX_FUNCTION_TEST 0x0E
#define CPX_FUNCTION_BOOTLOADER 0x0F

// Timeout for considering the link dead if no packets are received
const std::chrono::seconds MAX_SILENCE_DURATION(1);

class LoggingBlock;
class Commander;

class CrazyflieLink : public std::enable_shared_from_this<CrazyflieLink> {
public:
    using DataCallback = std::function<void(const std::vector<uint8_t>&)>;
    
    CrazyflieLink(const std::string& uri, int baudRate = 115200);
    ~CrazyflieLink();
    
    // Close connection
    void close();
    
    // Get logging interface
    std::shared_ptr<LoggingBlock> createLoggingBlock();
    
    // Get commander interface
    std::shared_ptr<Commander> getCommander();
    
    // Send a packet
    void sendPacket(const Packet& packet, const uint8_t CPX_function = CPX_FUNCTION_CRTP);
    
    // Register callback for specific port and channel
    void registerCallback(uint8_t blockId, DataCallback callback);

    // Get the current connection status
    bool isConnected() const { return connection_established_; }
    
private:
    // Serial port
    serial::Serial serial_;
    std::string uri_;    // Store URI for reconnection
    int baudRate_;       // Store BaudRate for reconnection
    
    // Thread management
    std::atomic<bool> thread_ending_; // For global shutdown
    std::atomic<bool> stop_worker_threads_{false}; // For temporary stop during reconnect
    std::thread send_thread_;
    std::thread recv_thread_;
    std::thread reconnection_thread_; // For managing reconnection attempts
    
    // Implementation details
    std::shared_ptr<ConnectionImpl> impl_;

    // flags
    std::atomic<bool> drone_ping_received_{false}; // Made atomic
    std::atomic<bool> connection_established_{false}; // Made atomic
    std::atomic<bool> needs_reconnect_{false}; // Flag to indicate reconnection is needed
    std::atomic<std::chrono::steady_clock::time_point> last_packet_received_time_; // Timestamp of the last received packet
    std::mutex reconnect_mutex_; // To ensure only one reconnect attempt at a time
    
    // Thread functions
    void send_run();
    void recv_run();
    void reconnection_manager_run(); // New reconnection manager
    
    // Connection establishment and retry logic
    bool establish_connection();    // New method for connection logic
    void attempt_reconnect();       // New method to attempt reconnection

    // Callback map
    std::map<uint8_t, DataCallback> callbacks_;
    std::mutex callbacks_mutex_;
    
    // Process complete packet
    void processPacket(const std::vector<uint8_t>& packet);
    
    // Send ACK for received data
    void sendAck();
    
    // CPX protocol helpers
    std::vector<uint8_t> encodeCPXPacket(const Packet& packet, const uint8_t source, 
            const uint8_t destination, const uint8_t function, const uint8_t version);
    uint8_t calculateXORChecksum(const std::vector<uint8_t>& data);
};

} // namespace crazyflie