// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstring>
#include "serial/serial.h"

bool g_running = true;

// Signal handler for clean shutdown
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received. Exiting..." << std::endl;
    g_running = false;
}

// Function to read data from the serial port
void readData(serial::Serial& ser) {
    uint8_t buffer[256];
    size_t bytes_read;

    while (g_running) {
        try {
            // Read data if available
            auto available_bytes = ser.available();
            if (available_bytes) {
                bytes_read = ser.read(buffer, available_bytes);
                if (bytes_read > 0) {
                    std::cout << "Received: ";
                    for (size_t i = 0; i < bytes_read; ++i) {
                        std::cout << std::hex << std::setw(2) << std::setfill('0')
                                  << static_cast<int>(buffer[i]) << " ";
                    }
                    std::cout << std::dec << std::endl;
                }
            }
            // Small delay to prevent CPU overload
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } catch (const std::exception& e) {
            std::cerr << "Error during serial communication: " << e.what() << std::endl;
            break;
        }
    }
}

// Function to display usage information
void displayUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -p, --port PORT      Serial port to connect to (default: /dev/ttyS1)" << std::endl;
    std::cout << "  -b, --baudrate RATE  Baud rate for the serial connection (default: 921600)" << std::endl;
    std::cout << "  -h, --help           Display this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string port = "/dev/ttyS1";
    uint32_t baudrate = 460800;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) {
            if (i + 1 < argc) {
                port = argv[++i];
            } else {
                std::cerr << "Error: Port argument missing" << std::endl;
                displayUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--baudrate") == 0) {
            if (i + 1 < argc) {
                baudrate = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: Baudrate argument missing" << std::endl;
                displayUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            displayUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            displayUsage(argv[0]);
            return 1;
        }
    }

    // Register signal handler
    signal(SIGINT, signalHandler);

    try {
        // Open the serial port
        std::cout << "Opening port " << port << " at " << baudrate << " baud with 8 data bits and 1 stop bit." << std::endl;
        
        // Configure serial port with 8 data bits and 1 stop bit
        serial::Serial ser(port, 
                         baudrate, 
                         serial::Timeout::simpleTimeout(0),
                         serial::eightbits,    // 8 data bits
                         serial::parity_none,  // No parity
                         serial::stopbits_one, // 1 stop bit
                         serial::flowcontrol_none); // No flow control
        
        if (!ser.isOpen()) {
            std::cerr << "Failed to open serial port." << std::endl;
            return 1;
        }
        
        std::cout << "Connected to " << port << " at " << baudrate << " baud." << std::endl;
        ser.flushInput();
        
        // Create a thread to read data
        std::thread reader_thread(readData, std::ref(ser));
        
        // Wait for thread to finish (will happen when signal is received)
        reader_thread.join();
        
        // Close the serial port
        ser.close();
        
    } catch (const serial::IOException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
