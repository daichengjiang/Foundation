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
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>

bool g_running = true;

// Signal handler for clean shutdown
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received. Exiting..." << std::endl;
    g_running = false;
}

// Helper function to configure serial port
bool configureSerialPort(int fd, int baudrate) {
    struct termios tty;
    
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error from tcgetattr: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Clear parity bit, disabling parity
    tty.c_cflag &= ~PARENB;
    // Clear stop field, only one stop bit used
    tty.c_cflag &= ~CSTOPB;
    // Clear all bits that set the data size
    tty.c_cflag &= ~CSIZE;
    // 8 bits per byte
    tty.c_cflag |= CS8;
    // Disable RTS/CTS hardware flow control
    tty.c_cflag &= ~CRTSCTS;
    // Turn on READ & ignore ctrl lines (CLOCAL = 1)
    tty.c_cflag |= CREAD | CLOCAL;
    
    // Disable canonical mode
    tty.c_lflag &= ~ICANON;
    // Disable echo
    tty.c_lflag &= ~ECHO;
    // Disable erasure
    tty.c_lflag &= ~ECHOE;
    // Disable new-line echo
    tty.c_lflag &= ~ECHONL;
    // Disable interpretation of INTR, QUIT and SUSP
    tty.c_lflag &= ~ISIG;
    
    // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    // Disable any special handling of received bytes
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
    
    // Prevent special interpretation of output bytes
    tty.c_oflag &= ~OPOST;
    // Prevent conversion of newline to carriage return/line feed
    tty.c_oflag &= ~ONLCR;
    
    // Configure non-blocking read
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    
    // Set baud rate
    speed_t speed;
    switch (baudrate) {
        case 9600: speed = B9600; break;
        case 19200: speed = B19200; break;
        case 38400: speed = B38400; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        case 460800: speed = B460800; break;
        case 921600: speed = B921600; break;
        default:
            std::cerr << "Unsupported baud rate: " << baudrate << std::endl;
            return false;
    }
    
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
    
    // Save tty settings
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error from tcsetattr: " << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

// Function to read data from the serial port
void readData(int fd) {
    uint8_t buffer[256];
    ssize_t bytes_read;

    while (g_running) {
        try {
            // Check if data is available
            int bytes_available;
            ioctl(fd, FIONREAD, &bytes_available);
            
            if (bytes_available > 0) {
                // Read data if available
                bytes_read = read(fd, buffer, bytes_available);
                if (bytes_read > 0) {
                    std::cout << "Received: ";
                    for (ssize_t i = 0; i < bytes_read; ++i) {
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
        
        int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            std::cerr << "Failed to open serial port: " << strerror(errno) << std::endl;
            return 1;
        }
        
        // Configure serial port
        if (!configureSerialPort(fd, baudrate)) {
            close(fd);
            return 1;
        }
        
        std::cout << "Connected to " << port << " at " << baudrate << " baud." << std::endl;
        
        // Flush any pending input
        tcflush(fd, TCIFLUSH);
        
        // Create a thread to read data
        std::thread reader_thread(readData, fd);
        
        // Wait for thread to finish (will happen when signal is received)
        reader_thread.join();
        
        // Close the serial port
        close(fd);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
