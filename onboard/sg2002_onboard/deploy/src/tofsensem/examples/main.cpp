// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include "tofImpl.h"
#include "init_serial.h"
#include "nlink_unpack/nlink_tofsensem_frame0.h"
#include "nlink_unpack/nlink_utils.h"
#include "protocol_extracter/nprotocol_extracter.h"

const int SERVER_PORT = 8080;
std::vector<int> client_sockets;
std::mutex client_mutex;
std::atomic<bool> running{true};

// Helper function to send data to a client
bool send_to_client(int client_socket, const std::string& data) {
    int bytes_sent = send(client_socket, data.c_str(), data.size(), 0);
    return bytes_sent == static_cast<int>(data.size());
}

// Function to accept client connections
void accept_clients(int server_socket) {
    while (running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            std::cerr << "Accept failed" << std::endl;
            continue;
        }
        
        std::cout << "New client connected: " << inet_ntoa(client_addr.sin_addr) 
                  << ":" << ntohs(client_addr.sin_port) << std::endl;
        
        std::lock_guard<std::mutex> lock(client_mutex);
        client_sockets.push_back(client_socket);
    }
}

void callback_fun(ntsm_frame0_t frame)
{
    // Log data to console
    std::cout << std::endl << "[TofsenseMFrame0] id: " << frame.id
              << ", system_time: " << frame.system_time << "\n";
    std::cout << "dis:\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;
            std::cout << (int)(frame.pixels[idx].dis) / 1000.0 << (j < 7 ? " " : "");
        }
        std::cout << "\n";
    }

    std::cout << "signal_strength:\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;
            std::cout << frame.pixels[idx].signal_strength << (j < 7 ? " " : "");
        }
        std::cout << "\n";
    }
    
    // Prepare data to send to clients
    std::string data = "FRAME " + std::to_string(frame.id) + " " + 
                      std::to_string(frame.system_time) + "\n";
    
    // Add distance data
    data += "DISTANCE\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;
            data += std::to_string((int)(frame.pixels[idx].dis) / 1000.0) + " ";
        }
        data += "\n";
    }
    
    // Add signal strength data
    data += "SIGNAL\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = i * 8 + j;
            data += std::to_string(frame.pixels[idx].signal_strength) + " ";
        }
        data += "\n";
    }
    data += "END\n";
    
    // Send data to all clients
    std::lock_guard<std::mutex> lock(client_mutex);
    auto it = client_sockets.begin();
    while (it != client_sockets.end()) {
        if (!send_to_client(*it, data)) {
            std::cout << "Client disconnected" << std::endl;
            close(*it);
            it = client_sockets.erase(it);
        } else {
            ++it;
        }
    }
}

int main() {
    // Initialize socket server
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }
    
    // Set socket options to allow reuse of the address
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Failed to set socket options" << std::endl;
        close(server_socket);
        return 1;
    }
    
    // Bind the socket to local address and port
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to bind socket" << std::endl;
        close(server_socket);
        return 1;
    }
    
    // Listen for incoming connections
    if (listen(server_socket, 5) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        close(server_socket);
        return 1;
    }
    
    std::cout << "Server started, listening on port " << SERVER_PORT << std::endl;
    
    // Start thread to accept client connections
    std::thread accept_thread(accept_clients, server_socket);
    
    tofsensem::Impl impl(callback_fun, "/dev/ttyUSB0", 921600);

    // Main loop for processing sensor data
    // while (running) {
    //     auto available_bytes = serial.available();
    //     if (available_bytes) {
    //         std::string str_received;
    //         serial.read(str_received, available_bytes);
    //         extracter.AddNewData(str_received);
    //     } else {
    //         std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //     }
    //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // }
    
    // Cleanup
    running = false;
    accept_thread.join();
    
    // Close all client sockets
    std::lock_guard<std::mutex> lock(client_mutex);
    for (int client_socket : client_sockets) {
        close(client_socket);
    }
    client_sockets.clear();
    
    // Close server socket
    close(server_socket);
    
    return 0;
}
