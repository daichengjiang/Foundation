// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "CrazyflieLink.h"
#include "Logging.h"
#include "Commander.h"
#include <iostream>
#include <iomanip>

namespace crazyflie
{

    CrazyflieLink::CrazyflieLink(const std::string &uri, int baudRate)
        : uri_(uri),                                  // Store URI
          baudRate_(baudRate),                        // Store BaudRate
          impl_(std::make_shared<ConnectionImpl>()),
          thread_ending_(false),
          stop_worker_threads_(false),                // Initialize new flag
          drone_ping_received_{false},                // Initialize atomic
          connection_established_{false},             // Initialize atomic
          needs_reconnect_{false},                    // Initialize new flag
          last_packet_received_time_(std::chrono::steady_clock::now()) // Initialize timestamp
    {
        if (!establish_connection())
        {
            std::cerr << "Initial connection failed. Exiting." << std::endl;
            // Maintaining original behavior: exit on initial failure.
            // If reconnection for initial failure is desired, set needs_reconnect_ = true; here
            // and remove exit().
            exit(EXIT_FAILURE);
        }
        // If establish_connection is successful, proceed.

        // Start reconnection manager thread
        reconnection_thread_ = std::thread(&CrazyflieLink::reconnection_manager_run, this);
    }

    CrazyflieLink::~CrazyflieLink()
    {
        thread_ending_ = true;      // Signal ALL threads to stop permanently
        stop_worker_threads_ = true; // Ensure worker threads check this for shutdown
        needs_reconnect_ = false;   // Prevent reconnection manager from further attempts

        if (reconnection_thread_.joinable())
        {
            reconnection_thread_.join();
        }

        if (send_thread_.joinable())
        {
            send_thread_.join();
        }

        if (recv_thread_.joinable())
        {
            recv_thread_.join();
        }

        close();
    }

    void CrazyflieLink::close()
    {
        if (serial_.isOpen())
        {
            serial_.close();
        }
    }

    std::shared_ptr<LoggingBlock> CrazyflieLink::createLoggingBlock()
    {
        return std::make_shared<LoggingBlock>(shared_from_this());
    }

    std::shared_ptr<Commander> CrazyflieLink::getCommander()
    {
        return std::make_shared<Commander>(shared_from_this());
    }

    void CrazyflieLink::registerCallback(uint8_t blockId, DataCallback callback)
    {
        const std::lock_guard<std::mutex> lock(callbacks_mutex_);
        callbacks_[blockId] = callback;
    }

    void CrazyflieLink::sendPacket(const Packet &packet, const uint8_t CPX_function)
    {
        const std::lock_guard<std::mutex> lock(impl_->queue_send_mutex_);
        if (!impl_->runtime_error_.empty())
        {
            throw std::runtime_error(impl_->runtime_error_);
        }

        Packet p = packet;
        p.seq_ = impl_->statistics_.enqueued_count;

        // Encode with CPX protocol
        std::vector<uint8_t> buffer = encodeCPXPacket(p, CPX_TARGET_HOST, CPX_TARGET_STM32, CPX_function, 0);
        impl_->queue_send_.push(buffer);
        ++impl_->statistics_.enqueued_count;
    }

    std::vector<uint8_t> CrazyflieLink::encodeCPXPacket(const Packet &packet, const uint8_t source,
                                                        const uint8_t destination, const uint8_t function, const uint8_t version)
    {

        std::vector<uint8_t> buffer;
        buffer.reserve(packet.size() + 5); // Start byte 1 + size byte 1 + CPX header 2 + data + checksum 1

        // Add start byte and size (required lines)
        buffer.push_back(UART_START);
        buffer.push_back(static_cast<uint8_t>(packet.size()) + 2); // +2 for header

        // Create CPX header bytes similar to Python version
        uint8_t targetsAndFlags = ((source & 0x7) << 3) | (destination & 0x7);
        bool lastPacket = false;
        if (lastPacket)
        {
            targetsAndFlags |= 0x40;
        }

        uint8_t functionAndVersion = (function & 0x3F) | ((version & 0x3) << 6);

        // Add header bytes
        buffer.push_back(targetsAndFlags);
        buffer.push_back(functionAndVersion);

        // Add payload data
        buffer.insert(buffer.end(), packet.raw(), packet.raw() + packet.size());

        // Calculate and add XOR checksum
        buffer.push_back(calculateXORChecksum(buffer));

        return buffer;
    }

    uint8_t CrazyflieLink::calculateXORChecksum(const std::vector<uint8_t> &data)
    {
        uint8_t checksum = 0;
        for (uint8_t byte : data)
        {
            checksum ^= byte;
        }
        return checksum;
    }

    void CrazyflieLink::send_run()
    {
        try
        {
            while (!thread_ending_.load() && !stop_worker_threads_.load())
            {
                if (!impl_->queue_send_.empty())
                {
                    std::vector<uint8_t> buffer;
                    { // Scope for lock
                        const std::lock_guard<std::mutex> lock(impl_->queue_send_mutex_);
                        // Double check if queue is still not empty after acquiring lock
                        if (impl_->queue_send_.empty()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            continue;
                        }
                        buffer = impl_->queue_send_.top(); // top() doesn't remove
                    } // Lock released

                    bool success = serial_.write(buffer.data(), buffer.size());
                    if (success)
                    {
                        const std::lock_guard<std::mutex> lock(impl_->queue_send_mutex_);
                        // Verify that the packet at the top is still the one we sent before popping
                        if (!impl_->queue_send_.empty() && impl_->queue_send_.top().size() == buffer.size() &&
                            std::equal(impl_->queue_send_.top().begin(), impl_->queue_send_.top().end(), buffer.begin())) {
                            impl_->queue_send_.pop();
                        }
                        impl_->statistics_.sent_count++;
                    }
                    else
                    {
                        // If not shutting down, signal reconnect
                        if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                            std::cerr << "Failed to send packet! Attempting reconnect." << std::endl;
                            if(impl_) impl_->runtime_error_ = "Failed to write to serial port";
                            connection_established_ = false;
                            needs_reconnect_ = true;
                            break; // Exit send loop, thread will be joined and restarted
                        }
                    }
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        catch (const serial::IOException& e)
        {
            if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Serial IOException in send_run: " << e.what() << ". Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = e.what();
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        catch (const std::exception &error)
        {
             if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Standard exception in send_run: " << error.what() << ". Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = error.what();
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        catch (...)
        {
            if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Unknown error in send_run. Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = "Unknown error in send thread";
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        // std::cout << "send_run thread finishing." << std::endl;
    }

    void CrazyflieLink::recv_run()
    {
        try
        {
            enum State
            {
                WAIT_START,
                READ_SIZE,
                READ_DATA,
                READ_CHECKSUM
            };
            State state = WAIT_START;
            uint8_t size = 0;
            std::vector<uint8_t> packet_data; // Renamed from 'packet' to avoid conflict

            while (!thread_ending_.load() && !stop_worker_threads_.load())
            {
                size_t available_bytes = 0;
                try {
                    available_bytes = serial_.available();
                } catch (const serial::SerialException& e) {
                     if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                        std::cerr << "SerialException while checking available bytes in recv_run: " << e.what() << ". Attempting reconnect." << std::endl;
                        if(impl_) impl_->runtime_error_ = e.what();
                        connection_established_ = false;
                        needs_reconnect_ = true;
                        break; // Exit recv loop
                    }
                }


                if (available_bytes > 0)
                {
                    std::vector<uint8_t> tempBuffer(available_bytes);
                    size_t bytes_read = 0;
                    try {
                        bytes_read = serial_.read(tempBuffer.data(), tempBuffer.size());
                    } catch (const serial::SerialException& e) {
                        if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                            std::cerr << "SerialException during read in recv_run: " << e.what() << ". Attempting reconnect." << std::endl;
                            if(impl_) impl_->runtime_error_ = e.what();
                            connection_established_ = false;
                            needs_reconnect_ = true;
                            break; // Exit recv loop
                        }
                    }
                    tempBuffer.resize(bytes_read); // Adjust buffer to actual bytes read

                    for (uint8_t byte : tempBuffer)
                    {
                        switch (state)
                        {
                        case WAIT_START:
                            if (byte == UART_START)
                            {
                                state = READ_SIZE;
                            }
                            break;

                        case READ_SIZE:
                            size = byte;
                            if (size > 50) // Max packet size check
                            {
                                std::cerr << "Invalid size byte: " << (int)size << std::endl;
                                state = WAIT_START;
                            }
                            else if (size == 0) // UART_RESET or sync packet
                            {
                                sendAck(); // Should this be conditional on connection state?
                                drone_ping_received_ = true; // This indicates a form of communication
                                last_packet_received_time_ = std::chrono::steady_clock::now(); // Update timestamp
                                state = WAIT_START;
                            }
                            else
                            {
                                packet_data.clear();
                                packet_data.reserve(size);
                                state = READ_DATA;
                            }
                            break;

                        case READ_DATA:
                            packet_data.push_back(byte);
                            if (packet_data.size() == size)
                            {
                                state = READ_CHECKSUM;
                            }
                            break;

                        case READ_CHECKSUM:
                        {
                            std::vector<uint8_t> buffer_checksum;
                            buffer_checksum.push_back(UART_START);
                            buffer_checksum.push_back(size);
                            buffer_checksum.insert(buffer_checksum.end(), packet_data.begin(), packet_data.end());

                            uint8_t calculated_checksum = calculateXORChecksum(buffer_checksum);
                            if (calculated_checksum == byte)
                            {
                                last_packet_received_time_ = std::chrono::steady_clock::now(); // Update timestamp
                                if (packet_data.size() >= 2) { // CPX header (2 bytes) must be present
                                    std::vector<uint8_t> processedPacket(packet_data.begin() + 2, packet_data.end());
                                    processPacket(processedPacket);
                                } else {
                                    std::cerr << "Received packet too short for CPX header." << std::endl;
                                }


                                if (drone_ping_received_.load()) // Only send ACK if drone is considered responsive
                                    sendAck();
                            }
                            else
                            {
                                std::cerr << "Checksum error!" << std::endl;
                            }
                            state = WAIT_START;
                        }
                        break;
                        }
                    }
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        catch (const serial::IOException& e)
        {
            if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Serial IOException in recv_run: " << e.what() << ". Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = e.what();
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        catch (const std::exception &error)
        {
            if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Standard exception in recv_run: " << error.what() << ". Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = error.what();
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        catch (...)
        {
            if (!thread_ending_.load() && !stop_worker_threads_.load()) {
                std::cerr << "Unknown error in recv_run. Attempting reconnect." << std::endl;
                if(impl_) impl_->runtime_error_ = "Unknown error in receive thread";
                connection_established_ = false;
                needs_reconnect_ = true;
            }
        }
        // std::cout << "recv_run thread finishing." << std::endl;
    }

    void CrazyflieLink::processPacket(const std::vector<uint8_t> &packet)
    {
        // Extract port and channel
        uint8_t header = packet[0];
        uint8_t port = (header >> 4) & 0x0F;
        uint8_t channel = header & 0x03;

        // Call registered callback if exists
        if (port == LOGGING && channel == CHAN_LOGDATA)
        {
            uint8_t key = packet[1];
            const std::lock_guard<std::mutex> lock(callbacks_mutex_);
            auto it = callbacks_.find(key);
            if (it != callbacks_.end() && it->second)
            {
                it->second(packet);
            }
            else
            {
                std::cerr << "No callback registered for block ID: " << (int)key << std::endl;
            }
        }
        else if (port == CONSOLE && channel == CHAN_TOC)
        {
            connection_established_ = true;
            for (auto c : packet)
                printf("%c", c);
        }
        else
        {
            std::cout << "Unimplemented packet type: port " << (int)port << ", channel " << (int)channel << std::endl;
        }
    }

    void CrazyflieLink::sendAck()
    {
        // Send reset/sync packet
        uint8_t ackData[] = {UART_START, UART_RESET};
        serial_.write(ackData, 2);
        // std::cout << "Sent ACK packet" << std::endl;
    }

    bool CrazyflieLink::establish_connection()
    {
        // Reset critical flags for a fresh attempt
        drone_ping_received_ = false;
        connection_established_ = false;
        if(impl_) impl_->runtime_error_.clear(); // Clear previous runtime errors
        last_packet_received_time_ = std::chrono::steady_clock::now(); // Reset timestamp at start of attempt

        try
        {
            if (serial_.isOpen()) { // Close if already open from a previous failed attempt
                serial_.close();
            }
            serial_.setPort(uri_);
            serial_.setBaudrate(static_cast<uint32_t>(baudRate_));
            std::cout << "Trying to open serial port with " << uri_ << ", " << baudRate_ << std::endl;

            auto timeout = serial::Timeout::simpleTimeout(10); // Original timeout
            serial_.setTimeout(timeout);
            serial_.open();

            if (serial_.isOpen())
            {
                std::cout << "Serial port opened successfully." << std::endl;
                if(impl_) impl_->uri_ = uri_;
            }
            else
            {
                std::cerr << "Failed to open serial port." << std::endl;
                return false;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception while opening serial port: " << e.what() << std::endl;
            if(serial_.isOpen()) serial_.close(); // Ensure closed on exception
            return false;
        }

        // stop_worker_threads_ should be false here, ensured by attempt_reconnect or constructor
        send_thread_ = std::thread(&CrazyflieLink::send_run, this);
        recv_thread_ = std::thread(&CrazyflieLink::recv_run, this);

        // Send an initial packet to elicit a response / drone ping
        // std::cout << "Sending initial packet to Crazyflie to prompt for ping..." << std::endl;
        Packet initialCpxSwitchPacket;
        initialCpxSwitchPacket.addByte(0x01);
        initialCpxSwitchPacket.raw()[0] = 0x21; 
        sendPacket(initialCpxSwitchPacket, CPX_FUNCTION_SYSTEM);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        int count = 0;
        // std::cout << "Waiting for drone ping with 30 s timeout." << std::endl;
        while (!drone_ping_received_.load() && !thread_ending_.load() && count++ < 3000) // Check thread_ending_ for global shutdown
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        if (thread_ending_.load()) { 
            std::cout << "Connection attempt aborted during ping wait (global shutdown)." << std::endl;
            stop_worker_threads_ = true; 
            if(send_thread_.joinable()) send_thread_.join();
            if(recv_thread_.joinable()) recv_thread_.join();
            stop_worker_threads_ = false;
            if(serial_.isOpen()) serial_.close();
            return false;
        }
        if (!drone_ping_received_.load())
        {
            std::cerr << "Failed to receive drone ping." << std::endl;
            stop_worker_threads_ = true; // Signal newly started threads to stop
            if(send_thread_.joinable()) send_thread_.join();
            if(recv_thread_.joinable()) recv_thread_.join();
            stop_worker_threads_ = false; // Reset for future attempts
            if(serial_.isOpen()) serial_.close();
            return false;
        }
        // std::cout << "Drone ping received." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        Packet switchToCpxPacket;
        switchToCpxPacket.addByte(0x01);
        switchToCpxPacket.raw()[0] = 0x21;
        sendPacket(switchToCpxPacket, CPX_FUNCTION_SYSTEM);
        switchToCpxPacket.raw()[0] = 0x20;
        sendPacket(switchToCpxPacket, CPX_FUNCTION_SYSTEM);

        Packet forceConnectPacket;
        forceConnectPacket.addByte(0x01); 
        forceConnectPacket.raw()[0] = 0x20; 
        sendPacket(forceConnectPacket, CPX_FUNCTION_SYSTEM);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        count = 0;
        // std::cout << "Waiting for connection establishment with 10 s timeout." << std::endl;
        while (!connection_established_.load() && !thread_ending_.load() && count++ < 1000) // Check thread_ending_
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (thread_ending_.load()) { 
            std::cout << "Connection attempt aborted during connection wait (global shutdown)." << std::endl;
            stop_worker_threads_ = true;
            if(send_thread_.joinable()) send_thread_.join();
            if(recv_thread_.joinable()) recv_thread_.join();
            stop_worker_threads_ = false;
            if(serial_.isOpen()) serial_.close();
            return false;
        }
        if (!connection_established_.load()) {
            std::cerr << "Failed to establish connection after ping." << std::endl;
            stop_worker_threads_ = true; // Signal newly started threads to stop
            if(send_thread_.joinable()) send_thread_.join();
            if(recv_thread_.joinable()) recv_thread_.join();
            stop_worker_threads_ = false; // Reset
            if(serial_.isOpen()) serial_.close();
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "Connection established." << std::endl;
        last_packet_received_time_ = std::chrono::steady_clock::now(); // Set timestamp on successful connection
        return true;
    }

    void CrazyflieLink::reconnection_manager_run()
    {
        // std::cout << "Reconnection manager thread started." << std::endl;
        while (!thread_ending_.load())
        {
            // Check for data timeout if we think we are connected
            if (connection_established_.load() && !needs_reconnect_.load()) // Check needs_reconnect_ to avoid redundant checks if already flagged
            {
                auto now = std::chrono::steady_clock::now();
                if (now - last_packet_received_time_.load() > MAX_SILENCE_DURATION)
                {
                    std::cerr << "No packet received for " << MAX_SILENCE_DURATION.count() 
                              << "s. Assuming disconnection due to timeout." << std::endl;
                    if(impl_) impl_->runtime_error_ = "Connection timed out (no received packets).";
                    connection_established_ = false;
                    needs_reconnect_ = true; // Signal for reconnection
                }
            }

            if (needs_reconnect_.load())
            {
                // std::cout << "Reconnection needed. Attempting..." << std::endl;
                attempt_reconnect(); 
                // attempt_reconnect will reset needs_reconnect_ on success.
                // If it fails, needs_reconnect_ remains true, and we'll retry after sleep.
            }
            // Sleep to avoid busy-waiting, and also after a failed attempt before retrying.
            std::this_thread::sleep_for(std::chrono::seconds(2)); // Retry every 2 seconds
        }
        // std::cout << "Reconnection manager thread exiting." << std::endl;
    }

    void CrazyflieLink::attempt_reconnect()
    {
        std::lock_guard<std::mutex> lock(reconnect_mutex_); // Ensure only one attempt at a time

        // Double check if reconnection is still needed and not already connected by another means
        if (!needs_reconnect_.load() || connection_established_.load()) {
            if (connection_established_.load()) needs_reconnect_ = false; // Clear if already connected
            return;
        }

        std::cout << "Attempting to re-establish connection..." << std::endl;

        // 1. Signal current worker threads to stop
        stop_worker_threads_ = true;

        // 2. Join existing threads
        if (send_thread_.joinable()) {
            send_thread_.join();
            // std::cout << "Old send_thread joined for reconnect." << std::endl;
        }
        if (recv_thread_.joinable()) {
            recv_thread_.join();
            // std::cout << "Old recv_thread joined for reconnect." << std::endl;
        }

        // Reset flag for new threads that will be started by establish_connection
        stop_worker_threads_ = false;

        // Clear send queue and reset statistics for a cleaner state before attempting to re-establish.
        if (impl_) {
            const std::lock_guard<std::mutex> lock(impl_->queue_send_mutex_); // Protect queue and statistics access
            // Clear the send queue. Assuming it's stack-like (supports empty(), top(), pop()).
            while(!impl_->queue_send_.empty()) {
                impl_->queue_send_.pop();
            }
            impl_->statistics_.enqueued_count = 0;
            impl_->statistics_.sent_count = 0;
            // impl_->runtime_error_ is cleared at the start of establish_connection()
        }

        // 3. Close serial port (ensure it's closed before trying to reopen)
        if (serial_.isOpen()) {
            serial_.close();
            // std::cout << "Serial port closed for reconnection." << std::endl;
        }
        
        // 4. Attempt to re-establish the connection
        if (establish_connection()) { // This method now handles its own thread creation and full setup
            std::cout << "Reconnection successful." << std::endl;
            needs_reconnect_ = false; // Clear the flag as connection is re-established
        } else {
            std::cout << "Reconnection attempt failed. Will retry." << std::endl;
            // ensure serial is closed if establish_connection failed partway and left it open
            if (serial_.isOpen()) {
                serial_.close();
            }
            // needs_reconnect_ remains true, so reconnection_manager_run will try again.
        }
    }

} // namespace crazyflie