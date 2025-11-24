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

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <map>
#include <string>
#include <chrono>
#include "logger.h"

class CallbackRateStatistics {
    public:
        // Constructor
        CallbackRateStatistics(bool enable = true, Logger* pLogger = nullptr)
            : m_logger(pLogger)
        {
            // Start a thread that prints stats every second
            if (enable)
                m_statsThread = std::thread(&CallbackRateStatistics::printStatsLoop, this);
        }
        
        // Destructor
        ~CallbackRateStatistics() {
            m_running = false;
            if (m_statsThread.joinable()) {
                m_statsThread.join();
            }
        }
        
        // Increment the counter for a specific callback
        void incrementCallbackCount(const std::string& name) {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Auto-register if not found
            if (m_callbackCounts.find(name) == m_callbackCounts.end()) {
                m_callbackCounts[name] = 0;
                m_callbackRates[name] = 0.0;
                m_logger->addVariable(name, "Rate: {}", LogLevel::INFO);
            }
            m_callbackCounts[name]++;
        }

        void setLogger(Logger* pLogger) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_logger = pLogger;
        }
        
    private:
        // Thread function to calculate and print stats
        void printStatsLoop() {
            while (m_running) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    
                    // Calculate rates
                    for (auto& pair : m_callbackCounts) {
                        const std::string& name = pair.first;
                        int count = pair.second;
                        
                        // Rate is count per second
                        m_callbackRates[name] = count;

                        m_logger->setValue(name, m_callbackRates[name]);

                        // Reset the counter
                        m_callbackCounts[name] = 0;
                    }
                }
            }
        }
        
        std::thread m_statsThread;
        std::mutex m_mutex;
        std::map<std::string, int> m_callbackCounts;
        std::map<std::string, double> m_callbackRates;
        bool m_running = true;
        Logger* m_logger = nullptr; // New logger pointer
    };
    
    // Global instance to be used by all callbacks
    CallbackRateStatistics g_callbackStats;