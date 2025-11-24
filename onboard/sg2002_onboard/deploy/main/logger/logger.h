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
#include <sstream>
#include <atomic>
#include <functional>
#include <vector>
#include <list>

// Log level enum
enum class LogLevel {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  NONE // For filtering
};

// Forward declaration
class Logger;

// Custom stream buffer to redirect stdout/stderr to logger
class LoggerStreamBuf : public std::streambuf {
public:
    LoggerStreamBuf(Logger* logger, LogLevel level);
    virtual ~LoggerStreamBuf();

protected:
    virtual int_type overflow(int_type c) override;
    virtual int sync() override;

private:
    Logger* logger_;
    LogLevel level_;
    std::string buffer_;
};

class Logger {
public:
    // Constructor/Destructor
    Logger(int printFrequencyMs = 1000, LogLevel minLogLevel = LogLevel::INFO);
    ~Logger();
    
    // Add a variable to be logged with format string
    void addVariable(const std::string& name, const std::string& format = "{}", LogLevel level = LogLevel::INFO);
    
    // Remove a variable from logging
    void removeVariable(const std::string& name);
    
    // Set the value of a variable (scalar)
    template<typename T>
    void setValue(const std::string& name, const T& value);
    
    // Set the value of a variable (vector)
    template<typename T>
    void setValue(const std::string& name, const std::vector<T>& values);
    
    // Add a temporary message with duration
    void addTempMessage(const std::string& message, LogLevel level = LogLevel::INFO, int durationMs = 1000);
    
    // Set the minimum log level to display
    void setMinLogLevel(LogLevel level);
    
    // Set the print frequency
    void setPrintFrequency(int frequencyMs);
    
    // Start/stop logging
    void start();
    void stop();
    
    // Start/stop redirection of stdout and stderr
    void startRedirection();
    void stopRedirection();
    
private:
    // Internal data structure to store variable info
    struct VariableInfo {
        std::string format;      // Format string with {} placeholder
        std::string value;       // Formatted value as string
        LogLevel level;          // Log level
    };
    
    // Internal structure to store temporary messages
    struct TempMessage {
        std::string message;                 // Message text
        LogLevel level;                      // Log level
        std::chrono::steady_clock::time_point expireTime; // Expiration time
    };
    
    // Replace placeholder {} with actual value
    template<typename T>
    std::string formatString(const std::string& format, const T& value);
    
    // Format vector values
    template<typename T>
    std::string formatVector(const std::string& format, const std::vector<T>& values);
    
    // Print thread function
    void printLoop();
    
    // Helper to format log level
    std::string logLevelToString(LogLevel level);
    
    // Helper to colorize based on level (ANSI escape codes)
    std::string colorizeLevel(LogLevel level, const std::string& text);
    
    // Map to store variables and their information
    std::map<std::string, VariableInfo> variables_;
    
    // List to store temporary messages
    std::list<TempMessage> tempMessages_;
    
    // Thread control
    std::thread printThread_;
    std::mutex mutex_;
    std::atomic<bool> running_;
    
    // Configuration
    int printFrequencyMs_;
    LogLevel minLogLevel_;
    bool useColors_;
    
    // Stream redirection
    std::streambuf* originalCoutBuf_;
    std::streambuf* originalCerrBuf_;
    std::unique_ptr<LoggerStreamBuf> coutRedirector_;
    std::unique_ptr<LoggerStreamBuf> cerrRedirector_;
    bool redirectionActive_;
};

// Template implementation
template<typename T>
void Logger::setValue(const std::string& name, const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = variables_.find(name);
    if (it != variables_.end()) {
        it->second.value = formatString(it->second.format, value);
    }
}

template<typename T>
void Logger::setValue(const std::string& name, const std::vector<T>& values) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = variables_.find(name);
    if (it != variables_.end()) {
        it->second.value = formatVector(it->second.format, values);
    }
}

template<typename T>
std::string Logger::formatString(const std::string& format, const T& value) {
    std::stringstream ss;
    size_t pos = format.find("{}");
    
    if (pos == std::string::npos) {
        // No placeholder, just output the value
        ss << value;
        return ss.str();
    }
    
    // Format the string by replacing {}
    ss << format.substr(0, pos);
    
    // Apply default formatting for numeric types
    if constexpr (std::is_floating_point<T>::value) {
        ss << std::fixed << std::setprecision(2) << value;
    } else {
        ss << value;
    }
    
    ss << format.substr(pos + 2);
    return ss.str();
}

template<typename T>
std::string Logger::formatVector(const std::string& format, const std::vector<T>& values) {
    std::stringstream ss;
    
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) ss << ", ";
        
        // Format each element
        if constexpr (std::is_floating_point<T>::value) {
            ss << std::fixed << std::setprecision(2) << values[i];
        } else {
            ss << values[i];
        }
    }
    ss << "]";
    
    // Replace placeholder in the format string
    std::string result = format;
    size_t pos = result.find("{}");
    if (pos != std::string::npos) {
        result.replace(pos, 2, ss.str());
    } else {
        result = ss.str();
    }
    
    return result;
}
