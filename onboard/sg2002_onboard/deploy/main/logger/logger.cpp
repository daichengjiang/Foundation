// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "logger.h"

// LoggerStreamBuf implementation
LoggerStreamBuf::LoggerStreamBuf(Logger* logger, LogLevel level)
    : logger_(logger), level_(level), buffer_() {
}

LoggerStreamBuf::~LoggerStreamBuf() {
    sync(); // Make sure to flush any remaining content
}

std::streambuf::int_type LoggerStreamBuf::overflow(int_type c) {
    if (c != EOF) {
        if (c == '\n') {
            // On newline, flush the buffer to the logger
            sync();
        } else {
            buffer_ += static_cast<char>(c);
        }
    }
    return c;
}

int LoggerStreamBuf::sync() {
    if (!buffer_.empty()) {
        // Send the buffer to the logger as a temp message
        logger_->addTempMessage(buffer_, level_);
        buffer_.clear();
    }
    return 0;
}

// Logger implementation
Logger::Logger(int printFrequencyMs, LogLevel minLogLevel)
    : printFrequencyMs_(printFrequencyMs)
    , minLogLevel_(minLogLevel)
    , running_(false)
    , useColors_(true)
    , originalCoutBuf_(nullptr)
    , originalCerrBuf_(nullptr)
    , redirectionActive_(false)
{
}

Logger::~Logger() {
    stop();
    stopRedirection();
}

void Logger::addVariable(const std::string& name, const std::string& format, LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    VariableInfo info;
    info.format = format;
    info.value = "N/A";
    info.level = level;
    
    variables_[name] = info;
}

void Logger::removeVariable(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = variables_.find(name);
    if (it != variables_.end()) {
        variables_.erase(it);
    }
}

void Logger::addTempMessage(const std::string& message, LogLevel level, int durationMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    TempMessage tempMsg;
    tempMsg.message = message;
    tempMsg.level = level;
    tempMsg.expireTime = std::chrono::steady_clock::now() + 
                        std::chrono::milliseconds(durationMs);
    
    tempMessages_.push_back(tempMsg);
}

void Logger::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        running_ = true;
        printThread_ = std::thread(&Logger::printLoop, this);
    }
}

void Logger::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    if (printThread_.joinable()) {
        printThread_.join();
    }
}

void Logger::setMinLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    minLogLevel_ = level;
}

void Logger::setPrintFrequency(int frequencyMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    printFrequencyMs_ = frequencyMs;
}

void Logger::startRedirection() {
    if (!redirectionActive_) {
        // Store original stream buffers
        originalCoutBuf_ = std::cout.rdbuf();
        originalCerrBuf_ = std::cerr.rdbuf();

        // Create new stream buffers
        coutRedirector_ = std::make_unique<LoggerStreamBuf>(this, LogLevel::INFO);
        cerrRedirector_ = std::make_unique<LoggerStreamBuf>(this, LogLevel::ERROR);

        // Redirect stdout and stderr
        std::cout.rdbuf(coutRedirector_.get());
        std::cerr.rdbuf(cerrRedirector_.get());

        redirectionActive_ = true;
    }
}

void Logger::stopRedirection() {
    if (redirectionActive_) {
        // Restore original stream buffers
        if (originalCoutBuf_) {
            std::cout.rdbuf(originalCoutBuf_);
        }
        if (originalCerrBuf_) {
            std::cerr.rdbuf(originalCerrBuf_);
        }

        // Reset the redirectors
        coutRedirector_.reset();
        cerrRedirector_.reset();

        redirectionActive_ = false;
    }
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO ";
        case LogLevel::WARNING:
            return "WARN ";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "NONE ";
    }
}

std::string Logger::colorizeLevel(LogLevel level, const std::string& text) {
    if (!useColors_) return text;
    
    // ANSI escape codes for colors
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string BLUE = "\033[34m";
    const std::string YELLOW = "\033[33m";
    const std::string RED = "\033[31m";
    
    switch (level) {
        case LogLevel::DEBUG:
            return BLUE + text + RESET;
        case LogLevel::INFO:
            return GREEN + text + RESET;
        case LogLevel::WARNING:
            return YELLOW + text + RESET;
        case LogLevel::ERROR:
            return RED + text + RESET;
        default:
            return text;
    }
}

void Logger::printLoop() {
    using namespace std::chrono;
    
    auto nextPrintTime = steady_clock::now();
    
    while (running_) {
        auto now = steady_clock::now();
        if (now >= nextPrintTime) {
            // Time to print
            std::map<std::string, VariableInfo> varsCopy;
            std::list<TempMessage> tempMsgsCopy;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                varsCopy = variables_; // Copy to print outside the lock
                
                // Copy temporary messages and remove expired ones
                auto it = tempMessages_.begin();
                while (it != tempMessages_.end()) {
                    if (now >= it->expireTime) {
                        it = tempMessages_.erase(it);
                    } else {
                        tempMsgsCopy.push_back(*it);
                        ++it;
                    }
                }
                
                nextPrintTime = now + milliseconds(printFrequencyMs_);
            }
            
            // Group by log level
            std::map<LogLevel, std::vector<std::pair<std::string, std::string>>> levelGroups;
            
            // Add regular variables
            for (const auto& var : varsCopy) {
                if (var.second.level >= minLogLevel_ && var.second.level < LogLevel::NONE) {
                    levelGroups[var.second.level].push_back({var.first, var.second.value});
                }
            }
            
            // Temporarily restore original stdout/stderr buffers during printing
            std::streambuf* tempCoutBuf = nullptr;
            std::streambuf* tempCerrBuf = nullptr;
            
            if (redirectionActive_) {
                tempCoutBuf = std::cout.rdbuf(originalCoutBuf_);
                tempCerrBuf = std::cerr.rdbuf(originalCerrBuf_);
            }
            
            // Print each level group
            if (!levelGroups.empty() || !tempMsgsCopy.empty()) {
                // Get current time as formatted string
                auto timeNow = system_clock::now();
                auto timeT = system_clock::to_time_t(timeNow);
                std::tm* timeInfo = std::localtime(&timeT);
                char timeBuffer[80];
                std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", timeInfo);

                // Clear the console
                std::cout << "\033[2J\033[1;1H"; // ANSI escape code to clear the console

                std::cout << "\n┌─────────────────────── Logger Output [" << timeBuffer << "] ───────────────────────┐" << std::endl;
                
                for (const auto& group : levelGroups) {
                    LogLevel level = group.first;
                    std::string levelStr = logLevelToString(level);
                    
                    for (const auto& var : group.second) {
                        // Don't print temporary messages here to avoid duplication
                        if (var.first != "[TEMP]") {
                            std::cout << "│ " << colorizeLevel(level, "[" + levelStr + "]") << " " 
                                      << std::left << std::setw(20) << var.first 
                                      << " : " << var.second << std::endl;
                        }
                    }
                }

                // Print temporary messages
                if (!tempMsgsCopy.empty()) {
                    std::cout << "\n│ --- Temporary Messages ---" << std::endl;
                    for (const auto& msg : tempMsgsCopy) {
                        if (msg.level >= minLogLevel_ && msg.level < LogLevel::NONE) {
                            std::cout << "│ " << colorizeLevel(msg.level, "[" + logLevelToString(msg.level) + "]") << " " 
                                      << msg.message << std::endl;
                        }
                    }
                }
                
                std::cout << "└───────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            }
            
            // Restore redirection if it was active
            if (redirectionActive_) {
                std::cout.rdbuf(tempCoutBuf);
                std::cerr.rdbuf(tempCerrBuf);
            }
        }
        
        // Sleep for a short time to check again
        std::this_thread::sleep_for(milliseconds(10));
    }
}
