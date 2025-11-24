// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>
#include <iomanip>
#include <cmath>
#include <getopt.h>

#include "CrazyflieLink.h"
#include "Logging.h"
#include "Commander.h"
#include "tofImpl.h"
#include "CallbackRateStatistics.h"
#include "observation_buffer.h"
#include "tof_converter.h"
#include "logger.h"
#include "recorder.h"
#include "receiver.h"

const float RAD_TO_DEG = 180.0f / std::acos(-1.0f);
const float DEG_TO_RAD = std::acos(-1.0f) / 180.0f;

// Add these constants near the beginning of main
const float EPS = 1e-3f;          // dead-zone threshold
const float V_MIN = 0.01f;        // minimum velocity magnitude
const float MAX_SPEED = 0.5f;     // existing maximum speed factor

// Add static variables to track last valid command
static float last_ref_vx = 1.0f;    // default forward direction
static float last_ref_vy = 0.0f;
static float last_target_vel = V_MIN;

// Add FPS tracking class for control loop
class FPSTracker {
public:
    FPSTracker(int windowSize = 50) : windowSize_(windowSize) {
        timestamps_.reserve(windowSize);
        reset();
    }

    void update() {
        auto now = std::chrono::steady_clock::now();
        timestamps_.push_back(now);
        
        // Keep only the latest windowSize_ timestamps
        if (timestamps_.size() > windowSize_) {
            timestamps_.erase(timestamps_.begin());
        }

        // Calculate FPS based on timestamps in the window
        if (timestamps_.size() > 1) {
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                timestamps_.back() - timestamps_.front()).count();
            
            // Avoid division by zero
            if (elapsed > 0) {
                // Convert to seconds and calculate FPS
                double seconds = elapsed / 1000000.0;
                currentFps_ = (timestamps_.size() - 1) / seconds;
            }
        }
    }

    double getFPS() const {
        return currentFps_;
    }

    void reset() {
        timestamps_.clear();
        currentFps_ = 0.0;
    }

private:
    std::vector<std::chrono::steady_clock::time_point> timestamps_;
    int windowSize_;
    double currentFps_;
};

bool g_running = true;

// Add global FPS tracker
FPSTracker g_controlLoopFps;

void signalHandler(int signum) {
    g_running = false;
}

// Create logger and add variables
std::shared_ptr<Logger> g_logger;
void loggerAddVariables() {
    g_logger->addVariable("inference time", "Inference: {}", LogLevel::INFO);
    g_logger->addVariable("position", "Position: {}", LogLevel::INFO);
    g_logger->addVariable("velocity", "Velocity: {}", LogLevel::INFO);
    g_logger->addVariable("velocity_body", "Velocity Body: {}", LogLevel::INFO);
    g_logger->addVariable("command", "Command: {}", LogLevel::INFO);
    g_logger->addVariable("auto_command", "Auto Command: {}", LogLevel::INFO);
    g_logger->addVariable("vbat", "Battery Voltage: {}", LogLevel::INFO);
    g_logger->addVariable("orientation", "Orientation: {}", LogLevel::INFO);
    g_logger->addVariable("tof_visualization", "{}", LogLevel::INFO);
    g_logger->addVariable("control_fps", "AUTO Command Rate: {:.2f} Hz", LogLevel::INFO);
    g_logger->addVariable("tof_omni", "ToF Directions: {}", LogLevel::INFO);
    g_logger->addVariable("connection_status", "Connection: {}", LogLevel::INFO);
}

CallbackRateStatistics callbackRateStats(true, nullptr);  // We'll set the logger later

// Create a global ObservationBuffer instance
std::shared_ptr<ObservationBuffer> g_obsBuffer;

// Add a global target height variable
float g_targetHeight = 0.0f;

// Create a global recorder instance
std::shared_ptr<DataRecorder> g_recorder;

// Modify the visualizeToFData function to update the logger variable instead of using temp messages
void visualizeToFData(const std::array<float, TOF_SIZE>& tofArray) {
    static auto lastPrintTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    
    // Limit update frequency to reduce console spam (update every 50ms)
    if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastPrintTime).count() >= 50) {
        lastPrintTime = currentTime;
        
        std::stringstream ss;
        ss << "\n┌─────────── ToF Data (meters) ───────────┐\n";
        
        // Print the 8x8 ToF data as a grid
        for (int row = 0; row < TOF_HEIGHT; row++) {
            ss << "│ ";
            for (int col = 0; col < TOF_WIDTH; col++) {
                int idx = row * TOF_WIDTH + col;
                float value = tofArray[idx];
                
                // Color coding based on distance
                std::string colorCode;
                if (value < 0.5f) {
                    // Red for close objects
                    colorCode = "\033[1;31m";
                } else if (value < 1.0f) {
                    // Yellow for medium distance
                    colorCode = "\033[1;33m";
                } else if (value < 2.0f) {
                    // Green for safe distance
                    colorCode = "\033[1;32m";
                } else {
                    // Blue for far distance
                    colorCode = "\033[1;34m";
                }
                
                // Reset color
                std::string resetCode = "\033[0m";
                
                // Format value with fixed width and precision
                ss << colorCode << std::fixed << std::setprecision(2) << std::setw(4) << value << resetCode << " ";
            }
            ss << "│\n";
        }
        
        ss << "└──────────────────────────────────────────┘";

        if (g_logger) {
            g_logger->setValue("tof_visualization", ss.str());
        } else {
            std::cout << ss.str() << std::endl;
        }
    }
}

// Function to print help message
void printHelp(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help            Display this help message and exit" << std::endl;
    std::cout << "  -r, --record          Enable data recording" << std::endl;
    std::cout << "  -l, --log-dir <dir>   Specify log directory (default: log)" << std::endl;
    std::cout << std::endl;
    std::cout << "Description:" << std::endl;
    std::cout << "  This program controls a Crazyflie drone using either manual or autonomous" << std::endl;
    std::cout << "  control. When recording is enabled, flight data is saved to CSV log files" << std::endl;
    std::cout << "  in the specified directory." << std::endl;
}

// Function to parse command line arguments
void parseArgs(int argc, char** argv, bool& enableRecording, std::string& logDir) {
    const struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"record", no_argument, 0, 'r'},
        {"log-dir", required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "hrl:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                printHelp(argv[0]);
                exit(0);
                break;
            case 'r':
                enableRecording = true;
                break;
            case 'l':
                logDir = optarg;
                break;
            default:
                printHelp(argv[0]);
                exit(1);
                break;
        }
    }
}

// Struct for accelerometer and gyroscope data
struct AccGyroData {
    float acc_x;
    float acc_y;
    float acc_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
} __attribute__((packed)) accGyroData;

// Callback for accelerometer and gyroscope data
void accGyroDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_accGyro");
    if (data.size() != sizeof(AccGyroData)) {
        std::cerr << "Data size mismatch: expected " << sizeof(AccGyroData) << ", got " << data.size() << std::endl;
        std::cerr << "If caused by memory alignment, assign the data manually." << std::endl;
        return;
    }

    std::memcpy(&accGyroData, data.data(), data.size());
}

// Struct for orientation data
struct OrientationData {
    float roll;
    float pitch;
    float yaw;
} __attribute__((packed)) orientationData;

// Callback for orientation data
void orientationDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_orientation");
    if (data.size() != sizeof(OrientationData)) {
        std::cerr << "Data size mismatch: expected " << sizeof(OrientationData) << ", got " << data.size() << std::endl;
        std::cerr << "If caused by memory alignment, assign the data manually." << std::endl;
        return;
    }
    
    std::memcpy(&orientationData, data.data(), data.size());
    orientationData.pitch = -orientationData.pitch; // special case for Crazyflie
    if (g_obsBuffer) {
        g_obsBuffer->updateOrientation(orientationData.roll * DEG_TO_RAD,
                                       orientationData.pitch * DEG_TO_RAD,
                                       orientationData.yaw * DEG_TO_RAD);
    }
}

// Struct for position and velocity data
struct alignas(4) PVData {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
} pvData;

// Callback for position and velocity data
void pvDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_pv");
    if (data.size() != sizeof(PVData)) {
        std::cerr << "Data size mismatch: expected " << sizeof(PVData) << ", got " << data.size() << std::endl;
        std::cerr << "If caused by memory alignment, assign the data manually." << std::endl;
        return;
    }

    std::memcpy(&pvData, data.data(), data.size());

    if (g_obsBuffer) {
        g_obsBuffer->updatePosition(pvData.x, pvData.y, pvData.z);
        g_obsBuffer->updateVelocity(pvData.vx, pvData.vy, pvData.vz);
    }
}

// Struct for flow data
struct FlowData {
    int16_t deltaX;
    int16_t deltaY;
    uint8_t squal;
} __attribute__((packed)) flowData;

// Callback for flow data
void flowDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_flow");
    flowData.deltaX = (int16_t)((data[1] << 8) | data[0]);
    flowData.deltaY = (int16_t)((data[3] << 8) | data[2]);
    flowData.squal = data[4];
}

// Struct for barometer data
struct BaroData {
    float asl;
} __attribute__((packed)) baroData;

// Callback for barometer data
void baroDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_baro");
    if (data.size() != sizeof(BaroData)) {
        std::cerr << "Data size mismatch: expected " << sizeof(BaroData) << ", got " << data.size() << std::endl;
        std::cerr << "If caused by memory alignment, assign the data manually." << std::endl;
        return;
    }

    std::memcpy(&baroData, data.data(), data.size());
}

// Struct for ToF range data - updated to include all directions
struct TofOmniData {
    float zrange;
    float back;
    float left;
    float right;
} __attribute__((packed)) tofOmniData;

// Callback for ToF range data - updated for all directions
void tofOmniDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_tofOmni");
    if (data.size() != sizeof(tofOmniData)) {
        std::cerr << "Data size mismatch: expected " << sizeof(tofOmniData) << ", got " << data.size() << std::endl;
        return;
    }

    std::memcpy(&tofOmniData, data.data(), data.size());

    // Convert mm to m and clamp to max 4m
    const float MAX_TOF_RANGE = 4.0f;
    tofOmniData.zrange = std::min(tofOmniData.zrange / 1000.0f, MAX_TOF_RANGE);
    tofOmniData.left = std::min(tofOmniData.left / 1000.0f, MAX_TOF_RANGE);
    tofOmniData.right = std::min(tofOmniData.right / 1000.0f, MAX_TOF_RANGE);
    tofOmniData.back = std::min(tofOmniData.back / 1000.0f, MAX_TOF_RANGE);
    
    // Update observation buffer with direction TOF data
    if (g_obsBuffer) {
        g_obsBuffer->updateDirectionalTof(tofOmniData.left, tofOmniData.right, tofOmniData.back);
    }
}

// Struct for system status data
struct SysData {
    uint8_t canfly;
    uint8_t isFlying;
    uint8_t isTumbled;
    float vbat;
}  __attribute__((packed)) sysData;

// Callback for system status data
void sysDataCallback(const uint32_t timestamp, const std::vector<uint8_t>& data) {
    callbackRateStats.incrementCallbackCount("Callback_sys");
    if (data.size() != sizeof(SysData)) {
        g_logger->addTempMessage("Data size mismatch: expected " + std::to_string(sizeof(SysData)) + ", got " + std::to_string(data.size()), LogLevel::ERROR);
        return;
    }

    std::memcpy(&sysData, data.data(), data.size());
}

void tofMCallback(ntsm_frame0_t frame) {
    callbackRateStats.incrementCallbackCount("Callback_tofM");
    if (g_obsBuffer) {
        auto tofArray = ToFConverter::frameToArray(frame);
        
        // Add visualization of ToF data
        visualizeToFData(tofArray);
        
        g_obsBuffer->updateToFData(tofArray);
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    // Parse command line arguments
    bool enableRecording = false;
    std::string logDir = "log";
    parseArgs(argc, argv, enableRecording, logDir);

    try {
        // Initialize logger
        g_logger = std::make_shared<Logger>(50, LogLevel::INFO);
        callbackRateStats.setLogger(g_logger.get());

        // Initialize recorder
        g_recorder = std::make_shared<DataRecorder>(enableRecording, logDir);

        g_obsBuffer = std::make_shared<ObservationBuffer>(2, "model.cvimodel");  // Updated model path to .cvimodel

        auto link = std::make_shared<crazyflie::CrazyflieLink>("/dev/ttyS2", 230400);

        loggerAddVariables();
        g_logger->start();
        g_logger->startRedirection();

        auto receiver = std::make_shared<Receiver>(2333, g_logger.get());
        receiver->start();

        auto accelGyroBlock = link->createLoggingBlock();
        accelGyroBlock->addVariable("acc.x", "float", 170);
        accelGyroBlock->addVariable("acc.y", "float", 171);
        accelGyroBlock->addVariable("acc.z", "float", 172);
        accelGyroBlock->addVariable("gyro.x", "float", 107);
        accelGyroBlock->addVariable("gyro.y", "float", 108);
        accelGyroBlock->addVariable("gyro.z", "float", 109);
        accelGyroBlock->setCallback(accGyroDataCallback);
        accelGyroBlock->start(1);

        auto orientationBlock = link->createLoggingBlock();
        orientationBlock->addVariable("stateEstimate.roll", "float", 231);
        orientationBlock->addVariable("stateEstimate.pitch", "float", 232);
        orientationBlock->addVariable("stateEstimate.yaw", "float", 233);
        orientationBlock->setCallback(orientationDataCallback);
        orientationBlock->start(1);

        auto pvBlock = link->createLoggingBlock();
        pvBlock->addVariable("stateEstimate.x", "float", 222);
        pvBlock->addVariable("stateEstimate.y", "float", 223);
        pvBlock->addVariable("stateEstimate.z", "float", 224);
        pvBlock->addVariable("stateEstimate.vx", "float", 225);
        pvBlock->addVariable("stateEstimate.vy", "float", 226);
        pvBlock->addVariable("stateEstimate.vz", "float", 227);
        pvBlock->setCallback(pvDataCallback);
        pvBlock->start(1);

        auto flowBlock = link->createLoggingBlock();
        flowBlock->addVariable("motion.deltaX", "int16_t", 3);
        flowBlock->addVariable("motion.deltaY", "int16_t", 4);
        flowBlock->addVariable("motion.squal", "uint8_t", 10);
        flowBlock->setCallback(flowDataCallback);
        flowBlock->start(1);

        auto baroBlock = link->createLoggingBlock();
        baroBlock->addVariable("baro.asl", "float", 173);
        baroBlock->setCallback(baroDataCallback);
        baroBlock->start(2);

        auto tofOmniBlock = link->createLoggingBlock();
        tofOmniBlock->addVariable("range.zrange", "float", 159);
        tofOmniBlock->addVariable("range.back", "float", 155);
        tofOmniBlock->addVariable("range.left", "float", 157);
        tofOmniBlock->addVariable("range.right", "float", 158);
        tofOmniBlock->setCallback(tofOmniDataCallback);
        tofOmniBlock->start(4);

        auto sysBlock = link->createLoggingBlock();
        sysBlock->addVariable("sys.canfly", "uint8_t", 252);
        sysBlock->addVariable("sys.isFlying", "uint8_t", 253);
        sysBlock->addVariable("sys.isTumbled", "uint8_t", 254);
        sysBlock->addVariable("pm.vbat", "float", 89);
        sysBlock->setCallback(sysDataCallback);
        sysBlock->start(10);

        tofsensem::Impl impl(tofMCallback, "/dev/ttyS1", 230400);

        auto commander = link->getCommander();

        std::cout << "Running... Press Ctrl+C to exit." << std::endl;
        commander->sendSetpoint(0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Define target loop period (100 Hz = 10 ms)
        const std::chrono::milliseconds targetPeriod(10);
        auto previousTime = std::chrono::steady_clock::now();

        // Initialize FPS tracker
        g_controlLoopFps.reset();

        // Add variable to track previous control state
        ControlState previousControlState = ControlState::LOCKED;

        while (g_running) {
            // Get current control state
            ControlState controlState = receiver->getControlState();
            
            // Check for state transitions to arm/disarm motors
            if (previousControlState != controlState) {
                // Handle state transitions
                if (previousControlState == ControlState::LOCKED && 
                   (controlState == ControlState::UNLOCKED || controlState == ControlState::AUTO)) {
                    // Transition from LOCKED to UNLOCKED/AUTO - arm motors
                    g_logger->addTempMessage("Arming motors", LogLevel::INFO);
                    commander->arm();
                    
                    // Reset timestamps to ensure smooth dt calculations after mode change
                    if (g_obsBuffer) {
                        g_obsBuffer->resetTimestamps();
                    }
                } 
                else if ((previousControlState == ControlState::UNLOCKED || previousControlState == ControlState::AUTO) && 
                          controlState == ControlState::LOCKED) {
                    // Transition from UNLOCKED/AUTO to LOCKED - disarm motors
                    g_logger->addTempMessage("Disarming motors", LogLevel::INFO);
                    commander->disarm();
                }
                
                // Update previous state
                previousControlState = controlState;
            }
            
            if(controlState == ControlState::LOCKED) {
                commander->sendSetpoint(0, 0, 0, 0);
            }

            // Get joystick input
            float roll = 0.0f;
            float pitch = 0.0f;
            float yawrate = 0.0f;
            float heightDelta = 0.0f;

            // Velocity-based control
            float vx = 0.0f;
            float vy = 0.0f;
            float vz = 0.0f;

            // Manual mode command variables
            float thrust = 0.0f;
            uint16_t scaledThrust = 0;

            // Auto mode command variables
            float auto_roll = 0.0f;
            float auto_pitch = 0.0f;
            float auto_yawrate = 0.0f;
            float auto_thrust = 0.0f;
            uint16_t auto_scaledThrust = 0;
            bool autoCommandReady = false;

            // Calculate UNLOCKED mode command (velocity based)
            std::stringstream cmdStr;
            bool joystickInputReceived = receiver->getJoystickCommand(roll, pitch, yawrate, heightDelta); //[-pi/6, pi/6], [-pi/6, pi/6], [-pi/6, pi/6], [-0.05, 0.05]
            if (joystickInputReceived) {receiver->clearJoystickCommand();}
            if (joystickInputReceived) {
                vx = pitch;
                vy = roll;
                vz = heightDelta * 10;
                cmdStr << "[Velocity: (" << vx << ", " << -vy << ", " << vz << "), Yaw rate: " << -yawrate * RAD_TO_DEG << "]";
                g_logger->setValue("command", cmdStr.str());

                // Manual control (preserved but commented)
                // scaledThrust = static_cast<uint16_t>(10000 + heightDelta * 1000000);
                // std::stringstream manualCmdStr;
                // manualCmdStr << "[Roll: " << roll * RAD_TO_DEG << ", Pitch: " << pitch * RAD_TO_DEG
                //              << ", Yaw rate: " << yawrate * RAD_TO_DEG << ", Thrust: " << scaledThrust << "]";
                // g_logger->setValue("command", manualCmdStr.str());
            }

            // Calculate AUTO mode command (attitude based)
            std::stringstream autoCmdStr;
            if (g_obsBuffer->isFrameComplete() && joystickInputReceived) {
                // Start timing the entire command calculation
                auto cmd_start_time = std::chrono::high_resolution_clock::now();
                // Normalize roll and pitch to [-1, 1]
                roll = roll / (3.14159f / 6.0f);
                pitch = pitch / (3.14159f / 6.0f);

                // Map roll, pitch to target velocity
                float ref_vx = pitch;  // Normalized direction vector X
                float ref_vy = -roll;   // Normalized direction vector Y
                float target_z = 0.5f;    // Target height with joystick adjustment

                // Set target velocity magnitude and target height manually
                // ref_vx = 0.707f;
                // ref_vy = -0.707f;

                // Compute raw direction and magnitude
                float norm = std::sqrt(ref_vx * ref_vx + ref_vy * ref_vy);
                float target_vel = norm * MAX_SPEED;  // original scaling

                if (norm > EPS) {
                    // Valid joystick input — normalize and enforce minimum speed
                    ref_vx /= norm;
                    ref_vy /= norm;
                    target_vel = std::max(norm * MAX_SPEED, V_MIN);

                    // Remember last valid command
                    last_ref_vx = ref_vx;
                    last_ref_vy = ref_vy;
                    last_target_vel = target_vel;
                } else {
                    // Joystick in dead-zone — reuse last direction & speed
                    ref_vx = last_ref_vx;
                    ref_vy = last_ref_vy;
                    target_vel = last_target_vel;
                }

                ref_vx = 1.0;
                ref_vy = 0;
                target_vel = 0.5f;

                // Pass direction vector and target values to observation buffer
                g_obsBuffer->setTargetVelocity(ref_vx, ref_vy, target_vel, target_z);

                // Start timing the TPU processing specifically
                auto tpu_start_time = std::chrono::high_resolution_clock::now();
                g_obsBuffer->processFrame();
                auto tpu_end_time = std::chrono::high_resolution_clock::now();
                float tpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(tpu_end_time - tpu_start_time).count() / 1000.0;

                g_obsBuffer->getCommand(auto_roll, auto_pitch, auto_yawrate, auto_thrust); // roll, pitch, yawrate [-1, 1], thrust [0, 1]
                // Map roll, pitch to [-pi/6, pi/6]
                auto_roll = auto_roll * 3.14159f / 6.0f;
                auto_pitch = auto_pitch * 3.14159f / 6.0f;
                auto_yawrate = auto_yawrate * 3.14159f / 6.0f;
                // Map thrust [0, 1] to [10000, 60000], Hover thrust is 37000, 46000 for brushless
                auto_scaledThrust = static_cast<uint16_t>(22000 + auto_thrust * 40000);

                autoCommandReady = true;

                // End timing for the entire command calculation
                auto cmd_end_time = std::chrono::high_resolution_clock::now();
                float cmd_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(cmd_end_time - cmd_start_time).count() / 1000.0;

                // Log both timing values with fixed precision
                std::stringstream inferenceTimeStr;
                inferenceTimeStr << "command time: " << std::fixed << std::setprecision(4) << cmd_time_ms << "ms, TPU time: " << std::fixed << std::setprecision(4) << tpu_time_ms << "ms";
                g_logger->setValue("inference time", inferenceTimeStr.str());

                autoCmdStr << "[Roll: " << std::fixed << std::setprecision(2) << auto_roll * RAD_TO_DEG
                        << ", Pitch: " << std::fixed << std::setprecision(2) << auto_pitch * RAD_TO_DEG
                        << ", Yaw rate: " << std::fixed << std::setprecision(2) << auto_yawrate * RAD_TO_DEG
                        << ", Thrust: " << std::setw(5) << auto_scaledThrust << "]";
                g_logger->setValue("auto_command", autoCmdStr.str());
            }

            // Execute the appropriate command based on the current control state
            if (joystickInputReceived) {
                switch (controlState) {
                    case ControlState::LOCKED:
                        // In locked state, all controls are zeroed
                        commander->sendSetpoint(0, 0, 0, 0);
                        break;

                    case ControlState::UNLOCKED:
                        // In unlocked state, use velocity commands
                        commander->sendVelocityWorld(vx, -vy, vz, -yawrate * RAD_TO_DEG);
                        // commander->sendSetpoint(roll * RAD_TO_DEG, -pitch * RAD_TO_DEG, -yawrate * RAD_TO_DEG, scaledThrust);
                        // if (g_recorder->isEnabled()) {
                        //     g_recorder->writeData(
                        //         // Control commands (no direct setpoint values available, so use velocity command values)
                        //         0, 0, -yawrate * RAD_TO_DEG, 0,
                        //         // Position
                        //         pvData.x, pvData.y, pvData.z,
                        //         // Velocity
                        //         pvData.vx, pvData.vy, pvData.vz,
                        //         // Attitude
                        //         orientationData.roll, orientationData.pitch, orientationData.yaw,
                        //         // Angular velocity
                        //         accGyroData.gyro_x, accGyroData.gyro_y, accGyroData.gyro_z,
                        //         // Acceleration
                        //         accGyroData.acc_x, accGyroData.acc_y, accGyroData.acc_z
                        //     );
                        // }
                        break;

                    case ControlState::AUTO:
                        // In auto state, use autonomous control if frame is complete
                        if (autoCommandReady) {
                            commander->sendSetpoint(auto_roll * RAD_TO_DEG, -auto_pitch * RAD_TO_DEG, -auto_yawrate * RAD_TO_DEG, auto_scaledThrust);
                            
                            // Update FPS tracker and logger for AUTO command rate
                            g_controlLoopFps.update();
                            g_logger->setValue("control_fps", g_controlLoopFps.getFPS());

                            if (g_recorder->isEnabled()) {
                                g_recorder->writeData(
                                    // Control commands
                                    auto_roll * RAD_TO_DEG, auto_pitch * RAD_TO_DEG, auto_yawrate * RAD_TO_DEG, auto_scaledThrust,
                                    // Position
                                    pvData.x, pvData.y, pvData.z,
                                    // Velocity
                                    pvData.vx, pvData.vy, pvData.vz,
                                    // Attitude
                                    orientationData.roll, orientationData.pitch, orientationData.yaw,
                                    // Angular velocity
                                    accGyroData.gyro_x, accGyroData.gyro_y, accGyroData.gyro_z,
                                    // Acceleration
                                    accGyroData.acc_x, accGyroData.acc_y, accGyroData.acc_z
                                );
                            }
                        } else {
                            commander->sendVelocityWorld(0, 0, 0, 0);
                        }
                        break;
                }
            }

            // Always update orientation, position and velocity logging
            std::stringstream oriStr;
            oriStr << "(" << orientationData.roll << ", " << orientationData.pitch << ", " << orientationData.yaw << ")";
            g_logger->setValue("orientation", oriStr.str());

            std::stringstream posStr;
            posStr << "(" << pvData.x << ", " << pvData.y << ", " << pvData.z << ")";
            g_logger->setValue("position", posStr.str());

            std::stringstream velStr;
            velStr << "(" << pvData.vx << ", " << pvData.vy << ", " << pvData.vz << ")";
            g_logger->setValue("velocity", velStr.str());

            float vx_body, vy_body, vz_body;
            float roll_trans = orientationData.roll * DEG_TO_RAD;
            float pitch_trans = orientationData.pitch * DEG_TO_RAD;
            float yaw_trans = orientationData.yaw * DEG_TO_RAD;
            transform_vel_from_world_to_body(pvData.vx, pvData.vy, pvData.vz,
                                             roll_trans, pitch_trans, yaw_trans,
                                             vx_body, vy_body, vz_body);
            std::stringstream velBodyStr;
            velBodyStr << "(" << vx_body << ", " << vy_body << ", " << vz_body << ")";
            g_logger->setValue("velocity_body", velBodyStr.str());

            g_logger->setValue("vbat", sysData.vbat);

            std::stringstream tofOmniStr;
            tofOmniStr << "Down: " << tofOmniData.zrange << ", Left: " << tofOmniData.left << ", Right: " << tofOmniData.right << ", Back: " << tofOmniData.back;
            g_logger->setValue("tof_omni", tofOmniStr.str());

            // Update connection status in logger
            static bool previously_connected = false;
            bool currently_connected = link->isConnected();
            
            if (currently_connected != previously_connected) {
                // Connection state changed
                if (currently_connected) {
                    g_logger->addTempMessage("Connection established", LogLevel::INFO);
                    
                    // Reset timestamps to ensure smooth dt calculations after reconnection
                    if (g_obsBuffer) {
                        g_obsBuffer->resetTimestamps();
                    }
                } else {
                    g_logger->addTempMessage("Connection lost", LogLevel::WARNING);
                }
                previously_connected = currently_connected;
            }
            
            if (currently_connected) {
                g_logger->setValue("connection_status", "Connected");
            } else {
                g_logger->setValue("connection_status", "Disconnected - Reconnecting...");
            }

            // Calculate next execution time
            auto nextExecutionTime = previousTime + targetPeriod;
            auto currentTime = std::chrono::steady_clock::now();
            
            // If we haven't exceeded the target period, sleep for the remaining time
            if (currentTime < nextExecutionTime) {
                std::this_thread::sleep_until(nextExecutionTime);
            } else {
                std::this_thread::yield();
            }
            
            // Update previous time for next iteration
            previousTime = nextExecutionTime;
        }

        receiver->stop();
        commander->sendSetpoint(0, 0, 0, 0);
        g_logger->stopRedirection();
        g_logger->stop();
        accelGyroBlock->stop();
        orientationBlock->stop();
        pvBlock->stop();
        flowBlock->stop();
        baroBlock->stop();
        tofOmniBlock->stop();
        sysBlock->stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        link->close();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
