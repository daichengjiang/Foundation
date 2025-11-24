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

#include "CrazyflieLink.h"
#include "Logging.h"
#include "tofImpl.h"
#include "CallbackRateStatistics.h"
#include "observation_buffer.h"
#include "tof_converter.h"
#include "logger.h"

bool g_running = true;

void signalHandler(int signum) {
    g_running = false;
}

// Create logger and add variables
Logger logger(100, LogLevel::INFO);

CallbackRateStatistics callbackRateStats(true, &logger);

// Create a global ObservationBuffer instance
std::shared_ptr<ObservationBuffer> g_obsBuffer;

// Struct for ToF range data
void tofMCallback(ntsm_frame0_t frame) {
    callbackRateStats.incrementCallbackCount("Callback_tofM");

    if (g_obsBuffer) {
        auto tofArray = ToFConverter::frameToArray(frame);
        g_obsBuffer->updateToFData(tofArray);
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    try {
        g_obsBuffer = std::make_shared<ObservationBuffer>(2, "model.cvimodel");

        // Initialize ToF multi-pixel sensor
        tofsensem::Impl impl(tofMCallback, "/dev/ttyS1", 230400);

        std::cout << "ToF Test Running... Press Ctrl+C to exit." << std::endl;
        logger.start();
        // Simple processing loop
        while (g_running) {
            if (g_obsBuffer->isFrameComplete()) {
                g_obsBuffer->processFrame();
                // Print some information about the processed ToF data
                std::cout << "Processed ToF frame" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // Cleanup
        logger.stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
