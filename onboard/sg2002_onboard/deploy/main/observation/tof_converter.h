/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#ifndef TOF_CONVERTER_H
#define TOF_CONVERTER_H

#include <array>
#include "tofImpl.h"

// Helper class to convert ToF sensor data
class ToFConverter {
public:
    // Convert TofsenseM frame to flat array of distances
    static std::array<float, 64> frameToArray(const ntsm_frame0_t& frame, float maxRange = 4.0f) {
        std::array<float, 64> result;
        
        for (int i = 0; i < 64; i++) {
            // Check if the measurement is valid
            if (frame.pixels[i].dis_status == 0) {
                // Valid measurement: convert from millimeters to meters and limit the maximum range
                float distance = static_cast<float>(frame.pixels[i].dis) / 1000.0f;
                
                // Still perform basic range check
                if (distance > maxRange || distance <= 0.05f) {
                    distance = maxRange;
                }
                
                result[i] = distance;
            } else {
                // Invalid measurement: set to max range
                result[i] = maxRange;
            }
        }
        
        return result;
    }
};

#endif // TOF_CONVERTER_H
