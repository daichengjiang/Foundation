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

#include <cviruntime.h>
#include <string>
#include <vector>

class TPUInference {
public:
    TPUInference(const std::string& model_path, bool verbose = false);
    ~TPUInference();

    // Inference method - takes input vector and returns output vector, also returns inference time
    std::vector<float> inference(const std::vector<float>& input, double& inference_time_ms);

    // Get input and output tensor information
    std::vector<int> getInputShape() const;
    std::vector<int> getOutputShape() const;
    int getInputSize() const;
    int getOutputSize() const;
    
    // Check if TPU is correctly initialized
    bool isInitialized() const;

private:
    CVI_MODEL_HANDLE model;
    CVI_TENSOR *input_tensors;
    CVI_TENSOR *output_tensors;
    int32_t input_num;
    int32_t output_num;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    int input_size;
    int output_size;
    bool initialized;
    bool verbose;
};
