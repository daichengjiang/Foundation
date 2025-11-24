// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "tpu.h"
#include <cassert>
#include <iostream>
#include <chrono>

TPUInference::TPUInference(const std::string& model_path, bool verbose) 
    : model(nullptr), 
      input_tensors(nullptr), 
      output_tensors(nullptr), 
      input_num(0), 
      output_num(0), 
      input_size(0), 
      output_size(0), 
      initialized(false),
      verbose(verbose) {
    
    // Initialize model
    int ret = CVI_NN_RegisterModel(model_path.c_str(), &model);
    if (CVI_RC_SUCCESS != ret) {
        std::cerr << "CVI_NN_RegisterModel failed, err " << ret << std::endl;
        throw std::runtime_error("Failed to load model");
    }
    if (verbose) std::cout << "CVI_NN_RegisterModel succeeded" << std::endl;
    
    // Get input and output tensors
    CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);
    
    // Get input tensor
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    if (!input) {
        CVI_NN_CleanupModel(model);
        throw std::runtime_error("Failed to get input tensor");
    }
    if (verbose) std::cout << "Input tensor name: " << input->name << std::endl;
    
    // Get output tensor
    CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num);
    if (!output) {
        CVI_NN_CleanupModel(model);
        throw std::runtime_error("Failed to get output tensor");
    }
    
    // Store input and output shapes
    CVI_SHAPE in_shape = CVI_NN_TensorShape(input);
    input_shape.reserve(in_shape.dim_size);
    for (size_t i = 0; i < in_shape.dim_size; i++) {
        input_shape.push_back(in_shape.dim[i]);
    }
    
    CVI_SHAPE out_shape = CVI_NN_TensorShape(output);
    output_shape.reserve(out_shape.dim_size);
    for (size_t i = 0; i < out_shape.dim_size; i++) {
        output_shape.push_back(out_shape.dim[i]);
    }
    
    // Calculate input and output sizes
    input_size = CVI_NN_TensorCount(input);
    output_size = CVI_NN_TensorCount(output);
    
    if (verbose) {
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], size: " << input_size << std::endl;
        
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], size: " << output_size << std::endl;
    }
    
    initialized = true;
}

TPUInference::~TPUInference() {
    // Clean up model
    if (model) {
        CVI_NN_CleanupModel(model);
        if (verbose) std::cout << "CVI_NN_CleanupModel succeeded" << std::endl;
    }
}

std::vector<float> TPUInference::inference(const std::vector<float>& input_data, double& inference_time_ms) {
    if (!initialized) {
        std::cerr << "TPU not properly initialized" << std::endl;
        return {};
    }

    // Get input tensor
    CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    
    // Get output tensor
    CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num);
    
    // Check if sizes match
    if (input_data.size() != static_cast<size_t>(input_size)) {
        std::cerr << "Input data size mismatch. Expected " << input_size 
                  << ", got " << input_data.size() << std::endl;
        return {};
    }
    
    // Copy input data to input tensor
    float *input_ptr = static_cast<float*>(CVI_NN_TensorPtr(input));
    std::copy(input_data.begin(), input_data.end(), input_ptr);
    
    // Measure inference time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run inference
    int ret = CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
    if (ret != CVI_RC_SUCCESS) {
        std::cerr << "Inference failed with error: " << ret << std::endl;
        return {};
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    inference_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Get output data
    float *output_ptr = static_cast<float*>(CVI_NN_TensorPtr(output));
    
    // Copy output data to return vector
    return std::vector<float>(output_ptr, output_ptr + output_size);
}

bool TPUInference::isInitialized() const {
    return initialized;
}

std::vector<int> TPUInference::getInputShape() const {
    return input_shape;
}

std::vector<int> TPUInference::getOutputShape() const {
    return output_shape;
}

int TPUInference::getInputSize() const {
    return input_size;
}

int TPUInference::getOutputSize() const {
    return output_size;
}
