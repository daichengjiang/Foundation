// Copyright (c) 2025 Xu Yang
// HKUST UAV Group
//
// Author: Xu Yang
// Affiliation: HKUST UAV Group
// Date: April 2025
// License: MIT License

#include "init_serial.h"

#include <iostream>
#include <string>
#include <exception>
#include <iomanip>

// Helper functions to convert serial settings to strings
std::string getByteSize(serial::bytesize_t bytesize) {
  switch(bytesize) {
    case serial::fivebits: return "5 bits";
    case serial::sixbits: return "6 bits";
    case serial::sevenbits: return "7 bits";
    case serial::eightbits: return "8 bits";
    default: return "unknown";
  }
}

std::string getParity(serial::parity_t parity) {
  switch(parity) {
    case serial::parity_none: return "None";
    case serial::parity_odd: return "Odd";
    case serial::parity_even: return "Even";
    case serial::parity_mark: return "Mark";
    case serial::parity_space: return "Space";
    default: return "unknown";
  }
}

std::string getStopBits(serial::stopbits_t stopbits) {
  switch(stopbits) {
    case serial::stopbits_one: return "1 bit";
    case serial::stopbits_one_point_five: return "1.5 bits";
    case serial::stopbits_two: return "2 bits";
    default: return "unknown";
  }
}

std::string getFlowControl(serial::flowcontrol_t flowcontrol) {
  switch(flowcontrol) {
    case serial::flowcontrol_none: return "None";
    case serial::flowcontrol_software: return "Software";
    case serial::flowcontrol_hardware: return "Hardware";
    default: return "unknown";
  }
}

void initSerial(serial::Serial *serial, std::string port_name, int baud_rate) {
  try {
    std::cout << "======== Serial Port Initialization Started ========" << std::endl;
    std::cout << "Port name: " << port_name << std::endl;
    std::cout << "Baud rate: " << baud_rate << std::endl;
    
    serial->setPort(port_name);
    serial->setBaudrate(static_cast<uint32_t>(baud_rate));
    
    // Get default settings and print them
    std::cout << "Default configuration:" << std::endl;
    std::cout << "  Bytesize: " << getByteSize(serial->getBytesize()) << std::endl;
    std::cout << "  Parity: " << getParity(serial->getParity()) << std::endl;
    std::cout << "  Stopbits: " << getStopBits(serial->getStopbits()) << std::endl;
    std::cout << "  Flowcontrol: " << getFlowControl(serial->getFlowcontrol()) << std::endl;
    
    auto timeout = serial->getTimeout();
    std::cout << "  Timeout settings:" << std::endl;
    std::cout << "    Inter-byte timeout: " << timeout.inter_byte_timeout << " ms" << std::endl;
    std::cout << "    Read timeout constant: " << timeout.read_timeout_constant << " ms" << std::endl;
    std::cout << "    Read timeout multiplier: " << timeout.read_timeout_multiplier << " ms/byte" << std::endl;
    std::cout << "    Write timeout constant: " << timeout.write_timeout_constant << " ms" << std::endl;
    std::cout << "    Write timeout multiplier: " << timeout.write_timeout_multiplier << " ms/byte" << std::endl;
    
    std::cout << "Setting timeout to simpleTimeout(10)..." << std::endl;
    auto new_timeout = serial::Timeout::simpleTimeout(10);
    serial->setTimeout(new_timeout);
    
    std::cout << "Opening serial port..." << std::endl;
    serial->open();

    if (serial->isOpen()) {
      std::cout << "===== Serial port opened successfully =====" << std::endl;
      std::cout << "Current configuration after port open:" << std::endl;
      std::cout << "  Port: " << serial->getPort() << std::endl;
      std::cout << "  Baud rate: " << serial->getBaudrate() << std::endl;
      std::cout << "  Bytesize: " << getByteSize(serial->getBytesize()) << std::endl;
      std::cout << "  Parity: " << getParity(serial->getParity()) << std::endl;
      std::cout << "  Stopbits: " << getStopBits(serial->getStopbits()) << std::endl;
      std::cout << "  Flowcontrol: " << getFlowControl(serial->getFlowcontrol()) << std::endl;
      
      auto current_timeout = serial->getTimeout();
      std::cout << "  Current timeout settings:" << std::endl;
      std::cout << "    Inter-byte timeout: " << current_timeout.inter_byte_timeout << " ms" << std::endl;
      std::cout << "    Read timeout constant: " << current_timeout.read_timeout_constant << " ms" << std::endl;
      std::cout << "    Read timeout multiplier: " << current_timeout.read_timeout_multiplier << " ms/byte" << std::endl;
      std::cout << "    Write timeout constant: " << current_timeout.write_timeout_constant << " ms" << std::endl;
      std::cout << "    Write timeout multiplier: " << current_timeout.write_timeout_multiplier << " ms/byte" << std::endl;
      
      size_t available_bytes = serial->available();
      std::cout << "  Available bytes in input buffer: " << available_bytes << std::endl;
      
      std::cout << "Serial port ready for communication." << std::endl;
    } else {
      std::cerr << "ERROR: Failed to open serial port, please check and retry." << std::endl;
      exit(EXIT_FAILURE);
    }
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Unhandled Exception during serial initialization: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}
