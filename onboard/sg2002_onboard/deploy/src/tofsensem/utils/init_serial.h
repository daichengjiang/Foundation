/*
 * Copyright (c) 2025 Xu Yang
 * HKUST UAV Group
 *
 * Author: Xu Yang
 * Affiliation: HKUST UAV Group
 * Date: April 2025
 * License: MIT License
 */

#ifndef INITSERIAL_H
#define INITSERIAL_H
#include <serial/serial.h>

void initSerial(serial::Serial *serial, std::string port_name, int baud_rate);

#endif // INITSERIAL_H
