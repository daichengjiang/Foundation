#!/usr/bin/env python
# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License


import serial, sys

if len(sys.argv) != 2:
    print "python: Usage_serial_test <port name like: /dev/ttyUSB0>"
    sys.exit(1)

sio = serial.Serial(sys.argv[1], 115200)
sio.timeout = 250

while True:
    sio.write("Testing.")
    print sio.read(8)

