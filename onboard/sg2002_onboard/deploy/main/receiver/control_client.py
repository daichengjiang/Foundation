#!/usr/bin/env python3
# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

import socket
import random
import argparse
import time
import sys
import math
import threading
import os
import re

# Import joystick utilities
try:
    from joystick_utils import JoystickController
    JOYSTICK_AVAILABLE = True
except ImportError:
    print("Joystick utils not available. Joystick control will be disabled.")
    JOYSTICK_AVAILABLE = False

class ReceiverTestClient:
    def __init__(self, host, port):
        """Initialize the test client with server address and port."""
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.joystick_controller = None
        self.stop_event = threading.Event()

        # Track height delta parameters for smoothness
        self.height_update_interval = 0.01  # 100Hz updates
        self.last_height_update_time = 0

    def connect(self):
        """Establish connection to the Receiver server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close the connection to the server."""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            print("Disconnected from server")
    
    def send_command_with_ack(self, command):
        """Send a command string to the server and wait for ACK."""
        if not self.connected:
            print("Not connected to server")
            return False
        
        try:
            # Add newline as message terminator
            command += "\n"
            self.socket.sendall(command.encode('utf-8'))
            
            # Wait for ACK
            response = self.socket.recv(1024).decode('utf-8').strip()
            if response == "ACK":
                return True
            else:
                print(f"Received NACK or unexpected response: {response}")
                return False
        except Exception as e:
            print(f"Error sending command with ACK: {e}")
            self.connected = False
            return False

    def send_command_no_ack(self, command):
        """Send a command string to the server without waiting for ACK."""
        if not self.connected:
            print("Not connected to server")
            return False
        try:
            command += "\n"
            self.socket.sendall(command.encode('utf-8'))
            return True  # Assume success if sendall doesn't raise exception
        except Exception as e:
            print(f"Error sending command (no_ack): {e}")
            self.connected = False  # If send fails, connection might be lost
            return False
    
    def send_joystick_command(self, roll, pitch, yawrate, height_delta):
        """Send direct control command from joystick inputs."""
        command = f"JOYSTICK {roll:.4f} {pitch:.4f} {yawrate:.4f} {height_delta:.4f}"
        return self.send_command_no_ack(command) # Use no-ACK version
    
    def send_state_command(self, command):
        """Send state transition command."""
        return self.send_command_with_ack(command) # Keep ACK for state changes
    
    def send_takeoff(self):
        """Send takeoff command."""
        return self.send_command_with_ack("TAKEOFF") # Keep ACK
    
    def send_land(self):
        """Send land command."""
        return self.send_command_with_ack("LAND") # Keep ACK
    
    def initialize_joystick(self):
        """Initialize joystick controller."""
        if not JOYSTICK_AVAILABLE:
            print("Joystick support is not available.")
            return False
            
        self.joystick_controller = JoystickController(deadzone=0.1)
        if not self.joystick_controller.initialize():
            print("No joystick detected or initialization failed.")
            return False
            
        print("Joystick initialized successfully.")
        return True
    
    def run(self):
        """Run the main control loop."""
        if not self.connected and not self.connect():
            print("Failed to connect to server. Exiting.")
            return
            
        if not self.joystick_controller and not self.initialize_joystick():
            print("Failed to initialize joystick. Exiting.")
            return
        
        # Update display instructions
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n=== Quadcopter Control Client ===\n")
        print("Control state: LOCKED")
        print("\nTo unlock: Center throttle stick and long press RB")
        print("\nControls:")
        print("- Left stick X-axis: Yaw rate")
        print("- Left stick Y-axis: Height control (incremental)")
        print("  * Push up: Increase height")
        print("  * Center: Maintain height")
        print("  * Push down: Decrease height")
        print("- Right stick X-axis: Roll")
        print("- Right stick Y-axis: Pitch")
        print("- Hold A: Takeoff (when unlocked)")
        print("- Hold B: Land (anytime)")
        print("- Hold LB: Activate auto mode (when unlocked)")
        print("- Short press RB: Lock controls")
        print("\nPress Ctrl+C to exit\n")
        
        # Track takeoff/land command sending to avoid spamming
        last_takeoff_sent = 0
        last_land_sent = 0
        command_cooldown = 0.5  # seconds between repeated commands
        
        target_period = 0.01 # For 100Hz
        
        try:
            # Main control loop
            while not self.stop_event.is_set():
                loop_start_time = time.time()

                if not self.connected:
                    print("Connection lost. Attempting to reconnect...")
                    if not self.connect():
                        time.sleep(target_period) # Avoid busy loop if reconnect fails
                        continue
                
                current_time = time.time()

                inputs = self.joystick_controller.get_control_inputs()
                
                if inputs is not None:
                    # Process state changes (uses ACK)
                    if inputs["state_change"]:
                        self.send_state_command(inputs["state_change"])
                    
                    # Takeoff/Land buttons (use ACK)
                    # Check for 'a_pressed' and 'b_pressed' existence before using
                    if inputs.get("a_pressed") and inputs["state"] == "UNLOCKED" and \
                       (current_time - last_takeoff_sent) > command_cooldown:
                        self.send_takeoff()
                        last_takeoff_sent = current_time
                    
                    if inputs.get("b_pressed") and (current_time - last_land_sent) > command_cooldown:
                        self.send_land() # Universal land, uses ACK
                        last_land_sent = current_time
                    
                    # Joystick commands (no ACK)
                    if inputs["state"] == "UNLOCKED":
                        if (current_time - self.last_height_update_time) >= self.height_update_interval:
                            self.send_joystick_command(
                                inputs["roll"],
                                inputs["pitch"],
                                inputs["yaw_rate"],
                                inputs["height_delta"]
                            )
                            self.last_height_update_time = current_time
                    elif inputs["state"] == "AUTO":
                        self.send_joystick_command(
                            inputs["roll"],
                            inputs["pitch"],
                            0,  # No yaw rate in AUTO mode
                            0  # No height delta in AUTO mode
                        )
                
                # Sleep to maintain control loop rate
                elapsed_time = time.time() - loop_start_time
                sleep_duration = max(0, target_period - elapsed_time)
                time.sleep(sleep_duration)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.joystick_controller:
                self.joystick_controller.cleanup()
            self.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Quadcopter Control Client')
    parser.add_argument('--host', type=str, default='192.168.0.176', help='IP address of the receiver')
    parser.add_argument('--port', type=int, default=2333, help='Port of the receiver (default: 2333)')
    
    args = parser.parse_args()
    
    client = ReceiverTestClient(args.host, args.port)
    client.run()

if __name__ == "__main__":
    main()
