# Copyright (c) 2025 Xu Yang
# HKUST UAV Group
#
# Author: Xu Yang
# Affiliation: HKUST UAV Group
# Date: April 2025
# License: MIT License

import pygame
import math
import time

# Axis  : [MIN, MAX]
# Axis 0: yaw [-1, 1]
# Axis 1: Thrust [1, -1]
# Axis 2: Roll [-1, 1]
# Axis 3: Pitch [1, -1]
# Button 0: A
# Button 1: B
# Button 6：LB
# Button 7：RB

class JoystickController:
    def __init__(self, deadzone=0.15):
        """Initialize the joystick controller with the specified deadzone."""
        self.deadzone = deadzone
        self.initialized = False
        self.joystick = None
        self.state = "LOCKED"  # Initial state: LOCKED, UNLOCKED, AUTO
        self.throttle_in_deadzone = False

        # Button press tracking for long press detection
        self.button_press_start = {
            'a': 0, 'b': 0, 'lb': 0, 'rb': 0
        }
        self.button_state = {
            'a': False, 'b': False, 'lb': False, 'rb': False
        }
        
        # Long press threshold in seconds
        self.long_press_threshold = 0.5
        
    def initialize(self):
        """Initialize pygame and joystick."""
        pygame.init()
        pygame.joystick.init()
        
        # Check if a joystick is connected
        if pygame.joystick.get_count() == 0:
            return False
            
        # Get the first joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.initialized = True
        
        print(f"Initialized joystick: {self.joystick.get_name()}")
        return True
        
    def cleanup(self):
        """Cleanup pygame and joystick resources."""
        if self.initialized:
            pygame.joystick.quit()
            pygame.quit()
            self.initialized = False
    
    def apply_deadzone(self, value):
        """Apply deadzone to the input value."""
        if abs(value) < self.deadzone:
            return 0.0
        
        # Scale the value to account for the deadzone
        # This ensures the output range is still [-1, 1]
        sign = 1.0 if value > 0 else -1.0
        scaled_value = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return sign * scaled_value
    
    def check_button_press(self, button_name, is_pressed):
        """Track button press duration and detect long presses.
        
        Args:
            button_name: Name of the button ('a', 'b', 'lb', 'rb')
            is_pressed: Current state of the button (True if pressed)
            
        Returns:
            tuple: (is_pressed, is_long_press, is_just_released)
        """
        current_time = time.time()
        was_pressed = self.button_state[button_name]
        self.button_state[button_name] = is_pressed
        
        # Button just pressed
        if is_pressed and not was_pressed:
            self.button_press_start[button_name] = current_time
            return (True, False, False)
        
        # Button held
        elif is_pressed and was_pressed:
            duration = current_time - self.button_press_start[button_name]
            is_long_press = duration >= self.long_press_threshold
            return (True, is_long_press, False)
        
        # Button just released
        elif not is_pressed and was_pressed:
            duration = current_time - self.button_press_start[button_name]
            was_long_press = duration >= self.long_press_threshold
            return (False, False, True)
        
        # Button not pressed
        else:
            return (False, False, False)
    
    def check_for_state_transitions(self, inputs):
        """Check and process state transitions based on joystick inputs."""
        if self.state == "LOCKED":
            # In LOCKED state, check if throttle is in deadzone
            if abs(inputs["raw"]["thrust"]) < self.deadzone:
                self.throttle_in_deadzone = True
                
            # If throttle is in deadzone and RB is long-pressed, unlock
            if self.throttle_in_deadzone and inputs["rb_long_press"]:
                self.state = "UNLOCKED"
                print("System UNLOCKED! Joystick control enabled.")
                self.throttle_in_deadzone = False
                return "UNLOCK"
                
        elif self.state == "UNLOCKED":
            # From UNLOCKED, LB press enters auto mode
            if inputs["lb_pressed"]:
                self.state = "AUTO"
                print("Entered AUTO mode. Autonomous control active.")
                return "AUTO_ON"
                
            # From UNLOCKED, RB short press locks the system
            if inputs["rb_pressed"] and not inputs["rb_long_press"]:
                self.state = "LOCKED"
                print("System LOCKED! Center throttle and long press RB to unlock.")
                return "LOCK"
                
        elif self.state == "AUTO":
            # From AUTO, LB release exits auto mode
            if not inputs["lb_pressed"]:
                self.state = "UNLOCKED"
                print("Exited AUTO mode. Manual control restored.")
                return "AUTO_OFF"
                
        return None
    
    def get_control_inputs(self):
        """Get mapped control inputs from joystick with state management.
        
        Returns:
            dict: Mapped control values, state info, and state change command
        """
        if not self.initialized:
            return None
            
        pygame.event.pump()  # Update joystick state
        
        # Read raw axis values
        try:
            yaw_raw = self.joystick.get_axis(0)
            thrust_raw = self.joystick.get_axis(1)
            roll_raw = self.joystick.get_axis(2)
            pitch_raw = self.joystick.get_axis(3)
            
            # Apply deadzone
            yaw = self.apply_deadzone(yaw_raw)
            height_delta = self.apply_deadzone(thrust_raw)
            roll = self.apply_deadzone(roll_raw)
            pitch = self.apply_deadzone(pitch_raw)
            
            # Map yaw and roll/pitch same as before
            yaw_rate = yaw * (math.pi / 6)  # [-pi/2, pi/2]
            roll_angle = roll * (math.pi / 6)  # [-pi/6, pi/6]
            pitch_angle = -pitch * (math.pi / 6)  # [-pi/6, pi/6] (inverted)
            height_delta = -height_delta * 0.05  # Maximum 5cm per update at 10Hz = 50cm/s

            # Check buttons and track press durations
            a_raw = bool(self.joystick.get_button(0))
            b_raw = bool(self.joystick.get_button(1))
            lb_raw = bool(self.joystick.get_button(6))
            rb_raw = bool(self.joystick.get_button(7))
            
            # Track button press states and durations
            a_pressed, a_long_press, a_released = self.check_button_press('a', a_raw)
            b_pressed, b_long_press, b_released = self.check_button_press('b', b_raw)
            lb_pressed, lb_long_press, lb_released = self.check_button_press('lb', lb_raw)
            rb_pressed, rb_long_press, rb_released = self.check_button_press('rb', rb_raw)
            
            # Create input state
            inputs = {
                "yaw_rate": yaw_rate,
                "height_delta": height_delta,  # Height change instead of thrust
                "roll": roll_angle,
                "pitch": pitch_angle,
                "a_pressed": a_pressed,
                "a_long_press": a_long_press,
                "a_released": a_released,
                "b_pressed": b_pressed,
                "b_long_press": b_long_press,
                "b_released": b_released,
                "lb_pressed": lb_pressed,
                "lb_long_press": lb_long_press,
                "lb_released": lb_released,
                "rb_pressed": rb_pressed,
                "rb_long_press": rb_long_press,
                "rb_released": rb_released,
                "raw": {
                    "yaw": yaw_raw,
                    "thrust": thrust_raw,
                    "roll": roll_raw,
                    "pitch": pitch_raw
                },
                "state": self.state
            }
            
            # Check for state transitions
            state_change = self.check_for_state_transitions(inputs)
            inputs["state_change"] = state_change
            
            return inputs
            
        except pygame.error as e:
            print(f"Joystick error: {e}")
            return None

# Simple test function
if __name__ == "__main__":
    controller = JoystickController()
    if controller.initialize():
        try:
            print("Joystick control test. Press Ctrl+C to exit.")
            while True:
                inputs = controller.get_control_inputs()
                print(f"Yaw: {inputs['yaw_rate']:.2f}, Height Delta: {inputs['height_delta']:.2f}, "
                      f"Roll: {inputs['roll']:.2f}, Pitch: {inputs['pitch']:.2f}, "
                      f"A: {inputs['a_pressed']}({inputs['a_long_press']}), "
                      f"B: {inputs['b_pressed']}({inputs['b_long_press']}), "
                      f"State: {inputs['state']}, Change: {inputs['state_change']}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Test ended.")
        finally:
            controller.cleanup()
    else:
        print("No joystick detected.")
