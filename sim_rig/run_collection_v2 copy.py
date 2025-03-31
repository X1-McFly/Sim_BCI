import pyvjoy
import time
import math

# Create a virtual joystick device (this is Joystick #1)
joystick = pyvjoy.VJoyDevice(1)

# Mapping throttle and brake pedal to the joystick
def map_pedal_to_vjoy(pedal_position):
    """
    Map pedal position to DirectInput throttle and brake values.
    - pedal_position: between -1 (full brake) and 1 (full throttle)
    """
    if pedal_position >= 0:
        throttle = int(pedal_position * 32767)  # Throttle: 0 to 32767
        brake = 0  # No brake
    else:
        throttle = 0  # No throttle
        brake = int(abs(pedal_position) * 32767)  # Brake: 0 to 32767
    
    return throttle, brake

# Function to update virtual joystick
def update_joystick(throttle, brake, steering):
    """
    Update the virtual joystick values (steering, throttle, brake).
    - throttle: throttle input from -1 to 1
    - brake: brake input from -1 to 1
    - steering: steering input from -1 to 1
    """
    # Convert throttle, brake, and steering to the range expected by DirectInput
    joystick.set_axis(pyvjoy.HID_USAGE_X, int(steering * 32767))  # Steering (left-right)
    joystick.set_axis(pyvjoy.HID_USAGE_Y, int(throttle * 32767))  # Throttle (forward)
    joystick.set_axis(pyvjoy.HID_USAGE_Z, int(brake * 32767))    # Brake (backward)

# Main loop to simulate the input
def main_loop():
    while True:
        # Simulate some pedal positions (throttle/brake) and steering
        throttle_position = 0.5  # Accelerating
        brake_position = -0.2   # Applying brake slightly
        steering_position = 0.0 # Centered steering

        # Map pedal position to throttle and brake values
        throttle, brake = map_pedal_to_vjoy(throttle_position)

        # Update the virtual joystick
        update_joystick(throttle, brake, steering_position)
        
        # Log for debugging
        print(f"Throttle: {throttle}, Brake: {brake}, Steering: {steering_position}")

        # Wait a bit before sending the next update
        time.sleep(1/60)  # Updates at ~60Hz (game polling frequency)

if __name__ == "__main__":
    main_loop()
