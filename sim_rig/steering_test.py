import sys
import time
import datetime
import keyboard  # Make sure to install this with pip
from logidrivepy import LogitechController

# Initialize the controller
controller = LogitechController()

# def calculate_steering():
#     time.sleep(0.01)


# Function to spin the controller
def spin_controller(controller):
    for i in range(-100, 102, 2):
        controller.LogiPlaySpringForce(0, i, 100, 40)
        controller.logi_update()
        time.sleep(0.08)

# Function to get the wheel state
def get_wheel_state(controller):
    controller.logi_update()
    state_pointer = controller.LogiGetStateENGINES(0)
    state = state_pointer.contents
    return {
        "Steering": state.lX,
        "Throttle": state.lY,
        "Brake": state.lRz,
        "Timestamp": datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S:%f")
    }

# Function to spin test
def spin_test():
    controller.steering_initialize()
    print("\n---Logitech Spin Test---")
    spin_controller(controller)
    print("Spin test passed.\n")
    controller.steering_shutdown()

# Function to set wheel position using keyboard input
def set_steering_position():
    current_position = 0  # Center position of the wheel

    interval = 5

    while True:
        if keyboard.is_pressed('left'):
            # Move wheel left
            current_position -= interval  # Adjust for more/less movement
            current_position = max(current_position, -100)  # Prevent going beyond max left
            controller.LogiPlaySpringForce(0, current_position, 100, 40)
            controller.logi_update()
            print(f"Steering left: {current_position}")

        elif keyboard.is_pressed('right'):
            # Move wheel right
            current_position += interval  # Adjust for more/less movement
            current_position = min(current_position, 100)  # Prevent going beyond max right
            controller.LogiPlaySpringForce(0, current_position, 100, 40)
            controller.logi_update()
            print(f"Steering right: {current_position}")

        elif keyboard.is_pressed('up'):
            # Reset to center position
            current_position = 0
            controller.LogiPlaySpringForce(0, current_position, 100, 40)
            controller.logi_update()
            print("Steering center")

        time.sleep(0.1)  # Poll for key press every 100ms

# Main function
if __name__ == "__main__":
    controller.steering_initialize()

    # if not controller.steering_initialize():
    #     print("Failed to initialize the controller.")
    #     sys.exit()

    print("\n---Controller Initialized---")
    print("Use arrow keys to move the steering wheel. Press 'Up' to center.")

    # Run the keyboard-controlled steering position
    try:
        while True:
            wheel_state = get_wheel_state(controller)

            print(wheel_state)

            set_steering_position()  # Allow the user to change wheel position with the keyboard

            time.sleep(0.01)  # Logging interval

    except KeyboardInterrupt:
        print("Exiting...")
        controller.steering_shutdown()
        sys.exit()
