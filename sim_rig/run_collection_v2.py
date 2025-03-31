import sys
sys.path.append('../logidrivepy')
from logidrivepy import LogitechController
import time
import datetime

def spin_controller(controller):
    for i in range(-100, 102, 2):
        controller.LogiPlaySpringForce(0, i, 100, 40)
        controller.logi_update()
        time.sleep(0.08)

def get_wheel_state(controller):
    controller.logi_update()
    state_pointer = controller.LogiGetStateENGINES(0)
    state = state_pointer.contents
    return { "Steering": state.lX, "Throttle": state.lY, "Brake": state.lRz, "Timestamp": datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S:%f") }

def spin_test():
    controller = LogitechController()
    controller.steering_initialize()
    print("\n---Logitech Spin Test---")
    spin_controller(controller)
    print("Spin test passed.\n")
    controller.steering_shutdown()

def save_to_csv(data, filename):
    with open(filename, 'a') as f:
        f.write(f"{data['Steering']}, {data['Throttle']}, {data['Brake']}, {data['Timestamp']}\n")

def write_csv_header(filename):
    with open(filename, 'w') as f:
        f.write("Steering, Throttle, Brake, Timestamp\n")

def map_pedal_to_throttle_brake(pedal_position):
    """ Map the pedal position to throttle and brake values.
        - Positive pedal values = accelerate
        - Negative pedal values = brake (with throttle zero)
    """
    if pedal_position > 0:
        throttle = pedal_position  # Accelerating
        brake = 0                  # No braking
    else:
        throttle = 0               # No acceleration
        brake = -pedal_position    # Braking (positive brake value as pedal is released)
    
    # Make sure the throttle and brake values stay within valid ranges
    throttle = max(-32768, min(32767, throttle))
    brake = max(-32768, min(32767, brake))
    
    return throttle, brake

if __name__ == "__main__":
    controller = LogitechController()
    # spin_test()

    if not controller.steering_initialize():
        print("Failed to initialize the controller.")
    
    # Create a new file with a timestamp and write the headers
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wheel_data_{timestamp}.csv"
    write_csv_header(filename)

    # SteeringPosition 0 is center,  -32768 is full left, 32767 is full right
    # ThrottlePosition 0 # 32767 is no throttle, -32768 is full throttle
    # BrakePosition 0 # 32767 is no brake, -32768 is full brake

    while True:
        wheel_state = get_wheel_state(controller)
        
        # Get the pedal position (throttle) and apply the combined acceleration/braking logic
        pedal_position = wheel_state["Throttle"]  # Assuming throttle corresponds to pedal position
        throttle, brake = map_pedal_to_throttle_brake(pedal_position)
        
        # Update the wheel state with the calculated throttle and brake
        wheel_state["Brake"] = -throttle
        wheel_state["Throttle"] = -brake
        
        print(wheel_state)
        save_to_csv(wheel_state, filename)
        time.sleep(0.01)
