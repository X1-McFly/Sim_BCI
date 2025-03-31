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

if __name__ == "__main__":
    controller = LogitechController()
    spin_test()

    if not controller.steering_initialize():
        print("Failed to initialize the controller.")
    
    # Create a new file with a timestamp and write the headers
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wheel_data_{timestamp}.csv"
    write_csv_header(filename)
    
    # SteeringPosition 0 is center, -32768 is full left, 32767 is full right
    # ThrottlePosition 0 # 32767 is no throttle, -32768 is full throttle
    # BrakePosition 0 # 32767 is no brake, -32768 is full brake
    while True:
        wheel_state = get_wheel_state(controller)
        
        print(wheel_state)
        save_to_csv(wheel_state, filename)
        time.sleep(0.01)