import socket
import struct
import time

# UDP settings
HOST = '127.0.0.1'  # Listen on all available network interfaces
PORT = 4242       # The port to listen on

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

# Define the packet size (ACC UDP telemetry packets are typically 1460 bytes)
BUFFER_SIZE = 10000

print(f"Listening for telemetry data on {HOST}:{PORT}...")

# This function will be used to parse the raw telemetry data
def parse_telemetry(data):
    # ACC's UDP telemetry data is packed in a specific format.
    # The structure and data types can be found in the ACC telemetry documentation.
    # Example: parsing a specific structure that contains the car's speed and position:
    
    # Assuming you're dealing with raw data that might include car telemetry info:
    # For the sake of example, we'll unpack a simplified structure:
    # This is just a sample, you'll need to refer to the actual ACC telemetry documentation for precise fields.
    
    try:
        # Example: unpacking a telemetry data block (simplified)
        telemetry = struct.unpack('<f f f f f', data[:20])  # Adjust structure to match actual data
        car_speed, car_pos_x, car_pos_y, car_pos_z, car_rot = telemetry
        print(f"Speed: {car_speed} | Position: ({car_pos_x}, {car_pos_y}, {car_pos_z}) | Rotation: {car_rot}")
    except Exception as e:
        print(f"Error parsing telemetry data: {e}")

# Main loop to listen for incoming telemetry data
while True:
    try:
        # Receive data (blocking call, it will wait for the telemetry packet)
        data, addr = sock.recvfrom(BUFFER_SIZE)
        
        # Parse and print the telemetry data
        print(struct.unpack(data))
        
        # Sleep for a short period to prevent CPU overload (adjust if necessary)
        time.sleep(0.01)
        
    except KeyboardInterrupt:
        print("Telemetry listening stopped.")
        break
