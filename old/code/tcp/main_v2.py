import socket
import time

TCP_IP = '192.168.7.2'  # BBB's IP address over USB
TCP_PORT = 5005

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the BBB server
sock.connect((TCP_IP, TCP_PORT))
print(f"Connected to {TCP_IP}:{TCP_PORT}")

buffer = ""  # Buffer to store partial messages

try:
    # Continuously receive data from the server
    while True:
        data = sock.recv(512).decode('utf-8')  # Receive and decode data
        if not data:
            print("Connection closed by server.")
            break

        # Add received data to the buffer
        buffer += data

        # Process all complete messages ending with '\n'
        while '\n' in buffer:
            message, buffer = buffer.split('\n', 1)  # Split at the first newline
            print(f"Received: {message.strip()}")  # Print the complete message

        # Small delay to avoid busy-waiting
        time.sleep(0.001)
finally:
    sock.close()
    print("Connection closed.")
