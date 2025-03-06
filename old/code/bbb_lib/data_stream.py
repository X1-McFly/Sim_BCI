#   data_stream.py
#
#   Creates and connects to the BeagleBone Black's local
#   UDP socket for inter-process communication.
#
#   created 12 Nov 2024
#   by Martin McCorkle
#
#   This example code is in the public domain.

import socket

def udp_server(host="127.0.0.1", port=50505):
    try:
        # Create a UDP socket
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the specified address and port
        udp_sock.bind(host, port)

        while True:
            # Receive data from any client
            data, addr = udp_sock.recvfrom(1024)  # Buffer size is 1024 bytes
            print(f"{data.decode()}")
    
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        udp_sock.close()
