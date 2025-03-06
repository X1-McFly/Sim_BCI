# client.py
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.7.2', 12345))
client.sendall(b"Hello, BeagleBone!")

while(1):
	data = client.recv(1024)
	print("Received:", data.decode())

client.close()
