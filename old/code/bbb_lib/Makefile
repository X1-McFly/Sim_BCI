# Define variables
CC = g++                  # Compiler
CFLAGS = -Wall            # Compiler flags (optional, can add more as needed)
SRC = main.cpp bbblib.cpp  # Source files
OUT = main.out            # Output binary file
BB_HOST = 192.168.7.2      # BeagleBone host (replace with your IP or hostname)
BB_USER = debian            # Username (typically root)
BB_PATH = /home/debian      # Destination path on BeagleBone

# Default target
all: $(OUT) transfer

# Compile and link the code
$(OUT): $(
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

# Transfer the binary to BeagleBone Black
transfer:
	scp $(OUT) $(BB_USER)@$(BB_HOST):$(BB_PATH)

# Clean the generated files
clean:
	rm -f $(OUT)

# Run the compiled code on BeagleBone Black (optional)
run:
	ssh $(BB_USER)@$(BB_HOST) '$(BB_PATH)/$(OUT)'

