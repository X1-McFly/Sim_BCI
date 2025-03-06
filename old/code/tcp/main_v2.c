#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SERVER_IP "192.168.7.2"  // BBB's IP address
#define SERVER_PORT 5005         // TCP port
#define BUFFER_SIZE 512          // Chunk size for receiving data

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE + 1];  // Buffer for receiving data (+1 for null terminator)
    char message_buffer[4096] = "";  // Store partial messages (4 KB buffer)

    // Create a TCP socket
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);

    // Convert IP address from string to binary form
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // Connect to the server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to server failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    printf("Connected to %s:%d\n", SERVER_IP, SERVER_PORT);

    // Continuously receive data from the server
    while (1) {
        memset(buffer, 0, sizeof(buffer));  // Clear buffer
        ssize_t bytes_received = recv(sockfd, buffer, BUFFER_SIZE, 0);  // Receive data

        if (bytes_received < 0) {
            perror("Receive failed");
            break;
        } else if (bytes_received == 0) {
            printf("Connection closed by server.\n");
            break;
        }

        buffer[bytes_received] = '\0';  // Null-terminate the received data

        // Add received data to message buffer
        strncat(message_buffer, buffer, sizeof(message_buffer) - strlen(message_buffer) - 1);

        // Process all complete messages ending with '\n'
        char *newline_pos;
        while ((newline_pos = strchr(message_buffer, '\n')) != NULL) {
            *newline_pos = '\0';  // Replace newline with null terminator

            // Print the complete message
            printf("Received: %s\n", message_buffer);

            // Shift remaining data to the start of the buffer
            memmove(message_buffer, newline_pos + 1, strlen(newline_pos + 1) + 1);
        }

        // Small delay to avoid busy-waiting (optional)
        usleep(1000);  // 1 ms delay
    }

    // Close the socket
    close(sockfd);
    printf("Connection closed.\n");

    return 0;
}
