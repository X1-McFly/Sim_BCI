#include "bbblib.hpp"

// BBB() {

// };


// Reads the value of the targetted GPIO pin.
int BBB::gpio::read(int pin) {
    
    std::string filename = gpioFileDir + std::to_string(pin) + "/value";

        std::ifstream file(filename);
        std::string value;

        if (file.is_open()) {

            file >> value;
            file.close();

            if (value == "0" || value == "1") {

                return std::stoi(value);
                //std::cout << value << std::endl;

            } else {

                std::cout << "Unknown value: " << value << std::endl;

            }
        } else {

            std::cerr << "Error opening file!" << std::endl;
            return -1;

        }

    return 0;
}

// Writes a 1 or 0 to the GPIO pin.
int BBB::gpio::write(int pin, bool value) {
    
    std::string filename = gpioFileDir + std::to_string(pin) + "/value";
    std::ofstream file(filename);

    if (file.is_open()) {

        file << (value ? 1 : 0);  // Write 1 if true, 0 if false
        file.close();

    } else {

        std::cerr << "Failed to open the file." << std::endl;

    }

    return 0;

}

// Sets the GPIO direction, True(1) = OUTPUT, False(0) = INPUT
int BBB::gpio::set(int pin, bool value) {
    
    std::string filename = gpioFileDir + std::to_string(pin) + "/direction";

    std::ofstream file(filename);

    if (file.is_open()) {

        file << (value ? "out" : "in");  // Write 1 if true, 0 if false
        file.close();

    } else {

        std::cerr << "Failed to open the file." << std::endl;

    }

    return 0;
}

int BBB::analog::read(int pin) {
    
    std::string filename = analogFileDir + "in_voltage" + std::to_string(pin) + "_raw";

    std::ifstream file(filename);
    std::string value;

    if (file.is_open()) {
        
        file >> value;
        file.close();

    } else {

        std::cerr << "Error opening file!" << std::endl;
        return -1;

    }

    return std::stoi(value);

}

BBB::ICP::ICP() {

    if ((*sockfd = socket(AF_INET, SOCK_DGRAM, 0)) <= 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return;
    } else {
        // std::cout << "Socket Initialized..." << std::endl;
    }

    int broadcast = 1;
    if (setsockopt(*sockfd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
        std::cerr << "Failed to set broadcast option" << std::endl;
        close(*sockfd);
        return;
    }

    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, IPV4_ADDR, &servAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address/Address not supported" << std::endl;
        close(*sockfd);
        return;
    }

}

BBB::ICP::~ICP() {
    close(*sockfd);
    delete sockfd;
}

// int BBB::ICP::connect() {
//     if(::connect(*sockfd, (struct sockaddr*)&servAddr, sizeof(servAddr)) < 0) {
//         std::cerr << "Could not connect" << std::endl;
//         close(*sockfd);
//         return -1;
//     } else {
//         std::cout << "Connected to: " << IPV4_ADDR << ", PORT: " << PORT << std::endl;
//         return 0;
//     }
// }

int BBB::ICP::send(const std::string *data) {
    if (*sockfd <= 0) {
        std::cerr << "Socket not initialized properly" << std::endl;
        return -1;
    }

    ssize_t sent_bytes = sendto(*sockfd, data->c_str(), data->size(), 0,
                                (struct sockaddr*)&servAddr, sizeof(servAddr));
    if (sent_bytes < 0) {
        std::cerr << "Failed to send broadcast message" << std::endl;
        return -1;
    }

    // std::cout << "Sending Data... " << std::endl;
    return 0;
}