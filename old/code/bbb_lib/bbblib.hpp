#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT 0
#define INPUT_PULLUP 2

#ifndef BBB_PINS
#define BBB_PINS

#define AIN0 0
#define AIN1 1
#define AIN2 2
#define AIN3 3
#define AIN4 4
#define AIN5 5
#define AIN6 6
#define AIN7 7

#endif


#ifndef BBB_H
#define BBB_H

#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

class BBB {

    public:

    // BBB();

    class gpio {

        std::string gpioFileDir = "/sys/class/gpio/gpio";

        public:
        // int pull_up();
        int read(int pin);
        int write(int pin, bool value);
        int set(int pin, bool value);

        enum pins {
            GPIO60 = 60,
            GPIO3 = 3,
        };

    };

    class analog {
        
        std::string analogFileDir = "/sys/bus/iio/devices/iio:device0/";

        public:
        int read(int pin);

    };

    class ICP {

        int *sockfd = new int; // socket file descriptor
        struct sockaddr_in servAddr; 

        public:
        ICP();
        ~ICP();

        // int create();
        // int connect();
        int send(const std::string *data);
        // int receive();

    };

};

#endif

#ifndef BBB_SOCKET

#define IPV4_ADDR "127.0.0.1"
#define PORT 50505

#endif