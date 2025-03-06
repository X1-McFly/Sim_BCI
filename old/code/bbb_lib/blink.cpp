/*
    blink.cpp

    Repeatedly turns GPIO60 on and off in 1 second intervals.

    created 12 Nov 2024
    by Martin McCorkle

    This example code is in the public domain.
*/
#include "bbblib.hpp"

int main() {
    
    // Initialize gpio and set GPIO60 to output
    BBB::gpio gpio;
    gpio.set(60, OUTPUT);

    // Sets GPIO60 to inverse of current state and waits 1 second
    while (true) {
        gpio.write(60, HIGH);
        sleep(1);
        gpio.write(60, LOW);
        sleep(1);
    }   
}