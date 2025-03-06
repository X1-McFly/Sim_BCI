#include <stdint.h>

// PRU GPIO registers
volatile register uint32_t __R30;  // Output register
volatile register uint32_t __R31;  // Input register

#define GPIO_PIN 0x1  // Use R30 bit 0 (e.g., P8_11)

void main(void) {
    while (1) {
        __R30 ^= GPIO_PIN;  // Toggle the GPIO pin
        volatile int i;     // Declare the loop variable
        for (i = 0; i < 100000; i++) {
            // Simple delay loop
        }
    }
}
