#include "bbblib.hpp"
#include <chrono>
#include <thread>
#include <random>

int main() {
    BBB::gpio gpio;
    BBB::analog analog;
    BBB::ICP icp;

    std::cout << "Transmitting Data..." << std::endl;
    std::vector<bool> select(4, false);
        
    const int target_frequency = 200; // Target frequency in Hz
    const std::chrono::nanoseconds target_duration(1'000'000'000 / target_frequency); // Target duration per loop

    while (true) {
        // Start time of the loop
        auto start = std::chrono::high_resolution_clock::now();

        // Simulate your loop logic
        std::string buffer = "";
        for (int i = 0; i < 16; i++) {
            if(i < 8) {
                buffer += std::to_string(analog.read(i)) + " ";
            } else {
                buffer += std::to_string(analog.read(i/2)) + " ";
            }
            
            select.at(0) = i & 1;
            select.at(1) = i & 2;
            select.at(2) = i & 4;
            select.at(3) = i & 8;

            gpio.write(60, select.at(0));
            gpio.write(3, select.at(1));
            gpio.write(48, select.at(2));
            gpio.write(4, select.at(3));

            // simulateCalculations();

        }
        icp.send(&buffer);

        // End time of the loop
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration of the loop
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Calculate the remaining time to sleep
        auto sleep_duration = target_duration - elapsed;

        if (sleep_duration > std::chrono::nanoseconds::zero()) {
            // Sleep to cap the loop to the target frequency
            std::this_thread::sleep_for(sleep_duration);
        }

        // Recalculate the actual elapsed time after sleeping
        auto actual_end = std::chrono::high_resolution_clock::now();
        auto actual_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(actual_end - start);

        // Output elapsed time and FPS for debugging
        std::cout << "Elapsed: " << actual_elapsed.count() / 1'000'000.0 << " ms\t";
        std::cout << "S/s: " << (1'000'000'000.0 / actual_elapsed.count()) << std::endl;
    }

}
