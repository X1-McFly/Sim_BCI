# **Quron Cape**
**REV 0.1** 
### **Description**
The Quron Cape serves as the first version of the Quron Project, designed to function as an add-on board (cape) for the BeagleBone Black (BBB). This modular approach leverages the BBB’s computing power, connectivity options, and real-time capabilities to create a powerful brain-computer interface (BCI) platform. The Quron Cape simplifies the development process, enabling rapid prototyping while maintaining compatibility with existing BBB hardware and software ecosystems.

### **Design Goals for the Quron Cape**
    
- **Modular Add-On**: The cape integrates seamlessly with the BeagleBone Black, utilizing its GPIO pins and other I/O interfaces.
  
- **EEG Signal Processing**: Captures brainwave data using onboard ADCs, optimized for real-time neural processing.
  
- **Reduced Complexity**: Leverages the BBB’s bootloader and operating system, minimizing the need for additional custom components.
  
- **Cost-Effective Prototyping**: Builds on widely available, affordable hardware to lower initial development costs.
  
- **Expandable Platform**: Allows further experimentation by connecting additional peripherals through I2C, SPI, UART, and GPIO.

### **Hardware Overview**
    
- **Processor**: Utilizes the BeagleBone Black’s AM335x ARM Cortex-A8.
  
- **Power Supply**: Powered through the BBB, supporting micro-USB and DC barrel jack inputs.
  
- **Interfaces**: GPIO, I2C, SPI, UART for communication with peripherals.
  
- **Analog Input**: Captures EEG signals through BBB’s onboard ADCs for real-time processing.
  
- **Onboard Components**: Essential filtering and amplification circuits for EEG signal conditioning.


### **Software Overview**

- **Operating System**: Runs on the BBB’s custom Linux image (Debian 12.2).
  
- **Machine Learning**: Integrates TensorFlow Lite for local neural processing and inference.
    
- **Remote Access**: Supports SSH and web-based monitoring via the BBB’s Ethernet interface.
    
- **Storage Options**: Logs and data stored on microSD or eMMC for analysis.
