# arm-linux-gnueabihf-g++ electrodes.cpp bbblib.cpp -o electrodes.out
# scp electrodes.out main.py run.sh debian@192.168.7.2:/home/debian/remote_code

arm-linux-gnueabihf-g++ blink.cpp bbblib.cpp -o blink.out
scp blink.out debian@192.168.7.2:/home/debian/remote_code

# arm-linux-gnueabihf-g++ main.cpp bbblib.cpp -o main.out
# scp main.cpp bbblib.cpp bbblib.hpp debian@192.168.7.2:/home/debian/remote_code
