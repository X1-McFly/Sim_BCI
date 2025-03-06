ti-cgt-pru_2.3.3/bin/clpru \
    --include_path=include \
    --include_path=/path/to/pru-software-support-package/include/am335x \
    --output_file=/lib/firmware/am335x-pru0-fw blink.c


ti-cgt-pru_2.3.3/bin/clpru --include_path=ti-cgt-pru_2.3.3/include \
    --include_path=ti-cgt-pru_2.3.3/include/am335x \
    --output_file=code/blink.out code/blink.c

scp code/blink.out debian@192.168.7.2:/home/debian
