#include <stdio.h>

int main() {
    unsigned int x = 0x12345678;
    unsigned char *ptr = (unsigned char *)&x;
    
    if (*ptr == 0x78) {
        printf("Little Endian\n");
    } else if (*ptr == 0x12) {
        printf("Big Endian\n");
    } else {
        printf("Unknown Endian\n");
    }

    return 0;
}
