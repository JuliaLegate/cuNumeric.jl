#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

double* allocate_array(int size);

int main()  {
    volatile double *array;
    for (uint64_t i = 0; i < 1000; i++) {
        array = allocate_array(500000);
        array[0] = 1.0;
    }
    return 0;
}