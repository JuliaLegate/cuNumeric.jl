#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

// Function to allocate an array of random doubles and return the pointer
double* allocate_array(int size) {
    double* arr = (double*)malloc(size * sizeof(double));
    return arr;
}



void set(double* arr, int index, double val) {
    arr[index] = val;
}