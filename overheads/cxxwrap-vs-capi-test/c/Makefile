all: libarray_allocator.so

libarray_allocator.so: array.c
	gcc -O3 -march=native -flto -fPIC -Wall -Wextra -funroll-loops -fopenmp -shared -o libarray_allocator.so array.c

clean:
	rm -f libarray_allocator.so
