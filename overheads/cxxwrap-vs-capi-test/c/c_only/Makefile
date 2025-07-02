all: test

test: main.c
	gcc -O3 -march=native -flto -fPIC -Wall -Wextra -funroll-loops -fopenmp main.c -L../ -larray_allocator -Wl,-rpath=../ -o test

clean:
	rm -f test