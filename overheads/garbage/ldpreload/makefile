all: build

build: 
	gcc -shared -fPIC -o test.so test.c
test: 
	LD_PRELOAD=./test.so julia -e 'using Pkg; Pkg.activate("../../"); include("../../examples/gray-scott.jl")'  2> stderr.txt
clean:
	rm *.so
	rm *.txt
	rm *.log