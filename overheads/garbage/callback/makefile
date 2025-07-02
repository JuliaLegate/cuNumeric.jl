all: build

build:
	g++ -shared -fPIC -o libcallback.so callback.cc  -pthread -ldl -I.
	#g++ -shared -o libcallback.so callback.cc -fPIC -I$(shell julia -e 'print(joinpath(Sys.BINDIR, "..", "include", "julia"))') -L$(shell julia -e 'print(Sys.BINDIR)')/../lib/ -ljulia -pthread -ldl -I.
clean:
	rm *.so
