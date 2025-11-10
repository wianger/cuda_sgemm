main: main.cu
	nvcc -O3 main.cu -o build/main
clean:
	rm -f build/main
run: main
	./build/main