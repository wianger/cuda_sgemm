main: main.cu
	nvcc -O3 -arch=sm_75 main.cu -o build/main
clean:
	rm -f build/*
run: main
	./build/main