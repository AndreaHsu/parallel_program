CFLAGS = -O3 -lm -fopenmp
TARGETS = hw3-1 hw3-2 hw3-3
NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm

alls: $(TARGETS)

clean:
	rm -f $(TARGETS)

hw3-1: hw3-1.cc
	g++ $(CFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) -Xcompiler="-fopenmp" -o $@ $?
