CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm# -L/opt/ipm_mpi/lib -lipm
TARGETS = hw1

.PHONY: all
all: 
	$(CXX) $(CXXFLAGS) -o $(TARGETS) hw1.cc

.PHONY: clean
clean:
	rm -f $(TARGETS)

