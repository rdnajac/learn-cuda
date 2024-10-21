# Makefile for compiling .cu files into executables

# Find all .cu files
CU_FILES := $(wildcard *.cu)

# Generate executable names by removing the .cu extension
EXECUTABLES := $(CU_FILES:.cu=)

# Default target
all: $(EXECUTABLES)

# Rule to compile each .cu file into an executable
%: %.cu
	nvcc -o $@ $<

# Clean up the generated executables
clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean

