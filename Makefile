# Builds every */*.cu into bin/<name>. Usage: make (all) | make bin/transpose | make clean
NVCC  ?= nvcc
FLAGS ?= -O3 -arch=native -std=c++17 -lineinfo -Icommon

SRCS    := $(wildcard */*.cu)
TARGETS := $(patsubst %.cu,bin/%,$(notdir $(SRCS)))
vpath %.cu $(sort $(dir $(SRCS)))

all: $(TARGETS)

bin/%: %.cu common/common.cuh $(wildcard */*.cuh)
	@mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@ $(LDLIBS)

# Per-target libraries.
bin/matmul_cublas: LDLIBS = -lcublas

clean:
	rm -rf bin

.PHONY: all clean
