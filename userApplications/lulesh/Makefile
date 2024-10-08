VCPKG_INCLUDE_PATH=/home/twang/sources/projects/optimize-cuda-memory-usage-v1/build/vcpkg_installed/x64-linux/include
MEMOPT_INCLUDE_PATH=/home/twang/sources/projects/optimize-cuda-memory-usage-v1/public

CUDA_PATH=/usr/local/cuda-12.3
NVCC=/home/twang/sources/projects/llvm-project2/build/bin/clang++
NVCC_FLAGS=-g -O3 -I. -I$(VCPKG_INCLUDE_PATH) -I$(MEMOPT_INCLUDE_PATH) --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_80 -mllvm --memred-mode=trace

all: compiler-pass

compiler-pass: lulesh.cu
	@echo "Analyzing lulesh.cu"
	$(NVCC) -c $(NVCC_FLAGS) -o lulesh.o lulesh.cu
	rm lulesh.o
	mv .memred.memory.analysis.out ../..
