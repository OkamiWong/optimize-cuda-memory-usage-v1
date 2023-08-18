# Build
.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -D CMAKE_BUILD_TYPE=Debug

.PHONY: config-release
config-release:
	cmake -S . -B ./build -D CMAKE_BUILD_TYPE=Release

.PHONY: config
config: config-release

.PHONY: build
build:
	cmake --build ./build

.PHONY: build-verbose
build-verbose:
	cmake --build ./build --verbose

.PHONY: clean
clean:
	rm -rf ./build

.PHONY: run
run:
	./build/playground/helloWorld

# Profile peak memory usage
.PHONY: profile-pmu
profile-pmu:
	CUDA_INJECTION64_PATH=$(CURDIR)/build/utilities/libinjectedPeakMemoryProfiler.so $(TARGET)