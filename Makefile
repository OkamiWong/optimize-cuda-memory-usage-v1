# Build
.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: config-release
config-release:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: config
config: config-release

.PHONY: build
build:
	cmake --build ./build -j

.PHONY: build-sequential
build-sequential:
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
