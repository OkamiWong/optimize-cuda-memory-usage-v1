.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -D CMAKE_BUILD_TYPE=Debug

.PHONY: config-release
config-release:
	cmake -S . -B ./build -D CMAKE_BUILD_TYPE=Release

.PHONY: build
build:
	cmake --build ./build

.PHONY: clean
clean:
	rm -rf ./build
