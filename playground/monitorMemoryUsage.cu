#include <fmt/core.h>

#include <iostream>

#include "../utilities/cudaUtilities.hpp"

float toMiB(size_t s) {
  return static_cast<float>(s) / 1024.0 / 1024.0;
}

void printMemInfo() {
  size_t free, total;
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  fmt::print(
    "free = {:.2f}\ntotal = {:.2f}\n\nused = {:.2f}\n",
    toMiB(free),
    toMiB(total),
    toMiB(total - free)
  );
}

constexpr size_t BYTES_TO_ALLOC = 1024 * 1024 * 1024;

int main() {
  std::string s;

  fmt::print("Start\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Allocate pinned memory\n");
  void *dev_p;
  checkCudaErrors(cudaMalloc(&dev_p, BYTES_TO_ALLOC));

  fmt::print("Allocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Deallocate\n");
  checkCudaErrors(cudaFree(dev_p));

  fmt::print("Deallocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Allocate unified memory\n");
  checkCudaErrors(cudaMallocManaged(&dev_p, BYTES_TO_ALLOC));
  checkCudaErrors(cudaMemPrefetchAsync(dev_p, BYTES_TO_ALLOC, 0));

  fmt::print("Allocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Deallocate\n");
  checkCudaErrors(cudaFree(dev_p));

  fmt::print("Deallocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  return 0;
}