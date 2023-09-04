#include <assert.h>

__global__ void testAssert(void) {
  int is_one = 1;
  int should_be_one = 0;
  // This will have no effect
  assert(is_one);
  // This will halt kernel execution
  assert(should_be_one);
}

int main(int argc, char* argv[]) {
  testAssert<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
