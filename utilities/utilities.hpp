#pragma once

#include <chrono>
#include <cstdlib>

template <typename T>
void fillRandomEntries(T *matrix, int m, int n, int lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i * lda + j] = 2 * static_cast<T>(drand48()) - 1;
    }
  }
}

class SystemWallClock {
 public:
  void start();
  void end();
  float getTimeInSeconds();

 private:
  std::chrono::time_point<std::chrono::system_clock> startTimePoint, endTimePoint;
};
