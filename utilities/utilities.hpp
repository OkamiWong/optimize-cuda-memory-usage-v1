#pragma once

#include <chrono>
#include <cstdlib>

class SystemWallClock {
 public:
  void start();
  void end();
  float getTimeInSeconds();

 private:
  std::chrono::time_point<std::chrono::system_clock> startTimePoint, endTimePoint;
};

int generateRandomInteger(int min, int max);
