#pragma once

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>

namespace memopt {

class SystemWallClock {
 public:
  void start();
  void end();
  float getTimeInSeconds();
  float peekCurrentTimeInSeconds();
  void logWithCurrentTime(const char *message);

 private:
  std::chrono::time_point<std::chrono::system_clock> startTimePoint, endTimePoint;
};

template <typename T>
std::string toStringWithPrecision(const T a_value, const int n = 6) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return std::move(out).str();
}

int generateRandomInteger(int min, int max);

}  // namespace memopt
