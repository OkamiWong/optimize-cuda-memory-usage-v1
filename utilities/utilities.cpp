#include "utilities.hpp"

#include <chrono>

void SystemWallClock::start() {
  this->startTimePoint = std::chrono::system_clock::now();
}

void SystemWallClock::end() {
  this->endTimePoint = std::chrono::system_clock::now();
}

float SystemWallClock::getTimeInSeconds() {
  std::chrono::duration<float> duration = this->endTimePoint - this->startTimePoint;
  return duration.count();
}
