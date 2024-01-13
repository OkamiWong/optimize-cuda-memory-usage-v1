#include "utilities.hpp"

#include <chrono>
#include <random>

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

float SystemWallClock::peekCurrentTimeInSeconds() {
  auto currentTimePoint = std::chrono::system_clock::now();
  std::chrono::duration<float> duration = currentTimePoint - this->startTimePoint;
  return duration.count();
}

void SystemWallClock::logWithCurrentTime(const char *message) {
  printf("(%.6f) %s\n", this->peekCurrentTimeInSeconds(), message);
}

int generateRandomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution distrib(min, max);
  return distrib(gen);
}
