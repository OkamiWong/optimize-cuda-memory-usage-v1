#pragma once

#include <chrono>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>

namespace memopt {

class ScopeGuard {
 public:
  ScopeGuard() = default;
  ScopeGuard(const std::function<void()>& callback) : m_callback{callback} {}
  ScopeGuard(std::function<void()>&& callback) : m_callback{std::move(callback)} {}
  ScopeGuard& operator=(const ScopeGuard& other) = delete;
  ScopeGuard(const ScopeGuard& other) = delete;
  ScopeGuard& operator=(ScopeGuard&& other) {
    std::swap(m_callback, other.m_callback);
    return *this;
  }
  ScopeGuard(ScopeGuard&& other) { *this = std::move(other); }
  ~ScopeGuard() {
    if (m_callback) {
      m_callback();
    }
  }

  void disarm() {
    m_callback = {};
  }

 private:
  std::function<void()> m_callback;
};

class SystemWallClock {
 public:
  void start();
  void end();
  float getTimeInSeconds();
  float peekCurrentTimeInSeconds();
  void logWithCurrentTime(const char* message);

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
