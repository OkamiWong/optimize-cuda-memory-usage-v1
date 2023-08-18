#pragma once

#include <cstdio>

namespace logger {
void log(const char* const file, const char* const function, const char* const info = nullptr) {
  printf("[%s/%s]", file, function);
  if (info != nullptr) {
    printf(" %s\n", info);
  } else {
    printf("\n");
  }
}
}  // namespace logger

#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)

#define LOG_TRACE() logger::log(__FILENAME__, __func__)

#define LOG_TRACE_WITH_INFO(info) logger::log(__FILENAME__, __func__, info)
