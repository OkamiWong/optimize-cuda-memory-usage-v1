#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>

#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)

int __log_trace(const char* file, const char* function, const char* fmt, ...) {
  printf("[%s/%s] ", file, function);

  va_list args;
  va_start(args, fmt);
  int result = vprintf(fmt, args);
  va_end(args);

  printf("\n");
  return result;
}

#define LOG_TRACE() __log_trace(__FILENAME__, __func__, "")

#define LOG_TRACE_WITH_INFO(...) __log_trace(__FILENAME__, __func__, __VA_ARGS__)
