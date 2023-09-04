#include "logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>

int __log_trace(const char* file, const char* function, const char* fmt, ...) {
  printf("[%s/%s] ", file, function);

  va_list args;
  va_start(args, fmt);
  int result = vprintf(fmt, args);
  va_end(args);

  printf("\n");
  return result;
}
