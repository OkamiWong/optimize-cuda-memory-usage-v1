#pragma once

int __log_trace(const char* file, const char* function, const char* fmt, ...);

#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)

#define LOG_TRACE() __log_trace(__FILENAME__, __func__, "")

#define LOG_TRACE_WITH_INFO(...) __log_trace(__FILENAME__, __func__, __VA_ARGS__)
