#pragma once

#include <cupti.h>

#include <cstdio>

namespace memopt {

constexpr size_t BUF_SIZE = 8 * 1024;
constexpr size_t ALIGN_SIZE = 8;

#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

#define CUPTI_CALL(call)                                                                                       \
  do {                                                                                                         \
    CUptiResult _status = call;                                                                                \
    if (_status != CUPTI_SUCCESS) {                                                                            \
      const char *errstr;                                                                                      \
      cuptiGetResultString(_status, &errstr);                                                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr); \
      exit(-1);                                                                                                \
    }                                                                                                          \
  } while (0)

}  // namespace memopt
