#pragma once
// Minimal shim so imatrix-loader.cpp can be compiled inside guanaco without
// pulling in all of llama.cpp's common infrastructure. Only LOG_ERR is used.
#include <cstdio>
#include <cstdarg>
static inline void llama_log_shim_err(const char * fmt, ...) {
    va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args);
}
#define LOG_ERR(...) llama_log_shim_err(__VA_ARGS__)
