#pragma once
// Minimal shim replacing common/common.h for guanaco's imatrix-prior build.
// imatrix-loader.cpp uses LOG_ERR (provided by log.h) plus the small string
// helpers below. Everything else from the real common.h is intentionally
// omitted to avoid pulling llama.cpp's full common infrastructure (and the
// circular link it would create) into guanaco.
#include "log.h"
#include <string>
#include <string_view>

inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}
