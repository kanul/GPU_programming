#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <exception>

#include <zlib.h>

struct Sample
{
    uint8_t label_;
    std::vector<int> data_;
};

struct Exception: public std::exception
{
    std::string message_;
    Exception(const std::string& msg): message_(msg) {}
    ~Exception() noexcept(true) {}
};

namespace mnist 
{
    int read(const std::string& images, const std::string& labels, std::vector<Sample>& samples);
}
