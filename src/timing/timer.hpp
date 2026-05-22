#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdint>
#include <string>

class Timer {
public:
    static std::int64_t now_ns();
    static std::string timestamp_yyyymmdd_hhmmss();
};

#endif
