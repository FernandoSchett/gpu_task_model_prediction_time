#include "timer.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

std::int64_t Timer::now_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string Timer::timestamp_yyyymmdd_hhmmss() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return out.str();
}
