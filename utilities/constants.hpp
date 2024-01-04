#pragma once

namespace Constants {
constexpr int DEVICE_ID = 0;
constexpr float PREFETCHING_BANDWIDTH_IN_GBPS = 281.0;
constexpr float PREFETCHING_BANDWIDTH = PREFETCHING_BANDWIDTH_IN_GBPS * 1e9;
constexpr float OFFLOADING_BANDWIDTH = PREFETCHING_BANDWIDTH;
constexpr float ACCEPTABLE_RUNNING_TIME_FACTOR = 2;
};  // namespace Constants
