struct Constants {
  static constexpr int DEVICE_ID = 0;
  static constexpr float PREFETCHING_BANDWIDTH_IN_GBPS = 281.0;
  static constexpr float PREFETCHING_BANDWIDTH = PREFETCHING_BANDWIDTH_IN_GBPS * 1e9;
  static constexpr float OFFLOADING_BANDWIDTH = PREFETCHING_BANDWIDTH;
};
