#include <future>
#include <thread>

namespace memopt {

class PeakMemoryUsageProfiler {
 public:
  PeakMemoryUsageProfiler(int sampleIntervalMilliseconds = 100);
  void start();
  size_t end();

 private:
  std::thread monitorThread;
  std::atomic<bool> stopFlag;
  std::promise<size_t> peakMemoryUsagePromise;
  int sampleIntervalMilliseconds;

  void periodicallyCheckMemoryUsage();
};

}  // namespace memopt
